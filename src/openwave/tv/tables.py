"""The tables that name a multiplex's contents.

Four of them answer the question a TV scan asks -- what channels are here, and what are they
called?

===========  =================================================================================
Table        What it says
===========  =================================================================================
PAT          Which programmes exist in this multiplex, and which PID describes each one
PMT          Which video and audio streams make up one programme
SDT          What each service is called, who runs it, and whether it is encrypted
NIT          What the network is, which frequencies carry it, and the channel numbers
===========  =================================================================================

A PAT alone gives a list of numbers. The names come from the SDT and the channel numbers from
the NIT, which is why a scan reads all four rather than stopping at the first.

**Lengths in these tables come off the air.** Every loop here is bounded by the data actually
present, not by the length a transmitter claims, and a table that runs out mid-entry is reported
as far as it was readable rather than discarded or guessed at.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

from openwave.tv.psi import Section

#: Descriptor tags OpenWave reads.
SERVICE_DESCRIPTOR_TAG: Final = 0x48
SERVICE_LIST_DESCRIPTOR_TAG: Final = 0x41
TERRESTRIAL_DELIVERY_DESCRIPTOR_TAG: Final = 0x5A
LOGICAL_CHANNEL_DESCRIPTOR_TAG: Final = 0x83
NETWORK_NAME_DESCRIPTOR_TAG: Final = 0x40

#: Programme number zero in a PAT points at the Network Information Table, not at a programme.
NETWORK_PROGRAMME_NUMBER: Final = 0x0000

#: Character set selectors from EN 300 468 Annex A that OpenWave understands.
_CHARSET_SELECTORS: Final = {
    0x01: "iso8859-5",
    0x02: "iso8859-6",
    0x03: "iso8859-7",
    0x04: "iso8859-8",
    0x05: "iso8859-9",
    0x06: "iso8859-10",
    0x07: "iso8859-11",
    0x09: "iso8859-13",
    0x0A: "iso8859-14",
    0x0B: "iso8859-15",
    0x11: "utf-16-be",
    0x15: "utf-8",
}


class ServiceType(IntEnum):
    """What kind of service an SDT entry describes.

    From EN 300 468. Only the types a scan needs to tell television from radio are named; the
    rest keep their number, because the list is long, country-specific in places, and still
    growing.
    """

    DIGITAL_TELEVISION = 0x01
    DIGITAL_RADIO = 0x02
    TELETEXT = 0x03
    NVOD_REFERENCE = 0x04
    NVOD_TIME_SHIFTED = 0x05
    MOSAIC = 0x06
    FM_RADIO = 0x07
    DATA_BROADCAST = 0x0C
    ADVANCED_CODEC_RADIO = 0x0A
    MPEG2_HD_TELEVISION = 0x11
    ADVANCED_CODEC_SD_TELEVISION = 0x16
    ADVANCED_CODEC_HD_TELEVISION = 0x19
    HEVC_TELEVISION = 0x1F

    @property
    def is_television(self) -> bool:
        """Whether this service type carries pictures."""
        return self in {
            ServiceType.DIGITAL_TELEVISION,
            ServiceType.MPEG2_HD_TELEVISION,
            ServiceType.ADVANCED_CODEC_SD_TELEVISION,
            ServiceType.ADVANCED_CODEC_HD_TELEVISION,
            ServiceType.HEVC_TELEVISION,
        }

    @property
    def is_radio(self) -> bool:
        """Whether this service type carries sound only."""
        return self in {
            ServiceType.DIGITAL_RADIO,
            ServiceType.FM_RADIO,
            ServiceType.ADVANCED_CODEC_RADIO,
        }


@dataclass(frozen=True, slots=True)
class Descriptor:
    """One descriptor: a tag and its bytes.

    Descriptors are how DVB extends its tables without changing them. A parser reads the tags it
    knows and steps over the rest by their length, which is what lets an old receiver cope with
    a stream carrying something it has never heard of.
    """

    tag: int
    data: bytes

    def __str__(self) -> str:
        return f"descriptor {self.tag:#04x} ({len(self.data)}B)"


def parse_descriptors(data: bytes) -> tuple[Descriptor, ...]:
    """Read a descriptor loop.

    Stops at the first entry that does not fit rather than reading past the end. A truncated
    loop is a damaged stream, and the descriptors before the damage are still good.
    """
    descriptors: list[Descriptor] = []
    offset = 0
    while offset + 2 <= len(data):
        tag = data[offset]
        length = data[offset + 1]
        start = offset + 2
        if start + length > len(data):
            break
        descriptors.append(Descriptor(tag=tag, data=data[start : start + length]))
        offset = start + length
    return tuple(descriptors)


def find_descriptor(descriptors: tuple[Descriptor, ...], tag: int) -> Descriptor | None:
    """The first descriptor with this tag, or ``None``."""
    return next((item for item in descriptors if item.tag == tag), None)


def dvb_text(data: bytes) -> str:
    """Decode a DVB text field.

    DVB text may begin with a byte selecting a character set; without one, the default is
    ISO 6937. That encoding has no Python codec, and it differs from Latin-1 mainly in how it
    writes accents -- as a prefix byte followed by the base letter. Latin-1 stands in for it,
    which is exact for unaccented text and drops the accents otherwise.

    Control codes in the 0x80 to 0x9F range are DVB's own; 0x8A means a line break and becomes a
    space, since a station name is going on one line.

    >>> dvb_text(b"BBC ONE")
    'BBC ONE'
    >>> dvb_text(b"\\x15Caf\\xc3\\xa9")
    'Café'
    """
    if not data:
        return ""

    encoding = "latin-1"
    body = data
    selector = data[0]
    if selector in _CHARSET_SELECTORS:
        encoding = _CHARSET_SELECTORS[selector]
        body = data[1:]
    elif selector == 0x10 and len(data) >= 3:
        # 0x10 followed by two bytes naming an ISO 8859 table.
        table = struct.unpack(">H", data[1:3])[0]
        encoding = f"iso8859-{table}"
        body = data[3:]
    elif selector < 0x20:
        # A selector OpenWave does not know. Skipping it beats decoding it as a character.
        body = data[1:]

    try:
        text = body.decode(encoding, errors="replace")
    except LookupError:
        text = body.decode("latin-1", errors="replace")

    # Strip DVB's own control codes, which are not characters.
    cleaned = "".join(" " if 0x80 <= ord(ch) <= 0x9F or ord(ch) < 0x20 else ch for ch in text)
    return cleaned.strip()


# -- Program Association Table -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pat:
    """The Program Association Table: which programmes a multiplex carries.

    Attributes:
        transport_stream_id: Identifier of this multiplex.
        programmes: Programme number to the PID of its PMT.
        network_pid: PID of the Network Information Table, if the PAT names one.
    """

    transport_stream_id: int
    programmes: dict[int, int] = field(default_factory=dict)
    network_pid: int | None = None

    def __str__(self) -> str:
        return (
            f"PAT for multiplex {self.transport_stream_id:#06x}: {len(self.programmes)} programmes"
        )


def parse_pat(section: Section) -> Pat:
    """Read a PAT section.

    Entries are four bytes each: a programme number and the PID that describes it. Programme
    number zero is not a programme but a pointer to the Network Information Table.
    """
    programmes: dict[int, int] = {}
    network_pid: int | None = None

    payload = section.payload
    for offset in range(0, len(payload) - 3, 4):
        number = struct.unpack(">H", payload[offset : offset + 2])[0]
        pid = struct.unpack(">H", payload[offset + 2 : offset + 4])[0] & 0x1FFF
        if number == NETWORK_PROGRAMME_NUMBER:
            network_pid = pid
        else:
            programmes[number] = pid

    return Pat(
        transport_stream_id=section.extension,
        programmes=programmes,
        network_pid=network_pid,
    )


# -- Program Map Table ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ElementaryStream:
    """One video, audio or data stream within a programme."""

    stream_type: int
    pid: int
    descriptors: tuple[Descriptor, ...] = ()

    @property
    def is_video(self) -> bool:
        """Whether this stream carries pictures."""
        return self.stream_type in _VIDEO_STREAM_TYPES

    @property
    def is_audio(self) -> bool:
        """Whether this stream carries sound.

        DVB does not give AC-3, E-AC-3 or AC-4 a stream type of their own: they travel as PES
        private data, type 0x06, and a descriptor says which codec it is. The same type also
        carries teletext and subtitles, so the descriptor is what decides.
        """
        if self.stream_type in _AUDIO_STREAM_TYPES:
            return True
        if self.stream_type != PES_PRIVATE_DATA_STREAM_TYPE:
            return False
        return any(_is_audio_descriptor(descriptor) for descriptor in self.descriptors)

    def __str__(self) -> str:
        kind = "video" if self.is_video else "audio" if self.is_audio else "data"
        return f"{kind} PID {self.pid:#06x} type {self.stream_type:#04x}"


#: Stream types that carry pictures, from ISO/IEC 13818-1 and its amendments.
_VIDEO_STREAM_TYPES: Final = frozenset(
    {
        0x01,  # MPEG-1 video
        0x02,  # MPEG-2 video
        0x10,  # MPEG-4 part 2
        0x1B,  # H.264
        0x24,  # HEVC
        0x33,  # VVC
    }
)

#: Stream types that carry sound.
_AUDIO_STREAM_TYPES: Final = frozenset(
    {
        0x03,  # MPEG-1 audio
        0x04,  # MPEG-2 audio
        0x0F,  # AAC in ADTS
        0x11,  # AAC in LATM
        0x81,  # AC-3, as used in the Americas
        0x87,  # Enhanced AC-3
    }
)


#: Stream type of PES private data, under which DVB carries AC-3, E-AC-3 and AC-4.
PES_PRIVATE_DATA_STREAM_TYPE: Final = 0x06

#: Descriptors that mark a private data stream as audio, from EN 300 468 Annex D.
_AUDIO_DESCRIPTOR_TAGS: Final = frozenset(
    {
        0x6A,  # AC-3
        0x7A,  # Enhanced AC-3
        0x7B,  # DTS
        0x7C,  # AAC
    }
)

#: The extension descriptor, whose first byte says what it extends.
EXTENSION_DESCRIPTOR_TAG: Final = 0x7F

#: Extension tags that mark a private data stream as audio.
_AUDIO_EXTENSION_TAGS: Final = frozenset(
    {
        0x0E,  # DTS-HD
        0x15,  # AC-4
    }
)


def _is_audio_descriptor(descriptor: Descriptor) -> bool:
    """Whether a descriptor says the private data stream it describes is audio."""
    if descriptor.tag in _AUDIO_DESCRIPTOR_TAGS:
        return True
    return (
        descriptor.tag == EXTENSION_DESCRIPTOR_TAG
        and bool(descriptor.data)
        and descriptor.data[0] in _AUDIO_EXTENSION_TAGS
    )


@dataclass(frozen=True, slots=True)
class Pmt:
    """The Program Map Table: what one programme is made of."""

    programme_number: int
    pcr_pid: int
    streams: tuple[ElementaryStream, ...] = ()
    descriptors: tuple[Descriptor, ...] = ()

    @property
    def has_video(self) -> bool:
        """Whether this programme carries pictures, which tells television from radio."""
        return any(stream.is_video for stream in self.streams)

    @property
    def video_pid(self) -> int | None:
        """PID of the first video stream, if there is one."""
        return next((stream.pid for stream in self.streams if stream.is_video), None)

    @property
    def audio_pid(self) -> int | None:
        """PID of the first audio stream, if there is one."""
        return next((stream.pid for stream in self.streams if stream.is_audio), None)

    def __str__(self) -> str:
        return f"PMT for programme {self.programme_number}: {len(self.streams)} streams"


def parse_pmt(section: Section) -> Pmt:
    """Read a PMT section."""
    payload = section.payload
    if len(payload) < 4:
        return Pmt(programme_number=section.extension, pcr_pid=0x1FFF)

    pcr_pid = struct.unpack(">H", payload[0:2])[0] & 0x1FFF
    programme_info_length = struct.unpack(">H", payload[2:4])[0] & 0x0FFF
    offset = 4
    if offset + programme_info_length > len(payload):
        return Pmt(programme_number=section.extension, pcr_pid=pcr_pid)
    descriptors = parse_descriptors(payload[offset : offset + programme_info_length])
    offset += programme_info_length

    streams: list[ElementaryStream] = []
    while offset + 5 <= len(payload):
        stream_type = payload[offset]
        pid = struct.unpack(">H", payload[offset + 1 : offset + 3])[0] & 0x1FFF
        info_length = struct.unpack(">H", payload[offset + 3 : offset + 5])[0] & 0x0FFF
        start = offset + 5
        if start + info_length > len(payload):
            break
        streams.append(
            ElementaryStream(
                stream_type=stream_type,
                pid=pid,
                descriptors=parse_descriptors(payload[start : start + info_length]),
            )
        )
        offset = start + info_length

    return Pmt(
        programme_number=section.extension,
        pcr_pid=pcr_pid,
        streams=tuple(streams),
        descriptors=descriptors,
    )


# -- Service Description Table -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceEntry:
    """One service's entry in the SDT: what it is called and what kind it is.

    Attributes:
        service_id: Identifier, the same number the PAT calls a programme number.
        name: What the service is called.
        provider: Who runs it.
        service_type: What kind of service it is, from :class:`ServiceType`.
        scrambled: Whether access is controlled. OpenWave reports this and stops; see
            docs/legal.md.
        running: Whether the service is on the air now.
    """

    service_id: int
    name: str = ""
    provider: str = ""
    service_type: int = 0
    scrambled: bool = False
    running: bool = True

    @property
    def is_television(self) -> bool:
        """Whether this service carries pictures."""
        try:
            return ServiceType(self.service_type).is_television
        except ValueError:
            return False

    @property
    def is_radio(self) -> bool:
        """Whether this service carries sound only."""
        try:
            return ServiceType(self.service_type).is_radio
        except ValueError:
            return False

    def __str__(self) -> str:
        label = self.name or f"service {self.service_id:#06x}"
        suffix = " (scrambled)" if self.scrambled else ""
        return f"{label}{suffix}"


@dataclass(frozen=True, slots=True)
class Sdt:
    """The Service Description Table: the names behind the numbers."""

    transport_stream_id: int
    original_network_id: int = 0
    services: tuple[ServiceEntry, ...] = ()

    def __str__(self) -> str:
        return f"SDT for multiplex {self.transport_stream_id:#06x}: {len(self.services)} services"


def parse_sdt(section: Section) -> Sdt:
    """Read an SDT section."""
    payload = section.payload
    if len(payload) < 3:
        return Sdt(transport_stream_id=section.extension)

    original_network_id = struct.unpack(">H", payload[0:2])[0]
    offset = 3  # the third byte is reserved

    services: list[ServiceEntry] = []
    while offset + 5 <= len(payload):
        service_id = struct.unpack(">H", payload[offset : offset + 2])[0]
        flags = payload[offset + 3]
        running_status = (flags >> 5) & 0x07
        free_ca_mode = bool((flags >> 4) & 0x01)
        descriptors_length = struct.unpack(">H", payload[offset + 3 : offset + 5])[0] & 0x0FFF
        start = offset + 5
        if start + descriptors_length > len(payload):
            break

        descriptors = parse_descriptors(payload[start : start + descriptors_length])
        name, provider, service_type = _read_service_descriptor(descriptors)
        services.append(
            ServiceEntry(
                service_id=service_id,
                name=name,
                provider=provider,
                service_type=service_type,
                scrambled=free_ca_mode,
                # Running status 4 means running; 1 means not running, and 0 is undefined,
                # which in practice means running.
                running=running_status in (0, 4),
            )
        )
        offset = start + descriptors_length

    return Sdt(
        transport_stream_id=section.extension,
        original_network_id=original_network_id,
        services=tuple(services),
    )


def _read_service_descriptor(
    descriptors: tuple[Descriptor, ...],
) -> tuple[str, str, int]:
    """Pull the name, provider and type out of a service descriptor."""
    descriptor = find_descriptor(descriptors, SERVICE_DESCRIPTOR_TAG)
    if descriptor is None or not descriptor.data:
        return "", "", 0

    data = descriptor.data
    service_type = data[0]
    offset = 1
    if offset >= len(data):
        return "", "", service_type

    provider_length = data[offset]
    offset += 1
    if offset + provider_length > len(data):
        return "", "", service_type
    provider = dvb_text(data[offset : offset + provider_length])
    offset += provider_length

    if offset >= len(data):
        return "", provider, service_type
    name_length = data[offset]
    offset += 1
    if offset + name_length > len(data):
        return "", provider, service_type
    name = dvb_text(data[offset : offset + name_length])
    return name, provider, service_type


# -- Network Information Table -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransportStreamEntry:
    """One multiplex as the NIT describes it.

    Attributes:
        transport_stream_id: Identifier of the multiplex.
        original_network_id: The network it originated on.
        centre_frequency_hz: Where it is transmitted, if the NIT says.
        bandwidth_hz: Channel bandwidth, if the NIT says.
        logical_channels: Service identifier to the number a viewer types, where the network
            publishes them. This descriptor is a private extension rather than part of the
            standard, but it is used across Europe and is the only source of the numbers.
        service_types: Service identifier to its type, from the service list descriptor.
    """

    transport_stream_id: int
    original_network_id: int
    centre_frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    logical_channels: dict[int, int] = field(default_factory=dict)
    service_types: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Nit:
    """The Network Information Table: the network, its frequencies and its channel numbers."""

    network_id: int
    network_name: str = ""
    transport_streams: tuple[TransportStreamEntry, ...] = ()

    @property
    def logical_channels(self) -> dict[int, int]:
        """Every logical channel number in the network, across all its multiplexes."""
        numbers: dict[int, int] = {}
        for entry in self.transport_streams:
            numbers.update(entry.logical_channels)
        return numbers

    def __str__(self) -> str:
        label = self.network_name or f"network {self.network_id:#06x}"
        return f"NIT for {label}: {len(self.transport_streams)} multiplexes"


#: Bandwidth codes in the terrestrial delivery system descriptor, in Hz.
_BANDWIDTH_CODES: Final = {0: 8e6, 1: 7e6, 2: 6e6, 3: 5e6}


def parse_nit(section: Section) -> Nit:
    """Read a NIT section."""
    payload = section.payload
    if len(payload) < 2:
        return Nit(network_id=section.extension)

    network_descriptors_length = struct.unpack(">H", payload[0:2])[0] & 0x0FFF
    offset = 2
    if offset + network_descriptors_length > len(payload):
        return Nit(network_id=section.extension)
    network_descriptors = parse_descriptors(payload[offset : offset + network_descriptors_length])
    offset += network_descriptors_length

    name_descriptor = find_descriptor(network_descriptors, NETWORK_NAME_DESCRIPTOR_TAG)
    network_name = dvb_text(name_descriptor.data) if name_descriptor else ""

    if offset + 2 > len(payload):
        return Nit(network_id=section.extension, network_name=network_name)
    loop_length = struct.unpack(">H", payload[offset : offset + 2])[0] & 0x0FFF
    offset += 2
    end = min(offset + loop_length, len(payload))

    entries: list[TransportStreamEntry] = []
    while offset + 6 <= end:
        transport_stream_id = struct.unpack(">H", payload[offset : offset + 2])[0]
        original_network_id = struct.unpack(">H", payload[offset + 2 : offset + 4])[0]
        descriptors_length = struct.unpack(">H", payload[offset + 4 : offset + 6])[0] & 0x0FFF
        start = offset + 6
        if start + descriptors_length > end:
            break
        descriptors = parse_descriptors(payload[start : start + descriptors_length])
        entries.append(
            _transport_stream_entry(transport_stream_id, original_network_id, descriptors)
        )
        offset = start + descriptors_length

    return Nit(
        network_id=section.extension,
        network_name=network_name,
        transport_streams=tuple(entries),
    )


def _transport_stream_entry(
    transport_stream_id: int,
    original_network_id: int,
    descriptors: tuple[Descriptor, ...],
) -> TransportStreamEntry:
    """Assemble one NIT loop entry from its descriptors."""
    centre_frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    logical_channels: dict[int, int] = {}
    service_types: dict[int, int] = {}

    delivery = find_descriptor(descriptors, TERRESTRIAL_DELIVERY_DESCRIPTOR_TAG)
    if delivery is not None and len(delivery.data) >= 5:
        # The frequency is in units of 10 Hz, which is why a 474 MHz channel reads as 47 400 000.
        centre_frequency_hz = float(struct.unpack(">I", delivery.data[0:4])[0]) * 10.0
        bandwidth_hz = _BANDWIDTH_CODES.get((delivery.data[4] >> 5) & 0x07)

    service_list = find_descriptor(descriptors, SERVICE_LIST_DESCRIPTOR_TAG)
    if service_list is not None:
        data = service_list.data
        for position in range(0, len(data) - 2, 3):
            service_id = struct.unpack(">H", data[position : position + 2])[0]
            service_types[service_id] = data[position + 2]

    channel_list = find_descriptor(descriptors, LOGICAL_CHANNEL_DESCRIPTOR_TAG)
    if channel_list is not None:
        data = channel_list.data
        for position in range(0, len(data) - 3, 4):
            service_id = struct.unpack(">H", data[position : position + 2])[0]
            number = struct.unpack(">H", data[position + 2 : position + 4])[0] & 0x03FF
            logical_channels[service_id] = number

    return TransportStreamEntry(
        transport_stream_id=transport_stream_id,
        original_network_id=original_network_id,
        centre_frequency_hz=centre_frequency_hz,
        bandwidth_hz=bandwidth_hz,
        logical_channels=logical_channels,
        service_types=service_types,
    )
