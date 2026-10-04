"""Tests for the tables that name a multiplex's contents.

Every length in these tables comes off the air, so as well as parsing well-formed tables these
tests feed the parsers loops that run past the end of their data, which is what a damaged or
hostile stream looks like.
"""

from __future__ import annotations

import pytest

from openwave.tv.psi import Section
from openwave.tv.tables import (
    LOGICAL_CHANNEL_DESCRIPTOR_TAG,
    NETWORK_NAME_DESCRIPTOR_TAG,
    SERVICE_DESCRIPTOR_TAG,
    Descriptor,
    ServiceType,
    dvb_text,
    find_descriptor,
    parse_descriptors,
    parse_nit,
    parse_pat,
    parse_pmt,
    parse_sdt,
)


def section(payload: bytes, *, table_id: int = 0x00, extension: int = 0x1234) -> Section:
    """A section carrying a given payload, for feeding one table parser."""
    return Section(table_id=table_id, payload=payload, extension=extension)


class TestDescriptors:
    def test_a_loop_is_read(self) -> None:
        data = bytes([0x48, 0x02, 0xAA, 0xBB, 0x41, 0x01, 0xCC])
        descriptors = parse_descriptors(data)
        assert [item.tag for item in descriptors] == [0x48, 0x41]
        assert descriptors[0].data == b"\xaa\xbb"
        assert descriptors[1].data == b"\xcc"

    def test_an_empty_loop_gives_nothing(self) -> None:
        assert parse_descriptors(b"") == ()

    def test_a_descriptor_claiming_more_than_is_there_stops_the_loop(self) -> None:
        # The length comes off the air. Reading past the end would hand a parser bytes from
        # whatever follows in memory.
        data = bytes([0x48, 0x02, 0xAA, 0xBB, 0x41, 0x50, 0xCC])
        descriptors = parse_descriptors(data)
        assert [item.tag for item in descriptors] == [0x48]

    def test_a_truncated_header_stops_the_loop(self) -> None:
        assert parse_descriptors(bytes([0x48])) == ()

    def test_a_descriptor_is_found_by_tag(self) -> None:
        descriptors = (Descriptor(0x48, b"x"), Descriptor(0x41, b"y"))
        found = find_descriptor(descriptors, 0x41)
        assert found is not None
        assert found.data == b"y"

    def test_an_absent_tag_gives_none(self) -> None:
        assert find_descriptor((Descriptor(0x48, b"x"),), 0x99) is None

    def test_it_reads_sensibly_when_printed(self) -> None:
        assert "0x48" in str(Descriptor(0x48, b"xy"))


class TestDvbText:
    def test_plain_text_is_read(self) -> None:
        assert dvb_text(b"BBC ONE") == "BBC ONE"

    def test_empty_text_is_empty(self) -> None:
        assert dvb_text(b"") == ""

    def test_a_utf8_selector_is_honoured(self) -> None:
        assert dvb_text(b"\x15Caf\xc3\xa9") == "Café"

    def test_a_latin_selector_is_honoured(self) -> None:
        assert dvb_text(b"\x0bCaf\xe9") == "Café"

    def test_dvb_control_codes_become_spaces(self) -> None:
        # 0x8A means a line break, which is not a character. A station name goes on one line.
        assert dvb_text(b"BBC\x8aONE") == "BBC ONE"

    def test_an_unknown_selector_is_skipped_rather_than_decoded(self) -> None:
        # Decoding a selector as a character would put a stray glyph at the start of a name.
        assert dvb_text(b"\x1fNAME") == "NAME"

    def test_surrounding_whitespace_is_trimmed(self) -> None:
        assert dvb_text(b"  BBC ONE  ") == "BBC ONE"

    def test_an_unknown_encoding_falls_back_rather_than_failing(self) -> None:
        assert dvb_text(b"\x10\xff\xffNAME") != ""


class TestPat:
    def test_programmes_and_their_pmt_pids_are_read(self) -> None:
        payload = bytes([0x00, 0x01, 0xE1, 0x00, 0x00, 0x02, 0xE1, 0x01])
        pat = parse_pat(section(payload, extension=0x1000))
        assert pat.transport_stream_id == 0x1000
        assert pat.programmes == {1: 0x0100, 2: 0x0101}

    def test_programme_zero_is_the_network_table_not_a_programme(self) -> None:
        payload = bytes([0x00, 0x00, 0xE0, 0x10, 0x00, 0x01, 0xE1, 0x00])
        pat = parse_pat(section(payload))
        assert pat.network_pid == 0x0010
        assert pat.programmes == {1: 0x0100}

    def test_an_empty_pat_is_allowed(self) -> None:
        # A multiplex can be on the air with nothing in it, during a reconfiguration.
        pat = parse_pat(section(b""))
        assert pat.programmes == {}
        assert pat.network_pid is None

    def test_a_trailing_partial_entry_is_ignored(self) -> None:
        payload = bytes([0x00, 0x01, 0xE1, 0x00, 0x00, 0x02])
        assert parse_pat(section(payload)).programmes == {1: 0x0100}

    def test_the_pid_is_masked_to_thirteen_bits(self) -> None:
        # The top three bits are reserved and set to one in real streams. Leaving them in gives
        # a PID that cannot exist.
        payload = bytes([0x00, 0x01, 0xFF, 0xFF])
        assert parse_pat(section(payload)).programmes == {1: 0x1FFF}

    def test_it_reads_sensibly_when_printed(self) -> None:
        pat = parse_pat(section(bytes([0x00, 0x01, 0xE1, 0x00]), extension=0x1000))
        assert "0x1000" in str(pat)


class TestPmt:
    def test_streams_are_read(self) -> None:
        payload = bytes(
            [
                0xE2,
                0x00,  # PCR PID 0x0200
                0xF0,
                0x00,  # no programme descriptors
                0x1B,
                0xE2,
                0x00,
                0xF0,
                0x00,  # H.264 video on 0x0200
                0x0F,
                0xE3,
                0x00,
                0xF0,
                0x00,  # AAC audio on 0x0300
            ]
        )
        pmt = parse_pmt(section(payload, table_id=0x02, extension=1))
        assert pmt.programme_number == 1
        assert pmt.pcr_pid == 0x0200
        assert len(pmt.streams) == 2
        assert pmt.has_video
        assert pmt.video_pid == 0x0200
        assert pmt.audio_pid == 0x0300

    def test_a_radio_programme_has_no_video(self) -> None:
        payload = bytes([0xE3, 0x00, 0xF0, 0x00, 0x0F, 0xE3, 0x00, 0xF0, 0x00])
        pmt = parse_pmt(section(payload, table_id=0x02, extension=4))
        assert not pmt.has_video
        assert pmt.video_pid is None
        assert pmt.audio_pid == 0x0300

    @pytest.mark.parametrize(
        "descriptor",
        [
            bytes([0x6A, 0x01, 0x00]),  # AC-3
            bytes([0x7A, 0x01, 0x00]),  # Enhanced AC-3
            bytes([0x7F, 0x01, 0x15]),  # extension: AC-4
        ],
        ids=["ac3", "eac3", "ac4"],
    )
    def test_dvb_audio_carried_as_private_data_is_audio(self, descriptor: bytes) -> None:
        # DVB sends AC-3 and its successors as stream type 0x06, identified by a descriptor,
        # not under the ATSC stream types. A radio service whose only sound is AC-3 would
        # otherwise have no audio stream and be listed as unplayable.
        payload = (
            bytes([0xE3, 0x00, 0xF0, 0x00, 0x06, 0xE3, 0x00, 0xF0, len(descriptor)]) + descriptor
        )
        pmt = parse_pmt(section(payload, table_id=0x02, extension=4))
        assert pmt.streams[0].is_audio
        assert pmt.audio_pid == 0x0300

    def test_private_data_without_an_audio_descriptor_is_not_audio(self) -> None:
        # The same stream type carries teletext (0x56) and subtitles (0x59).
        payload = bytes([0xE3, 0x00, 0xF0, 0x00, 0x06, 0xE3, 0x00, 0xF0, 0x02, 0x56, 0x00])
        pmt = parse_pmt(section(payload, table_id=0x02, extension=4))
        assert not pmt.streams[0].is_audio
        assert pmt.audio_pid is None

    def test_programme_descriptors_are_read(self) -> None:
        payload = bytes([0xE2, 0x00, 0xF0, 0x03, 0x09, 0x01, 0xAA])
        pmt = parse_pmt(section(payload, table_id=0x02))
        assert [item.tag for item in pmt.descriptors] == [0x09]

    def test_a_truncated_pmt_gives_what_was_readable(self) -> None:
        pmt = parse_pmt(section(b"\xe2", table_id=0x02, extension=1))
        assert pmt.programme_number == 1
        assert pmt.streams == ()

    def test_a_descriptor_length_running_past_the_end_stops_the_loop(self) -> None:
        payload = bytes([0xE2, 0x00, 0xF0, 0x00, 0x1B, 0xE2, 0x00, 0xF0, 0x50])
        assert parse_pmt(section(payload, table_id=0x02)).streams == ()

    @pytest.mark.parametrize("stream_type", [0x01, 0x02, 0x1B, 0x24])
    def test_video_stream_types_are_recognised(self, stream_type: int) -> None:
        payload = bytes([0xE2, 0x00, 0xF0, 0x00, stream_type, 0xE2, 0x00, 0xF0, 0x00])
        assert parse_pmt(section(payload, table_id=0x02)).has_video

    @pytest.mark.parametrize("stream_type", [0x03, 0x04, 0x0F, 0x81])
    def test_audio_stream_types_are_recognised(self, stream_type: int) -> None:
        payload = bytes([0xE3, 0x00, 0xF0, 0x00, stream_type, 0xE3, 0x00, 0xF0, 0x00])
        pmt = parse_pmt(section(payload, table_id=0x02))
        assert not pmt.has_video
        assert pmt.audio_pid == 0x0300


def service_descriptor(name: str, provider: str, service_type: int) -> bytes:
    """A service descriptor, as the SDT carries it."""
    body = (
        bytes([service_type, len(provider)])
        + provider.encode()
        + bytes([len(name)])
        + name.encode()
    )
    return bytes([SERVICE_DESCRIPTOR_TAG, len(body)]) + body


class TestSdt:
    def test_a_service_name_and_provider_are_read(self) -> None:
        descriptor = service_descriptor("BBC ONE", "BBC", 0x01)
        payload = bytes([0x20, 0x00, 0xFF])  # original network id, reserved
        payload += bytes(
            [0x00, 0x01, 0xFC, 0x80 | ((len(descriptor) >> 8) & 0x0F), len(descriptor) & 0xFF]
        )
        payload += descriptor

        sdt = parse_sdt(section(payload, table_id=0x42, extension=0x1000))
        assert sdt.transport_stream_id == 0x1000
        assert sdt.original_network_id == 0x2000
        assert len(sdt.services) == 1
        entry = sdt.services[0]
        assert entry.service_id == 1
        assert entry.name == "BBC ONE"
        assert entry.provider == "BBC"
        assert entry.is_television

    def test_a_scrambled_service_is_marked(self) -> None:
        descriptor = service_descriptor("Premium", "Someone", 0x01)
        flags = 0x80 | 0x10  # running, access controlled
        payload = bytes([0x20, 0x00, 0xFF, 0x00, 0x05, 0xFC])
        payload += bytes([flags | ((len(descriptor) >> 8) & 0x0F), len(descriptor) & 0xFF])
        payload += descriptor
        entry = parse_sdt(section(payload, table_id=0x42)).services[0]
        assert entry.scrambled

    def test_a_radio_service_is_recognised(self) -> None:
        descriptor = service_descriptor("Radio 4", "BBC", int(ServiceType.DIGITAL_RADIO))
        payload = bytes([0x20, 0x00, 0xFF, 0x00, 0x04, 0xFC, 0x80, len(descriptor)])
        payload += descriptor
        entry = parse_sdt(section(payload, table_id=0x42)).services[0]
        assert entry.is_radio
        assert not entry.is_television

    def test_a_service_with_no_descriptor_still_appears(self) -> None:
        payload = bytes([0x20, 0x00, 0xFF, 0x00, 0x07, 0xFC, 0x80, 0x00])
        entry = parse_sdt(section(payload, table_id=0x42)).services[0]
        assert entry.service_id == 7
        assert entry.name == ""

    def test_a_truncated_sdt_gives_nothing_rather_than_guesses(self) -> None:
        assert parse_sdt(section(b"\x20", table_id=0x42)).services == ()

    def test_a_descriptor_loop_running_past_the_end_stops_the_loop(self) -> None:
        payload = bytes([0x20, 0x00, 0xFF, 0x00, 0x01, 0xFC, 0x80, 0x50])
        assert parse_sdt(section(payload, table_id=0x42)).services == ()

    def test_an_unknown_service_type_is_neither_television_nor_radio(self) -> None:
        # The list is long and still growing, so an unrecognised number must not be guessed at.
        descriptor = service_descriptor("Mystery", "X", 0xEE)
        payload = bytes([0x20, 0x00, 0xFF, 0x00, 0x09, 0xFC, 0x80, len(descriptor)])
        payload += descriptor
        entry = parse_sdt(section(payload, table_id=0x42)).services[0]
        assert not entry.is_television
        assert not entry.is_radio


class TestNit:
    def build_nit_payload(self) -> bytes:
        name = b"Freeview"
        network_descriptors = bytes([NETWORK_NAME_DESCRIPTOR_TAG, len(name)]) + name

        delivery = (
            bytes([0x5A, 0x0B])
            + (47_400_000).to_bytes(4, "big")
            + bytes([0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
        )
        channels = bytes([0x00, 0x01, 0xFC, 0x01, 0x00, 0x02, 0xFC, 0x02])
        channel_descriptor = bytes([LOGICAL_CHANNEL_DESCRIPTOR_TAG, len(channels)]) + channels
        transport_descriptors = delivery + channel_descriptor

        loop = (
            (0x1000).to_bytes(2, "big")
            + (0x3000).to_bytes(2, "big")
            + (0xF000 | len(transport_descriptors)).to_bytes(2, "big")
            + transport_descriptors
        )
        return (
            (0xF000 | len(network_descriptors)).to_bytes(2, "big")
            + network_descriptors
            + (0xF000 | len(loop)).to_bytes(2, "big")
            + loop
        )

    def test_the_network_name_is_read(self) -> None:
        nit = parse_nit(section(self.build_nit_payload(), table_id=0x40, extension=0x2000))
        assert nit.network_id == 0x2000
        assert nit.network_name == "Freeview"

    def test_the_frequency_is_in_units_of_ten_hertz(self) -> None:
        # Which is why a 474 MHz channel reads as 47 400 000 in the descriptor. Treating it as
        # hertz would report the transmitter at 47 MHz.
        nit = parse_nit(section(self.build_nit_payload(), table_id=0x40))
        entry = nit.transport_streams[0]
        assert entry.centre_frequency_hz == 474_000_000.0

    def test_the_bandwidth_code_is_decoded(self) -> None:
        nit = parse_nit(section(self.build_nit_payload(), table_id=0x40))
        assert nit.transport_streams[0].bandwidth_hz == 8e6

    def test_logical_channel_numbers_are_read(self) -> None:
        nit = parse_nit(section(self.build_nit_payload(), table_id=0x40))
        assert nit.logical_channels == {1: 1, 2: 2}

    def test_a_truncated_nit_gives_what_was_readable(self) -> None:
        nit = parse_nit(section(b"\xf0", table_id=0x40, extension=0x2000))
        assert nit.network_id == 0x2000
        assert nit.transport_streams == ()

    def test_it_reads_sensibly_when_printed(self) -> None:
        nit = parse_nit(section(self.build_nit_payload(), table_id=0x40))
        assert "Freeview" in str(nit)


class TestServiceType:
    def test_television_types_are_grouped(self) -> None:
        assert ServiceType.DIGITAL_TELEVISION.is_television
        assert ServiceType.ADVANCED_CODEC_HD_TELEVISION.is_television
        assert ServiceType.HEVC_TELEVISION.is_television
        assert not ServiceType.DIGITAL_RADIO.is_television

    def test_radio_types_are_grouped(self) -> None:
        assert ServiceType.DIGITAL_RADIO.is_radio
        assert ServiceType.ADVANCED_CODEC_RADIO.is_radio
        assert not ServiceType.DIGITAL_TELEVISION.is_radio

    def test_data_is_neither(self) -> None:
        assert not ServiceType.DATA_BROADCAST.is_television
        assert not ServiceType.DATA_BROADCAST.is_radio
