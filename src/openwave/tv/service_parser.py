"""Turning a multiplex's tables into a channel list.

The tables each hold part of the answer. The PAT says which programmes exist, the SDT says what
they are called, the PMTs say what they are made of, and the NIT says what number a viewer types.
This module joins them up.

**A service in one table and not another is the normal case, not an error.** A multiplex is
reconfigured while it is on the air; a scan reads its tables over a second or two and can easily
catch an SDT from before a change and a PAT from after it. So the join is deliberately generous:
a service named in the SDT is listed even if no PMT for it arrived, and a programme in the PAT is
listed even if nothing named it. Dropping either would mean a channel disappearing from the list
for reasons the viewer cannot see.
"""

from __future__ import annotations

from openwave.sdr.device import SignalQuality
from openwave.tv.channel import Channel
from openwave.tv.mux_parser import MuxTables
from openwave.tv.tables import ServiceEntry


def channels_from_tables(
    tables: MuxTables,
    *,
    mux_freq_hz: float,
    quality: SignalQuality | None = None,
    include_scrambled: bool = True,
    include_not_running: bool = False,
) -> tuple[Channel, ...]:
    """Build the channel list for one multiplex.

    Args:
        tables: What was read from the multiplex.
        mux_freq_hz: The frequency it was found on. Taken from the tuner rather than from the
            NIT: the NIT says where a transmitter *claims* to be, and the tuner knows where it
            actually heard it.
        quality: What the tuner reported about the signal, recorded against every channel in
            the multiplex since they share it.
        include_scrambled: Whether to list services that cannot be watched. On by default: a
            list that silently omits them leaves a viewer wondering where a channel went.
        include_not_running: Whether to list services the SDT says are off the air. Off by
            default, because they are usually placeholders for events that have not started.

    Returns:
        Channels in logical channel order where the network publishes numbers, and in service
        identifier order otherwise.
    """
    if mux_freq_hz <= 0:
        raise ValueError(f"multiplex frequency must be positive, got {mux_freq_hz}")

    named = {entry.service_id: entry for entry in tables.sdt.services} if tables.sdt else {}
    programmes = set(tables.pat.programmes) if tables.pat else set()
    transport_stream_id = _transport_stream_id(tables)
    numbers = _logical_channels(tables, transport_stream_id)

    channels: list[Channel] = []
    for service_id in sorted(programmes | set(named)):
        entry = named.get(service_id)
        if entry is not None and not entry.running and not include_not_running:
            continue
        if entry is not None and entry.scrambled and not include_scrambled:
            continue

        pmt = tables.pmts.get(service_id)
        channels.append(
            Channel(
                name=_name_for(service_id, entry),
                service_id=service_id,
                mux_freq_hz=mux_freq_hz,
                transport_stream_id=transport_stream_id,
                service_type=_service_type_for(entry, has_video=bool(pmt and pmt.has_video)),
                logical_channel=numbers.get(service_id),
                provider=(entry.provider or None) if entry else None,
                scrambled=bool(entry.scrambled) if entry else False,
                running=bool(entry.running) if entry else True,
                video_pid=pmt.video_pid if pmt else None,
                audio_pid=pmt.audio_pid if pmt else None,
                snr_db=quality.snr_db if quality else None,
                strength_dbm=quality.strength_dbm if quality else None,
            )
        )

    return tuple(sorted(channels, key=_ordering))


def _name_for(service_id: int, entry: ServiceEntry | None) -> str:
    """What to call a service, falling back to its number.

    A service with no name is still a channel. Calling it by its identifier is less useful than
    a name and far more useful than leaving it out.
    """
    if entry is not None and entry.name:
        return entry.name
    return f"Service {service_id:#06x}"


def _service_type_for(entry: ServiceEntry | None, *, has_video: bool) -> int:
    """The service type to record, inferring one when the SDT did not arrive."""
    if entry is not None and entry.service_type:
        return entry.service_type
    # 0x01 is digital television, 0x02 digital radio. Inferring from the streams present is a
    # guess, but a better one than reporting a type of zero, which means nothing at all.
    return 0x01 if has_video else 0x02


def _ordering(channel: Channel) -> tuple[int, int]:
    """Sort by the number a viewer types, with unnumbered channels after the numbered ones."""
    if channel.logical_channel is None:
        return (1, channel.service_id)
    return (0, channel.logical_channel)


def _logical_channels(tables: MuxTables, transport_stream_id: int) -> dict[int, int]:
    """The channel numbers the NIT gives this multiplex's services.

    Only this multiplex's entry is read when its identifier is known: the NIT describes the
    whole network, and the same service identifier can mean a different channel in another
    multiplex.
    """
    if tables.nit is None:
        return {}
    if tables.pat is None and tables.sdt is None:
        return tables.nit.logical_channels
    return tables.nit.logical_channels_for(transport_stream_id)


def _transport_stream_id(tables: MuxTables) -> int:
    """The multiplex's identifier, from whichever table carried it.

    The PAT and the SDT both state it, and they agree in a healthy stream. The PAT is preferred
    because it is the table a scan is most likely to have received: it is small, sent often, and
    on a fixed PID.
    """
    if tables.pat is not None:
        return tables.pat.transport_stream_id
    if tables.sdt is not None:
        return tables.sdt.transport_stream_id
    return 0
