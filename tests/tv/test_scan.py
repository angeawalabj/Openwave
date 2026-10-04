"""End-to-end tests for the television scan.

These are the tests that decide whether OpenWave reports the right channels. They work the way
the whole project works: build a multiplex whose contents are known, transmit it through a
simulated tuner, and assert the scan recovers exactly that.
"""

from __future__ import annotations

import pytest

from openwave.sdr.device import TuningRange
from openwave.sdr.mock import MockDvbDevice, SyntheticMux
from openwave.tv.band import UHF_PLAN, VHF_BAND_III_PLAN
from openwave.tv.channel import Channel, ChannelKind
from openwave.tv.demo import DEMO_MULTIPLEXES, demo_tuner
from openwave.tv.dvb_scanner import DvbScanSettings, scan_channel, scan_dvb
from openwave.tv.mux_parser import MuxParser, parse_mux
from openwave.tv.service_parser import channels_from_tables
from openwave.tv.synthesis import (
    SyntheticMultiplex,
    SyntheticService,
    build_transport_stream,
    demo_multiplex,
)
from openwave.tv.tables import Nit, ServiceType, TransportStreamEntry
from openwave.tv.ts import iter_packets

#: Quick settings: the simulator locks instantly, so a long timeout only slows the tests.
QUICK = DvbScanSettings(lock_timeout_s=0.1)


class TestSynthesisRoundTrip:
    def test_every_service_comes_back(self) -> None:
        multiplex = demo_multiplex()
        tables = parse_mux(build_transport_stream(multiplex, repeats=2))
        assert tables.is_complete
        expected = {service.service_id: service.name for service in multiplex.services}
        assert tables.sdt is not None
        assert {entry.service_id: entry.name for entry in tables.sdt.services} == expected

    def test_the_frequency_comes_back(self) -> None:
        multiplex = demo_multiplex(centre_frequency_hz=546e6)
        tables = parse_mux(build_transport_stream(multiplex, repeats=2))
        assert tables.nit is not None
        assert tables.nit.transport_streams[0].centre_frequency_hz == 546e6

    def test_the_channel_numbers_come_back(self) -> None:
        multiplex = demo_multiplex()
        tables = parse_mux(build_transport_stream(multiplex, repeats=2))
        assert tables.nit is not None
        expected = {
            service.service_id: service.logical_channel
            for service in multiplex.services
            if service.logical_channel is not None
        }
        assert tables.nit.logical_channels == expected

    def test_a_radio_service_has_no_video_stream(self) -> None:
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        radio = next(
            service
            for service in demo_multiplex().services
            if service.service_type == int(ServiceType.DIGITAL_RADIO)
        )
        assert not tables.pmts[radio.service_id].has_video

    def test_the_stream_is_a_whole_number_of_packets(self) -> None:
        stream = build_transport_stream(demo_multiplex(), repeats=3)
        assert len(stream) % 188 == 0

    def test_two_services_sharing_an_identifier_are_refused(self) -> None:
        with pytest.raises(ValueError, match="share a service identifier"):
            SyntheticMultiplex(
                services=(
                    SyntheticService(service_id=1, name="A"),
                    SyntheticService(service_id=1, name="B"),
                )
            )

    def test_a_multiplex_with_no_services_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one service"):
            SyntheticMultiplex(services=())

    def test_a_service_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            SyntheticService(service_id=1, name="")


class TestMuxParser:
    def test_it_learns_the_pmt_pids_from_the_pat(self) -> None:
        # A parser that listened only to fixed PIDs would never read a PMT, because the PAT is
        # the only thing that says where they are.
        multiplex = demo_multiplex()
        parser = MuxParser()
        parser.feed(build_transport_stream(multiplex, repeats=2))
        expected = {multiplex.pmt_pid(index) for index in range(len(multiplex.services))}
        assert parser.pmt_pids == expected

    def test_it_knows_when_it_has_enough(self) -> None:
        # Which is what lets a scan stop reading instead of waiting out a timeout.
        multiplex = demo_multiplex()
        parser = MuxParser()
        for index, packet in enumerate(
            iter_packets(build_transport_stream(multiplex, repeats=3)), start=1
        ):
            parser.feed_packet(packet)
            if parser.is_complete:
                assert index < 3 * 20
                return
        pytest.fail("never became complete")

    def test_an_incomplete_stream_says_what_is_missing(self) -> None:
        stream = build_transport_stream(demo_multiplex(), repeats=1)
        parser = MuxParser()
        parser.feed(stream[: 188 * 2])
        missing = parser.tables.missing
        assert missing
        assert any("PMT" in item or item in {"SDT", "NIT"} for item in missing)

    def test_the_nit_is_not_required_for_completeness(self) -> None:
        # It carries the channel numbers, which are a convenience, and plenty of multiplexes
        # send it rarely. Waiting for one would mean a scan that never finishes.
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        assert tables.is_complete
        assert "NIT" not in tables.missing

    def test_resetting_forgets_the_previous_multiplex(self) -> None:
        parser = MuxParser()
        parser.feed(build_transport_stream(demo_multiplex(), repeats=2))
        parser.reset()
        assert not parser.has_services
        assert parser.pmt_pids == frozenset()

    def test_malformed_packets_do_not_stop_the_parse(self) -> None:
        stream = bytearray(build_transport_stream(demo_multiplex(), repeats=3))
        stream[188 * 2] = 0x00  # break one packet's sync byte
        assert parse_mux(bytes(stream)).has_services


class TestChannelList:
    def test_channels_come_out_in_viewing_order(self) -> None:
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        channels = channels_from_tables(tables, mux_freq_hz=498e6)
        numbers = [channel.logical_channel for channel in channels]
        assert numbers == sorted(n for n in numbers if n is not None)

    def test_television_and_radio_are_distinguished(self) -> None:
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        channels = {c.name: c.kind for c in channels_from_tables(tables, mux_freq_hz=498e6)}
        assert channels["OpenWave One"] is ChannelKind.TELEVISION
        assert channels["OpenWave Radio"] is ChannelKind.RADIO

    def test_a_scrambled_service_is_listed_but_not_playable(self) -> None:
        # Listing it matters: a channel list that silently omits what it cannot play leaves a
        # viewer wondering where a channel went.
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        channels = channels_from_tables(tables, mux_freq_hz=498e6)
        scrambled = next(channel for channel in channels if channel.scrambled)
        assert not scrambled.is_playable
        assert scrambled.name == "Premium Sport"

    def test_scrambled_services_can_be_left_out(self) -> None:
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        channels = channels_from_tables(tables, mux_freq_hz=498e6, include_scrambled=False)
        assert not any(channel.scrambled for channel in channels)

    def test_a_service_in_the_pat_but_not_the_sdt_is_still_listed(self) -> None:
        # A multiplex reconfigured while on the air gives exactly this: a PAT from after the
        # change and an SDT from before it. Dropping the service would make a channel vanish
        # for reasons a viewer cannot see.
        multiplex = demo_multiplex()
        tables = parse_mux(build_transport_stream(multiplex, repeats=2))
        trimmed = type(tables)(
            pat=tables.pat,
            sdt=None,
            nit=tables.nit,
            pmts=tables.pmts,
        )
        channels = channels_from_tables(trimmed, mux_freq_hz=498e6)
        assert len(channels) == len(multiplex.services)
        assert all(channel.name.startswith("Service ") for channel in channels)

    def test_the_frequency_comes_from_the_tuner_not_the_tables(self) -> None:
        # The NIT says where a transmitter claims to be; the tuner knows where it heard it.
        tables = parse_mux(
            build_transport_stream(demo_multiplex(centre_frequency_hz=498e6), repeats=2)
        )
        channels = channels_from_tables(tables, mux_freq_hz=546e6)
        assert all(channel.mux_freq_hz == 546e6 for channel in channels)

    def test_a_non_positive_frequency_is_refused(self) -> None:
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        with pytest.raises(ValueError, match="must be positive"):
            channels_from_tables(tables, mux_freq_hz=0.0)

    def test_channel_numbers_come_from_this_multiplex_not_another(self) -> None:
        # The NIT describes the whole network, and a service identifier is only unique within
        # its own multiplex. Another multiplex listed after this one, reusing the same service
        # identifier under a different number, must not renumber this one's channel.
        tables = parse_mux(build_transport_stream(demo_multiplex(), repeats=2))
        assert tables.pat is not None and tables.nit is not None
        own = next(
            entry
            for entry in tables.nit.transport_streams
            if entry.transport_stream_id == tables.pat.transport_stream_id
        )
        clashing = TransportStreamEntry(
            transport_stream_id=own.transport_stream_id + 1,
            original_network_id=own.original_network_id,
            logical_channels={service_id: 900 for service_id in own.logical_channels},
        )
        network = Nit(
            network_id=tables.nit.network_id,
            network_name=tables.nit.network_name,
            transport_streams=(own, clashing),
        )
        shared = type(tables)(pat=tables.pat, sdt=tables.sdt, nit=network, pmts=tables.pmts)

        channels = channels_from_tables(shared, mux_freq_hz=498e6)
        numbers = {c.service_id: c.logical_channel for c in channels if c.logical_channel}
        assert numbers == own.logical_channels


class TestChannelModel:
    def channel(self, **overrides: object) -> Channel:
        fields: dict[str, object] = {
            "name": "OpenWave One",
            "service_id": 1,
            "mux_freq_hz": 498e6,
            "transport_stream_id": 0x1000,
            "service_type": int(ServiceType.DIGITAL_TELEVISION),
        }
        fields.update(overrides)
        return Channel(**fields)  # type: ignore[arg-type]

    def test_an_unknown_service_type_falls_back_to_the_streams_present(self) -> None:
        # The service type is the station's own claim and is occasionally wrong; the presence
        # of pictures is a fact.
        assert self.channel(service_type=0xEE, video_pid=0x200).kind is ChannelKind.TELEVISION
        assert self.channel(service_type=0xEE, video_pid=None).kind is ChannelKind.OTHER

    def test_a_channel_with_no_streams_is_not_playable(self) -> None:
        assert not self.channel(video_pid=None, audio_pid=None).is_playable

    def test_a_scrambled_channel_is_not_playable(self) -> None:
        assert not self.channel(video_pid=0x200, scrambled=True).is_playable

    def test_the_label_includes_the_channel_number_when_there_is_one(self) -> None:
        assert self.channel(logical_channel=7).label == "7. OpenWave One"
        assert self.channel().label == "OpenWave One"

    def test_it_is_immutable(self) -> None:
        from pydantic import ValidationError

        channel = self.channel()
        with pytest.raises(ValidationError):
            channel.name = "Something Else"  # type: ignore[misc]

    def test_it_serialises_for_the_api(self) -> None:
        payload = self.channel(logical_channel=7).model_dump()
        assert payload["name"] == "OpenWave One"
        assert payload["logical_channel"] == 7


class TestScanChannel:
    def test_an_occupied_channel_is_read(self) -> None:
        tuner = demo_tuner()
        with tuner:
            mux = scan_channel(tuner, UHF_PLAN.frequency_of(24), bandwidth_hz=8e6, settings=QUICK)
        assert mux is not None
        assert mux.tables.is_complete
        assert mux.channels

    def test_an_empty_channel_gives_nothing(self) -> None:
        # The result for most of a band, and not an error.
        tuner = demo_tuner()
        with tuner:
            assert (
                scan_channel(tuner, UHF_PLAN.frequency_of(22), bandwidth_hz=8e6, settings=QUICK)
                is None
            )

    def test_the_tuner_statistics_are_recorded(self) -> None:
        tuner = demo_tuner()
        with tuner:
            mux = scan_channel(tuner, UHF_PLAN.frequency_of(24), bandwidth_hz=8e6, settings=QUICK)
        assert mux is not None
        assert mux.quality.locked
        assert mux.quality.snr_db is not None
        assert all(channel.snr_db == mux.quality.snr_db for channel in mux.channels)

    def test_a_locked_channel_with_no_tables_is_reported_as_locked(self) -> None:
        # "Something is transmitting here and I could not read it" is a different answer from
        # "nothing is here", and the difference tells a user whether to improve their aerial.
        noise = SyntheticMux(
            freq_hz=UHF_PLAN.frequency_of(30),
            transport_stream=bytes([0x47]) + bytes(187) * 1,
            snr_db=5.0,
        )
        tuner = MockDvbDevice(muxes=(noise,))
        with tuner:
            mux = scan_channel(
                tuner,
                UHF_PLAN.frequency_of(30),
                bandwidth_hz=8e6,
                settings=DvbScanSettings(lock_timeout_s=0.1, max_reads=2),
            )
        assert mux is not None
        assert not mux.tables.has_services


class TestScanSettingsValidation:
    def test_a_read_size_that_is_not_whole_packets_is_refused(self) -> None:
        with pytest.raises(ValueError, match="multiple of 188"):
            DvbScanSettings(read_bytes=1000)

    def test_a_negative_lock_timeout_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            DvbScanSettings(lock_timeout_s=-1.0)

    def test_a_non_positive_read_limit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_reads must be positive"):
            DvbScanSettings(max_reads=0)


class TestFullScan:
    def test_the_demonstration_multiplexes_are_all_found(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        assert result.mux_count == len(DEMO_MULTIPLEXES)
        found = {mux.freq_hz for mux in result.muxes}
        assert found == {multiplex.centre_frequency_hz for multiplex in DEMO_MULTIPLEXES}

    def test_every_service_is_found_exactly_once(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        expected = {
            service.name for multiplex in DEMO_MULTIPLEXES for service in multiplex.services
        }
        assert {channel.name for channel in result.channels} == expected
        assert len(result.channels) == len(expected)

    def test_multiplexes_are_labelled_with_their_channel_number(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        assert {mux.channel_number for mux in result.muxes} == {24, 27, 31}

    def test_every_channel_of_the_plan_is_tried(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        assert result.channels_tried == UHF_PLAN.channel_count

    def test_progress_is_reported_once_per_channel(self) -> None:
        seen: list[int] = []
        scan_dvb(
            demo_tuner(),
            UHF_PLAN,
            settings=QUICK,
            progress=lambda index, total, freq: seen.append(index),
        )
        assert seen == list(range(UHF_PLAN.channel_count))

    def test_an_empty_band_finds_nothing(self) -> None:
        result = scan_dvb(MockDvbDevice(), UHF_PLAN, settings=QUICK)
        assert result.mux_count == 0
        assert result.channels == ()

    def test_channels_the_tuner_cannot_reach_are_skipped(self) -> None:
        # A tuner that covers UHF but not VHF is ordinary, and should not fail a scan.
        narrow = MockDvbDevice(tuning_range=TuningRange(min_hz=470e6, max_hz=700e6))
        result = scan_dvb(narrow, VHF_BAND_III_PLAN, settings=QUICK)
        assert result.channels_tried == 0

    def test_it_leaves_the_tuner_as_it_found_it(self) -> None:
        tuner = demo_tuner()
        assert not tuner.is_open
        scan_dvb(tuner, UHF_PLAN, settings=QUICK)
        assert not tuner.is_open

    def test_unnumbered_services_come_after_numbered_ones(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        numbers = [channel.logical_channel for channel in result.channels]
        first_unnumbered = next(
            (index for index, number in enumerate(numbers) if number is None), len(numbers)
        )
        assert all(number is None for number in numbers[first_unnumbered:])

    def test_it_counts_the_television_services(self) -> None:
        result = scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK)
        assert 0 < result.television_count < result.channel_count

    def test_it_summarises_itself_for_a_terminal(self) -> None:
        summary = str(scan_dvb(demo_tuner(), UHF_PLAN, settings=QUICK))
        assert "3 multiplexes" in summary
        assert "28 channels tried" in summary
