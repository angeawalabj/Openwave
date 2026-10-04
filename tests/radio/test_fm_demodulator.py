"""Tests for FM demodulation.

Each stage is checked against a signal whose content is known in advance: a tone goes in, the
same tone must come out, at the same pitch, in the right channel.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as scipy_signal

from openwave.radio.constants import STEREO_PILOT_HZ
from openwave.radio.fm_demodulator import (
    AUDIO_SAMPLE_RATE_HZ,
    MPX_SAMPLE_RATE_HZ,
    decimate_to,
    decode_mono,
    decode_stereo,
    deemphasise,
    demodulate,
    is_stereo,
    pilot_level_db,
    quadrature_demodulate,
    shift_to_baseband,
)
from openwave.radio.synthesis import SyntheticFmStation
from openwave.sdr.mock import MockSdrDevice

SAMPLE_RATE_HZ = 2_400_000.0
CENTER_FREQ_HZ = 99_000_000.0


def capture(
    station: SyntheticFmStation, *, noise_floor_dbfs: float = -80.0, count: int = 1 << 19
) -> np.ndarray:
    device = MockSdrDevice(sources=[station], noise_floor_dbfs=noise_floor_dbfs)
    with device:
        device.set_sample_rate(SAMPLE_RATE_HZ)
        device.set_center_freq(CENTER_FREQ_HZ)
        return device.read_samples(count)


def dominant_frequency(signal: np.ndarray, sample_rate_hz: float) -> float:
    """The strongest frequency present, in hertz."""
    nperseg = min(16384, len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")
    return float(freqs[int(np.argmax(psd))])


def absolute_level_at(signal: np.ndarray, sample_rate_hz: float, target_hz: float) -> float:
    """Level at ``target_hz`` in dB, on no particular reference.

    Only differences between two measurements of the same quantity are meaningful. Use this to
    compare one frequency before and after a filter, where ``level_at`` cannot: filtering moves
    the median it normalises against, so a difference of two median-relative levels is not an
    attenuation.
    """
    nperseg = min(16384, len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")
    in_db = 10 * np.log10(psd + 1e-30)
    return float(in_db[int(np.argmin(np.abs(freqs - target_hz)))])


def level_at(signal: np.ndarray, sample_rate_hz: float, target_hz: float) -> float:
    """Level at ``target_hz``, in dB above the median of the spectrum."""
    nperseg = min(16384, len(signal))
    freqs, psd = scipy_signal.welch(signal, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")
    in_db = 10 * np.log10(psd + 1e-30)
    return float(in_db[int(np.argmin(np.abs(freqs - target_hz)))] - np.median(in_db))


def mpx_of(station: SyntheticFmStation, **kwargs: object) -> tuple[np.ndarray, float]:
    """The recovered multiplex of one station, and its sample rate."""
    samples = capture(station, **kwargs)  # type: ignore[arg-type]
    baseband, rate = decimate_to(samples, sample_rate_hz=SAMPLE_RATE_HZ)
    return quadrature_demodulate(baseband, sample_rate_hz=rate), rate


class TestShiftToBaseband:
    def test_a_station_off_centre_comes_down_to_zero(self) -> None:
        station = SyntheticFmStation(freq_hz=99.7e6, deviation_hz=100.0)
        samples = capture(station)
        shifted = shift_to_baseband(samples, offset_hz=700e3, sample_rate_hz=SAMPLE_RATE_HZ)
        assert abs(dominant_frequency(shifted, SAMPLE_RATE_HZ)) < 2000.0

    def test_a_zero_offset_leaves_the_samples_alone(self) -> None:
        samples = capture(SyntheticFmStation(freq_hz=99.0e6))
        shifted = shift_to_baseband(samples, offset_hz=0.0, sample_rate_hz=SAMPLE_RATE_HZ)
        np.testing.assert_array_equal(shifted, samples)

    def test_a_non_positive_sample_rate_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sample rate must be positive"):
            shift_to_baseband(np.zeros(4, dtype=np.complex64), offset_hz=0.0, sample_rate_hz=0.0)


class TestDecimation:
    def test_it_reports_the_rate_it_actually_reached(self) -> None:
        # Decimation is by an integer factor, so the rate out is not exactly the target.
        # Assuming it is the target is how a demodulator reports a tone at the wrong pitch.
        samples = np.zeros(1 << 16, dtype=np.complex64)
        _, rate = decimate_to(samples, sample_rate_hz=2.4e6, target_rate_hz=480e3)
        assert rate == 480_000.0

        _, odd_rate = decimate_to(samples, sample_rate_hz=2.4e6, target_rate_hz=500e3)
        assert odd_rate == pytest.approx(2.4e6 / 4)

    def test_a_target_above_the_input_rate_changes_nothing(self) -> None:
        samples = np.zeros(1024, dtype=np.complex64)
        decimated, rate = decimate_to(samples, sample_rate_hz=240e3, target_rate_hz=480e3)
        assert rate == 240e3
        assert decimated.size == 1024

    def test_it_reduces_the_number_of_samples(self) -> None:
        samples = np.zeros(1 << 16, dtype=np.complex64)
        decimated, _ = decimate_to(samples, sample_rate_hz=2.4e6, target_rate_hz=480e3)
        assert decimated.size == pytest.approx((1 << 16) / 5, rel=0.01)

    def test_a_large_reduction_is_done_in_stages(self) -> None:
        # scipy's FIR decimation is unreliable beyond a factor of about 13 in one pass.
        samples = np.zeros(1 << 18, dtype=np.complex64)
        decimated, rate = decimate_to(samples, sample_rate_hz=2.4e6, target_rate_hz=24e3)
        assert rate == pytest.approx(24e3, rel=0.01)
        assert decimated.size > 0

    @pytest.mark.parametrize("rate", [0.0, -1.0])
    def test_non_positive_rates_are_refused(self, rate: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            decimate_to(np.zeros(16, dtype=np.complex64), sample_rate_hz=rate)


class TestQuadratureDemodulation:
    def test_a_constant_frequency_offset_comes_back_as_that_offset(self) -> None:
        # The simplest possible check of the scaling: an unmodulated carrier 1 kHz off centre
        # must demodulate to a constant 1 kHz.
        rate = 48_000.0
        n = np.arange(4096)
        tone = np.exp(2j * np.pi * 1000.0 * n / rate).astype(np.complex64)
        recovered = quadrature_demodulate(tone, sample_rate_hz=rate)
        assert np.mean(recovered) == pytest.approx(1000.0, abs=1.0)
        assert np.std(recovered) < 1.0

    def test_the_output_is_one_sample_shorter(self) -> None:
        samples = np.ones(100, dtype=np.complex64)
        assert quadrature_demodulate(samples, sample_rate_hz=48e3).size == 99

    def test_fewer_than_two_samples_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two samples"):
            quadrature_demodulate(np.ones(1, dtype=np.complex64), sample_rate_hz=48e3)

    @pytest.mark.parametrize("tone_hz", [400.0, 1000.0, 3500.0, 10_000.0])
    def test_the_modulating_tone_is_recovered(self, tone_hz: float) -> None:
        station = SyntheticFmStation(freq_hz=99.0e6, left_tone_hz=tone_hz, deviation_hz=50_000.0)
        mpx, rate = mpx_of(station)
        assert dominant_frequency(mpx, rate) == pytest.approx(tone_hz, rel=0.02, abs=20.0)


class TestPilotDetection:
    def test_a_stereo_transmission_shows_a_strong_pilot(self) -> None:
        station = SyntheticFmStation(
            freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
        )
        mpx, rate = mpx_of(station)
        assert pilot_level_db(mpx, sample_rate_hz=rate) > 40.0
        assert is_stereo(mpx, sample_rate_hz=rate)

    def test_a_mono_transmission_shows_no_pilot(self) -> None:
        station = SyntheticFmStation(freq_hz=99.0e6, left_tone_hz=1000.0)
        mpx, rate = mpx_of(station)
        assert pilot_level_db(mpx, sample_rate_hz=rate) < 10.0
        assert not is_stereo(mpx, sample_rate_hz=rate)

    def test_a_mono_station_at_full_deviation_is_still_not_mistaken_for_stereo(self) -> None:
        # The failure this guards against. Decimating to a rate sized for the 60 kHz multiplex
        # rather than for the 270 kHz modulated signal clips the outer FM sidebands, and the
        # clipping puts distortion right across the multiplex. At 240 kHz a mono station
        # modulated by a 1 kHz tone at full deviation showed 62.8 dB at 19 kHz and was reported
        # as stereo. The margin at 480 kHz is around 60 dB.
        station = SyntheticFmStation(
            freq_hz=99.0e6, left_tone_hz=1000.0, deviation_hz=75_000.0, stereo=False
        )
        mpx, rate = mpx_of(station)
        assert pilot_level_db(mpx, sample_rate_hz=rate) < 10.0

    def test_the_multiplex_rate_is_wide_enough_for_the_modulated_signal(self) -> None:
        # Carson's rule for stereo at full deviation: 2 * (75 + 60) = 270 kHz. scipy's
        # decimation filter passes about 0.8 of Nyquist, so the rate must exceed 338 kHz.
        carson_hz = 2 * (75_000.0 + 60_000.0)
        assert carson_hz < MPX_SAMPLE_RATE_HZ * 0.8

    def test_a_pilot_survives_a_poor_signal_to_noise_ratio(self) -> None:
        station = SyntheticFmStation(
            freq_hz=99.0e6, power_dbfs=-25.0, stereo=True, right_tone_hz=400.0
        )
        mpx, rate = mpx_of(station, noise_floor_dbfs=-35.0)
        assert is_stereo(mpx, sample_rate_hz=rate)

    def test_too_few_samples_to_judge_is_refused_rather_than_guessed(self) -> None:
        with pytest.raises(ValueError, match="at least 1024 samples"):
            pilot_level_db(np.zeros(500, dtype=np.float64), sample_rate_hz=MPX_SAMPLE_RATE_HZ)

    def test_a_rate_too_low_to_represent_the_pilot_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot represent"):
            pilot_level_db(np.zeros(4096, dtype=np.float64), sample_rate_hz=STEREO_PILOT_HZ * 1.5)


class TestDeemphasis:
    def test_it_attenuates_treble_and_leaves_bass_alone(self) -> None:
        rate = 48_000.0
        duration = np.arange(48_000) / rate
        for tone_hz, expected_loss_db in ((100.0, 0.0), (10_000.0, 10.0)):
            signal = np.sin(2 * np.pi * tone_hz * duration)
            filtered = deemphasise(signal, sample_rate_hz=rate)
            loss_db = 20 * np.log10(np.std(signal) / (np.std(filtered) + 1e-30))
            assert loss_db == pytest.approx(expected_loss_db, abs=4.0)

    def test_a_longer_time_constant_attenuates_more(self) -> None:
        # 75 µs is used in the Americas, 50 µs in Europe. Applying the wrong one sounds dull or
        # harsh rather than broken, but the difference has to be real.
        rate = 48_000.0
        signal = np.sin(2 * np.pi * 8000.0 * np.arange(24_000) / rate)
        europe = deemphasise(signal, sample_rate_hz=rate, tau_s=50e-6)
        americas = deemphasise(signal, sample_rate_hz=rate, tau_s=75e-6)
        assert np.std(americas) < np.std(europe)

    @pytest.mark.parametrize(("rate", "tau"), [(0.0, 50e-6), (48_000.0, 0.0), (48_000.0, -1.0)])
    def test_nonsensical_arguments_are_refused(self, rate: float, tau: float) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            deemphasise(np.zeros(16), sample_rate_hz=rate, tau_s=tau)


class TestMultiplexDecoding:
    def test_mono_decoding_keeps_the_audio_and_removes_the_pilot(self) -> None:
        # The failure this guards against: a low-pass at 15 kHz built with an arbitrary 129
        # taps has a transition region about 6 kHz wide at this sample rate, which puts the
        # 19 kHz pilot inside it. The filter meant to remove the pilot then barely touches it,
        # and the mono output carries a loud whistle at the edge of hearing. Designing the
        # filter from its transition width instead gives well over 100 dB of rejection.
        station = SyntheticFmStation(
            freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
        )
        mpx, rate = mpx_of(station)
        mono = decode_mono(mpx, sample_rate_hz=rate)

        pilot_rejection_db = absolute_level_at(mpx, rate, STEREO_PILOT_HZ) - absolute_level_at(
            mono, rate, STEREO_PILOT_HZ
        )
        audio_loss_db = absolute_level_at(mpx, rate, 1000.0) - absolute_level_at(mono, rate, 1000.0)

        assert pilot_rejection_db > 60.0
        assert abs(audio_loss_db) < 1.0

    def test_stereo_decoding_separates_the_channels(self) -> None:
        # The real test of the stereo decoder: a different tone in each channel must come out
        # in the channel it went into, with the other one well suppressed.
        station = SyntheticFmStation(
            freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
        )
        mpx, rate = mpx_of(station)
        left, right = decode_stereo(mpx, sample_rate_hz=rate)

        left_separation = level_at(left, rate, 1000.0) - level_at(left, rate, 400.0)
        right_separation = level_at(right, rate, 400.0) - level_at(right, rate, 1000.0)
        assert left_separation > 20.0
        assert right_separation > 20.0

    def test_stereo_decoding_follows_the_phase_the_standard_specifies(self) -> None:
        # Built by hand from ITU-R BS.450 rather than through the simulator, so the decoder and
        # the generator cannot share a wrong convention. With the pilot and subcarrier as sines,
        # a reference made by squaring the pilot is a quarter-turn off, the difference signal
        # demodulates to nothing, and a tone in the left channel comes out equally in both.
        rate = MPX_SAMPLE_RATE_HZ
        t = np.arange(int(rate * 0.5)) / rate
        pilot_rad = 2.0 * np.pi * STEREO_PILOT_HZ * t
        left = np.sin(2.0 * np.pi * 1000.0 * t)
        right = np.zeros_like(t)
        mpx = (
            0.45 * (left + right) / 2.0
            + 0.45 * (left - right) / 2.0 * np.sin(2.0 * pilot_rad)
            + 0.1 * np.sin(pilot_rad)
        )

        decoded_left, decoded_right = decode_stereo(mpx, sample_rate_hz=rate)

        edge = len(t) // 20
        separation_db = 20.0 * np.log10(
            np.std(decoded_left[edge:-edge]) / np.std(decoded_right[edge:-edge])
        )
        assert separation_db > 20.0


class TestDemodulate:
    def test_it_delivers_audio_at_the_requested_rate(self) -> None:
        audio = demodulate(
            capture(SyntheticFmStation(freq_hz=99.0e6)), sample_rate_hz=SAMPLE_RATE_HZ
        )
        assert audio.sample_rate_hz == AUDIO_SAMPLE_RATE_HZ
        assert audio.left.dtype == np.float32
        assert audio.right.dtype == np.float32
        assert audio.left.size == audio.right.size
        assert audio.duration_s > 0

    def test_the_audio_stays_within_full_scale(self) -> None:
        # The demodulator produces hertz of deviation, a number in the tens of thousands.
        # Playback needs values between -1 and 1.
        audio = demodulate(
            capture(SyntheticFmStation(freq_hz=99.0e6, deviation_hz=75_000.0)),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        assert np.max(np.abs(audio.left)) <= 1.0
        assert np.max(np.abs(audio.right)) <= 1.0

    def test_a_mono_station_gives_the_same_signal_in_both_channels(self) -> None:
        audio = demodulate(
            capture(SyntheticFmStation(freq_hz=99.0e6)), sample_rate_hz=SAMPLE_RATE_HZ
        )
        assert not audio.is_stereo
        np.testing.assert_array_equal(audio.left, audio.right)

    def test_a_stereo_station_gives_different_channels(self) -> None:
        audio = demodulate(
            capture(
                SyntheticFmStation(
                    freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
                )
            ),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        assert audio.is_stereo
        assert not np.array_equal(audio.left, audio.right)
        assert level_at(audio.left, audio.sample_rate_hz, 1000.0) > level_at(
            audio.left, audio.sample_rate_hz, 400.0
        )

    @pytest.mark.parametrize("tone_hz", [400.0, 1000.0, 5000.0])
    def test_the_tone_survives_the_whole_chain(self, tone_hz: float) -> None:
        audio = demodulate(
            capture(
                SyntheticFmStation(freq_hz=99.0e6, left_tone_hz=tone_hz, deviation_hz=50_000.0)
            ),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        found = dominant_frequency(audio.mono, audio.sample_rate_hz)
        assert found == pytest.approx(tone_hz, rel=0.02, abs=5.0)

    def test_a_station_away_from_the_tuned_frequency_is_demodulated(self) -> None:
        audio = demodulate(
            capture(SyntheticFmStation(freq_hz=99.7e6, left_tone_hz=1000.0, deviation_hz=50_000.0)),
            sample_rate_hz=SAMPLE_RATE_HZ,
            offset_hz=700_000.0,
        )
        assert dominant_frequency(audio.mono, audio.sample_rate_hz) == pytest.approx(
            1000.0, abs=5.0
        )

    def test_the_mono_sum_of_a_stereo_signal_is_the_average(self) -> None:
        audio = demodulate(
            capture(
                SyntheticFmStation(
                    freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
                )
            ),
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        np.testing.assert_allclose(audio.mono, (audio.left + audio.right) / 2.0, atol=1e-6)

    def test_a_sample_rate_too_low_for_the_multiplex_is_refused(self) -> None:
        # 100 kS/s cannot carry a 60 kHz multiplex, and saying so beats returning noise.
        samples = np.zeros(1 << 15, dtype=np.complex64)
        with pytest.raises(ValueError, match="cannot carry"):
            demodulate(samples, sample_rate_hz=100_000.0)

    def test_the_stereo_threshold_can_be_raised_to_force_mono(self) -> None:
        station = SyntheticFmStation(
            freq_hz=99.0e6, stereo=True, left_tone_hz=1000.0, right_tone_hz=400.0
        )
        audio = demodulate(
            capture(station), sample_rate_hz=SAMPLE_RATE_HZ, stereo_threshold_db=200.0
        )
        assert not audio.is_stereo
