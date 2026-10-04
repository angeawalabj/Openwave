"""Recovering audio from an FM carrier.

The chain, from samples to sound:

1. **Shift** the wanted station to zero hertz. A window holds several stations; this picks one.
2. **Decimate** to a rate wide enough for the *modulated* signal, not merely for the multiplex
   inside it. See :data:`MPX_SAMPLE_RATE_HZ`: getting this wrong produces distortion that
   imitates a stereo pilot. Decimating also throws away most of the noise and all the
   neighbouring stations.
3. **Demodulate.** FM carries its information in how fast the phase turns, so the instantaneous
   frequency is the phase difference between consecutive samples. What comes out is the
   multiplex.
4. **Decode the multiplex.** Mono is the part below 15 kHz. Stereo adds a pilot at 19 kHz and
   the difference signal on a 38 kHz subcarrier, which has to be brought back down and
   recombined.
5. **De-emphasise.** Transmitters boost the treble before sending; receivers must undo it or
   the result is harsh and noisy.

Every function here is pure: arrays in, arrays out. The tests drive them with signals from
:mod:`openwave.radio.synthesis`, where the right answer is known in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from openwave.radio.constants import (
    FM_AUDIO_BANDWIDTH_HZ,
    FM_DEEMPHASIS_TAU_S,
    FM_MAX_DEVIATION_HZ,
    FM_MPX_BANDWIDTH_HZ,
    STEREO_PILOT_HZ,
    STEREO_SUBCARRIER_HZ,
)
from openwave.sdr.device import IqSamples

#: Sample rate the multiplex is decoded at, in Hz.
#:
#: The number to size this against is not the width of the multiplex but the width of the
#: *modulated signal carrying it*. By Carson's rule a stereo transmission at full deviation
#: occupies ``2 * (75 kHz + 60 kHz)`` = 270 kHz, and scipy's decimation filter passes about
#: 0.8 of Nyquist, so the rate has to exceed ``2 * 135 / 0.8`` = 338 kHz.
#:
#: Sizing it from the 60 kHz multiplex instead -- Nyquist says 120 kHz, so 240 kHz is generous
#: -- is wrong, and wrong in a way that does not look like an error. The decimation filter
#: clips the outer FM sidebands, and the clipping puts distortion products across the
#: multiplex. Measured against the simulator, a mono station modulated by a 1 kHz tone at full
#: deviation then showed 62.8 dB of energy at 19 kHz and was confidently reported as stereo. At
#: 480 kHz the same signal shows 3.6 dB, against 63.8 dB for a real pilot.
#:
#: 480 kHz also divides the common SDR rates evenly: 2.4 MS/s decimates by exactly 5, and the
#: result decimates by exactly 10 to deliver 48 kHz audio.
MPX_SAMPLE_RATE_HZ = 480_000.0

#: Sample rate the audio is delivered at, in Hz.
AUDIO_SAMPLE_RATE_HZ = 48_000.0

#: Half-width of the band-pass used to isolate the pilot tone, in Hz.
PILOT_FILTER_WIDTH_HZ = 500.0

#: Audio signal, mono or two-channel, with values between -1.0 and 1.0.
AudioSamples = npt.NDArray[np.float32]

#: Real-valued baseband signal, such as the recovered multiplex.
RealSignal = npt.NDArray[np.float64]

#: How to set the output level. See :func:`_normalising_scale`.
Normalisation = Literal["deviation", "peak"]


@dataclass(frozen=True, slots=True)
class DemodulatedAudio:
    """The result of demodulating one station."""

    left: AudioSamples
    right: AudioSamples
    sample_rate_hz: float
    is_stereo: bool
    """Whether a pilot tone was found and the difference channel was decoded."""

    pilot_level_db: float
    """How far the pilot stands above the surrounding multiplex, in dB."""

    @property
    def duration_s(self) -> float:
        """How long the audio lasts."""
        return len(self.left) / self.sample_rate_hz

    @property
    def mono(self) -> AudioSamples:
        """The two channels summed back to one."""
        summed: AudioSamples = ((self.left + self.right) / 2.0).astype(np.float32)
        return summed


def shift_to_baseband(samples: IqSamples, *, offset_hz: float, sample_rate_hz: float) -> IqSamples:
    """Move the signal at ``offset_hz`` down to zero hertz.

    ``offset_hz`` is where the station sits relative to the tuned frequency, so it is negative
    for a station below it.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")
    if offset_hz == 0.0:
        return samples
    n = np.arange(samples.size, dtype=np.float64)
    rotation = np.exp(-2j * np.pi * offset_hz * n / sample_rate_hz)
    shifted: IqSamples = (samples * rotation).astype(np.complex64)
    return shifted


def decimate_to(
    samples: IqSamples, *, sample_rate_hz: float, target_rate_hz: float = MPX_SAMPLE_RATE_HZ
) -> tuple[IqSamples, float]:
    """Reduce the sample rate towards ``target_rate_hz``.

    Decimation is by an integer factor, so the rate that comes out is the input rate divided by
    that factor and not exactly the target. The actual rate is returned alongside, because
    assuming it is the target is how a demodulator ends up reporting a tone at the wrong pitch.

    Returns:
        The decimated samples and their true sample rate.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")
    if target_rate_hz <= 0:
        raise ValueError(f"target rate must be positive, got {target_rate_hz}")

    factor = max(1, int(sample_rate_hz // target_rate_hz))
    if factor == 1:
        return samples, sample_rate_hz

    # scipy's decimate is limited to a factor of 13 per pass with an FIR filter, so a large
    # reduction is done in stages. Each stage filters before discarding samples, which is what
    # keeps the neighbours from folding on top of the wanted station.
    current: IqSamples = samples
    current_rate = sample_rate_hz
    remaining = factor
    while remaining > 1:
        stage = min(remaining, 10)
        current = scipy_signal.decimate(current, stage, ftype="fir", zero_phase=True).astype(
            np.complex64
        )
        current_rate /= stage
        remaining //= stage
    return current, current_rate


def quadrature_demodulate(samples: IqSamples, *, sample_rate_hz: float) -> RealSignal:
    """Recover the instantaneous frequency of ``samples``, in hertz.

    FM encodes the signal in how fast the phase turns, so the recovered signal is the phase
    advance between consecutive samples, scaled by the sample rate. Multiplying by the previous
    sample's conjugate and taking the angle gives that difference while unwrapping it for free:
    the result is always in the range the arctangent can express, so a phase crossing pi does
    not produce a spurious jump.

    The output is one sample shorter than the input, since differences need pairs.
    """
    if samples.size < 2:
        raise ValueError(f"need at least two samples to take a difference, got {samples.size}")
    if sample_rate_hz <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")

    product = samples[1:] * np.conj(samples[:-1])
    phase_advance = np.angle(product)
    recovered: RealSignal = np.asarray(
        phase_advance * (sample_rate_hz / (2.0 * np.pi)), dtype=np.float64
    )
    return recovered


def pilot_level_db(mpx: RealSignal, *, sample_rate_hz: float) -> float:
    """How far the 19 kHz pilot stands above the multiplex around it, in dB.

    The pilot is the only thing that marks a transmission as stereo, so this is what the stereo
    decision is made on. It is measured against the level just either side of 19 kHz rather than
    against the whole multiplex, because the multiplex level depends on how loud the programme
    happens to be.

    Returns a large negative number when there is no pilot at all.
    """
    if sample_rate_hz <= 2 * STEREO_PILOT_HZ:
        raise ValueError(
            f"a sample rate of {sample_rate_hz} Hz cannot represent a {STEREO_PILOT_HZ} Hz pilot"
        )

    nperseg = min(16384, _largest_power_of_two(mpx.size))
    if nperseg < 1024:
        raise ValueError(
            f"need at least 1024 samples of multiplex to measure the pilot, got {mpx.size}"
        )

    freqs, psd = scipy_signal.welch(mpx, fs=sample_rate_hz, nperseg=nperseg, detrend="constant")
    resolution_hz = float(freqs[1] - freqs[0])

    pilot_band = np.abs(freqs - STEREO_PILOT_HZ) <= PILOT_FILTER_WIDTH_HZ
    # Reference taken from two patches either side of the pilot, close enough to share its
    # noise but far enough not to include it.
    low = np.abs(freqs - (STEREO_PILOT_HZ - 3_000.0)) <= PILOT_FILTER_WIDTH_HZ
    high = np.abs(freqs - (STEREO_PILOT_HZ + 3_000.0)) <= PILOT_FILTER_WIDTH_HZ
    reference_band = low | high

    if not pilot_band.any() or not reference_band.any():
        raise ValueError(
            f"a spectrum of resolution {resolution_hz:.1f} Hz is too coarse to isolate the pilot"
        )

    pilot_power = float(np.sum(psd[pilot_band]))
    reference_power = float(np.sum(psd[reference_band]) * pilot_band.sum() / reference_band.sum())
    if reference_power <= 0:
        return 0.0 if pilot_power > 0 else -300.0
    if pilot_power <= 0:
        return -300.0
    return 10.0 * float(np.log10(pilot_power / reference_power))


def is_stereo(mpx: RealSignal, *, sample_rate_hz: float, threshold_db: float = 10.0) -> bool:
    """Whether a transmission is stereo, judged by its pilot tone."""
    return pilot_level_db(mpx, sample_rate_hz=sample_rate_hz) >= threshold_db


def deemphasise(
    audio: RealSignal, *, sample_rate_hz: float, tau_s: float = FM_DEEMPHASIS_TAU_S
) -> RealSignal:
    """Undo the treble boost a transmitter applies.

    FM broadcasting pre-emphasises the high frequencies before transmission, because noise in an
    FM system grows with frequency; the receiver attenuates them again, and the noise with them.
    The filter is a single-pole low-pass with time constant ``tau_s``, which is 50 µs in ITU
    Region 1 and 75 µs in Region 2 and Japan. Using the wrong one does not break the audio, it
    just sounds dull or harsh.
    """
    if sample_rate_hz <= 0:
        raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")
    if tau_s <= 0:
        raise ValueError(f"time constant must be positive, got {tau_s}")

    # Bilinear transform of 1 / (1 + s*tau), normalised to unity gain at zero frequency.
    weight = 1.0 / (1.0 + 2.0 * sample_rate_hz * tau_s)
    numerator = np.array([weight, weight], dtype=np.float64)
    denominator = np.array(
        [1.0, (1.0 - 2.0 * sample_rate_hz * tau_s) / (1.0 + 2.0 * sample_rate_hz * tau_s)],
        dtype=np.float64,
    )
    filtered: RealSignal = scipy_signal.lfilter(numerator, denominator, audio).astype(np.float64)
    return filtered


def decode_mono(mpx: RealSignal, *, sample_rate_hz: float) -> RealSignal:
    """Take the mono audio out of a multiplex: everything below 15 kHz.

    The filter has to be fully stopped by 19 kHz, where the pilot sits. A gentler one leaves the
    pilot in the audio as a loud whistle just at the edge of hearing.
    """
    return _low_pass(
        mpx,
        sample_rate_hz=sample_rate_hz,
        cutoff_hz=FM_AUDIO_BANDWIDTH_HZ,
        transition_hz=STEREO_PILOT_HZ - FM_AUDIO_BANDWIDTH_HZ,
    )


def decode_stereo(mpx: RealSignal, *, sample_rate_hz: float) -> tuple[RealSignal, RealSignal]:
    """Take the left and right channels out of a stereo multiplex.

    The multiplex carries the sum of the channels at baseband and their difference on a 38 kHz
    subcarrier. Recovering the difference means multiplying by a 38 kHz tone locked to the
    pilot, which brings it back down to baseband -- and locked is the important word, because
    the sum and difference have to be in phase to recombine into left and right.

    The 38 kHz reference is built from the extracted pilot by doubling its phase, which is why
    the standard puts the pilot at exactly half the subcarrier. Doubling has to be done on the
    analytic pilot rather than by squaring the real one. ITU-R BS.450 sends the pilot as
    ``sin(wt)`` and the subcarrier as ``sin(2wt)``, and ``sin(wt)`` squared is
    ``(1 - cos(2wt)) / 2``: a reference a quarter-turn away from the subcarrier, against which
    the difference signal demodulates to nothing and a stereo station comes out mono.
    """
    sum_signal = decode_mono(mpx, sample_rate_hz=sample_rate_hz)

    pilot = _band_pass(
        mpx,
        sample_rate_hz=sample_rate_hz,
        low_hz=STEREO_PILOT_HZ - PILOT_FILTER_WIDTH_HZ,
        high_hz=STEREO_PILOT_HZ + PILOT_FILTER_WIDTH_HZ,
        transition_hz=PILOT_FILTER_WIDTH_HZ * 2.0,
    )
    # The analytic pilot is exp(j(wt - pi/2)) for a pilot of sin(wt). Squaring it doubles the
    # phase to exp(j(2wt - pi)) = -exp(j2wt), whose imaginary part, negated, is sin(2wt): the
    # subcarrier exactly as the standard places it.
    analytic = scipy_signal.hilbert(pilot)
    reference = -np.imag(analytic * analytic)
    reference = _band_pass(
        reference,
        sample_rate_hz=sample_rate_hz,
        low_hz=STEREO_SUBCARRIER_HZ - 2_000.0,
        high_hz=STEREO_SUBCARRIER_HZ + 2_000.0,
        transition_hz=2_000.0,
    )
    peak = float(np.max(np.abs(reference)))
    if peak > 0:
        reference = reference / peak

    # Multiplying by the reference shifts the difference signal down to baseband. The factor of
    # two is the usual coherent-detection gain: a double-sideband signal multiplied by its own
    # carrier comes back at half amplitude.
    difference = decode_mono(mpx * reference * 2.0, sample_rate_hz=sample_rate_hz)

    # The standard sends the sum and difference at equal deviation shares, so they come back at
    # the same scale and recombine directly.
    left = sum_signal + difference
    right = sum_signal - difference
    return left, right


def demodulate(
    samples: IqSamples,
    *,
    sample_rate_hz: float,
    offset_hz: float = 0.0,
    audio_rate_hz: float = AUDIO_SAMPLE_RATE_HZ,
    stereo_threshold_db: float = 10.0,
    deemphasis_tau_s: float = FM_DEEMPHASIS_TAU_S,
    normalise: Normalisation = "deviation",
    reference_deviation_hz: float = FM_MAX_DEVIATION_HZ,
) -> DemodulatedAudio:
    """Turn IQ samples into audio, deciding stereo or mono from the pilot.

    Args:
        samples: Complex baseband samples from a receiver.
        sample_rate_hz: Rate they were sampled at.
        offset_hz: Where the wanted station sits relative to the tuned frequency.
        audio_rate_hz: Rate to deliver audio at.
        stereo_threshold_db: How far above its surroundings the pilot must stand before the
            difference channel is decoded. Decoding a difference channel that is not there adds
            noise to both outputs, so the threshold errs towards mono.
        deemphasis_tau_s: De-emphasis time constant. 50 µs in Europe, 75 µs in the Americas.
        normalise: How to set the output level. ``"deviation"`` uses a fixed reference and so
            gives the same level for every block, which is what continuous listening needs.
            ``"peak"`` fills the available range using the loudest sample present, which only
            makes sense for a single finite block.
        reference_deviation_hz: Deviation treated as full scale when ``normalise`` is
            ``"deviation"``.
    """
    shifted = shift_to_baseband(samples, offset_hz=offset_hz, sample_rate_hz=sample_rate_hz)
    baseband, mpx_rate_hz = decimate_to(shifted, sample_rate_hz=sample_rate_hz)
    if mpx_rate_hz <= 2 * FM_MPX_BANDWIDTH_HZ:
        raise ValueError(
            f"a multiplex rate of {mpx_rate_hz / 1e3:.0f} kHz cannot carry the "
            f"{FM_MPX_BANDWIDTH_HZ / 1e3:.0f} kHz multiplex; sample faster than "
            f"{2 * FM_MPX_BANDWIDTH_HZ / 1e3:.0f} kHz"
        )

    mpx = quadrature_demodulate(baseband, sample_rate_hz=mpx_rate_hz)
    pilot_db = pilot_level_db(mpx, sample_rate_hz=mpx_rate_hz)
    stereo = pilot_db >= stereo_threshold_db

    if stereo:
        left, right = decode_stereo(mpx, sample_rate_hz=mpx_rate_hz)
    else:
        mono = decode_mono(mpx, sample_rate_hz=mpx_rate_hz)
        left, right = mono, mono

    left = deemphasise(left, sample_rate_hz=mpx_rate_hz, tau_s=deemphasis_tau_s)
    right = deemphasise(right, sample_rate_hz=mpx_rate_hz, tau_s=deemphasis_tau_s)

    left_audio, actual_rate = _resample_to(left, from_rate_hz=mpx_rate_hz, to_rate_hz=audio_rate_hz)
    right_audio, _ = _resample_to(right, from_rate_hz=mpx_rate_hz, to_rate_hz=audio_rate_hz)

    scale = _normalising_scale(
        left_audio,
        right_audio,
        normalise=normalise,
        reference_deviation_hz=reference_deviation_hz,
    )
    return DemodulatedAudio(
        left=np.clip(left_audio * scale, -1.0, 1.0).astype(np.float32),
        right=np.clip(right_audio * scale, -1.0, 1.0).astype(np.float32),
        sample_rate_hz=actual_rate,
        is_stereo=stereo,
        pilot_level_db=pilot_db,
    )


# -- Filtering helpers ---------------------------------------------------------------------------


def _filter_length(
    *, transition_hz: float, sample_rate_hz: float, attenuation_db: float, available: int
) -> tuple[int, float]:
    """Work out how long a filter has to be to meet a specification.

    Choosing a tap count by eye is how a filter ends up not filtering. A low-pass at 15 kHz
    built with 129 taps at 480 kHz has a transition region about 6 kHz wide, which puts the
    19 kHz pilot inside it -- so the filter that is supposed to remove the pilot barely touches
    it, and the "mono" output still carries it at full strength. The Kaiser design formula gives
    the length from the transition width and the attenuation wanted, which is the specification
    actually being asked for.

    Returns the tap count, always odd for a symmetric linear-phase filter, and the Kaiser beta.
    The length is capped at what ``filtfilt`` can apply to a signal of ``available`` samples.
    """
    nyquist_hz = sample_rate_hz / 2.0
    numtaps, beta = scipy_signal.kaiserord(attenuation_db, transition_hz / nyquist_hz)
    numtaps = int(numtaps) | 1

    # filtfilt needs the signal to be longer than three times the filter order.
    longest = max(31, ((available // 4) | 1))
    return min(numtaps, longest), float(beta)


def _low_pass(
    signal: RealSignal,
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    transition_hz: float,
    attenuation_db: float = 60.0,
) -> RealSignal:
    """Apply a zero-phase low-pass filter, designed to the given transition and attenuation."""
    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        return signal
    numtaps, beta = _filter_length(
        transition_hz=transition_hz,
        sample_rate_hz=sample_rate_hz,
        attenuation_db=attenuation_db,
        available=signal.size,
    )
    taps = scipy_signal.firwin(
        numtaps, cutoff_hz / nyquist_hz, window=("kaiser", beta), pass_zero=True
    )
    filtered: RealSignal = scipy_signal.filtfilt(taps, 1.0, signal).astype(np.float64)
    return filtered


def _band_pass(
    signal: RealSignal,
    *,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    transition_hz: float,
    attenuation_db: float = 60.0,
) -> RealSignal:
    """Apply a zero-phase band-pass filter, designed to the given transition and attenuation."""
    nyquist_hz = sample_rate_hz / 2.0
    low = max(low_hz / nyquist_hz, 1e-6)
    high = min(high_hz / nyquist_hz, 1.0 - 1e-6)
    if low >= high:
        raise ValueError(
            f"band {low_hz} Hz to {high_hz} Hz is not representable at {sample_rate_hz} Hz"
        )
    numtaps, beta = _filter_length(
        transition_hz=transition_hz,
        sample_rate_hz=sample_rate_hz,
        attenuation_db=attenuation_db,
        available=signal.size,
    )
    taps = scipy_signal.firwin(numtaps, [low, high], window=("kaiser", beta), pass_zero=False)
    filtered: RealSignal = scipy_signal.filtfilt(taps, 1.0, signal).astype(np.float64)
    return filtered


def _resample_to(
    signal: RealSignal, *, from_rate_hz: float, to_rate_hz: float
) -> tuple[RealSignal, float]:
    """Resample by a rational factor.

    The ratio is approximated as a fraction with a bounded denominator, so an awkward pair of
    rates gives a close rate rather than an enormous filter. The rate achieved is returned
    alongside, because it is not always exactly the one asked for.
    """
    if from_rate_hz == to_rate_hz:
        return signal, from_rate_hz

    ratio = Fraction(to_rate_hz / from_rate_hz).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    if up < 1 or down < 1:
        raise ValueError(f"cannot resample from {from_rate_hz} Hz to {to_rate_hz} Hz")
    resampled: RealSignal = scipy_signal.resample_poly(signal, up, down).astype(np.float64)
    return resampled, from_rate_hz * up / down


def _normalising_scale(
    left: RealSignal,
    right: RealSignal,
    *,
    normalise: Normalisation,
    reference_deviation_hz: float,
) -> float:
    """A factor bringing demodulated audio into the range playback expects.

    The demodulator produces hertz of deviation, a number in the tens of thousands. Playback
    wants values between -1 and 1.

    ``"deviation"`` divides by a fixed reference, so the same loudness always comes out at the
    same level. ``"peak"`` scales by the loudest sample in what it was given, which fills the
    range better but makes the gain depend on the block -- and for continuous listening, where
    audio arrives in blocks of a tenth of a second, that is audible as the volume breathing in
    time with the blocks. Hence the fixed reference by default, and peak only for one-shot use.
    """
    if normalise == "deviation":
        if reference_deviation_hz <= 0:
            raise ValueError(f"reference deviation must be positive, got {reference_deviation_hz}")
        return 1.0 / reference_deviation_hz

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 0.0)
    if peak == 0.0:
        return 1.0
    return 0.99 / peak


def _largest_power_of_two(value: int) -> int:
    """The largest power of two not exceeding ``value``, or 0 for a non-positive value."""
    if value <= 0:
        return 0
    return 1 << (value.bit_length() - 1)
