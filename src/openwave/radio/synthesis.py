"""Synthetic FM broadcast signals.

This module generates FM transmissions that do not exist. It is not a toy: with no receiver
available to the maintainer, it is how every detector and demodulator in OpenWave is tested, and
how a test can state the answer in advance -- inject a stereo station at 98.0 MHz, 20 dB above
the noise, and assert that the scanner finds exactly that.

**How the phase is computed.** The stereo multiplex is built as a sum of sinusoids, which means
its integral has a closed form, which means the FM phase can be evaluated directly at any sample
index::

    m(t)  = sum over i of  a_i cos(2*pi*f_i*t + p_i)
    phi(t) = 2*pi*dev * integral of m  =  dev * sum of (a_i/f_i) sin(2*pi*f_i*t + p_i)

Two useful consequences follow. Successive reads are continuous in phase without carrying any
state, because sample index 5000 evaluates to the same thing whether it was reached in one read
or in five. And the output is exactly reproducible, because nothing accumulates.

**What it does not model.** A synthetic signal is clean in ways the air never is: no multipath,
no adjacent-channel splatter, no tuner nonlinearity, no drifting oscillator. Code that passes
against this module can still fail on a rooftop antenna, which is why ``hardware validation``
issues exist. See :doc:`/architecture`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from openwave.core.units import dbfs_to_amplitude
from openwave.radio.constants import (
    FM_MAX_DEVIATION_HZ,
    FM_MPX_BANDWIDTH_HZ,
    RDS_DEVIATION_SHARE,
    STEREO_AUDIO_DEVIATION_SHARE,
    STEREO_PILOT_DEVIATION_SHARE,
    STEREO_PILOT_HZ,
    STEREO_SUBCARRIER_HZ,
)
from openwave.radio.rds.encoder import RdsBaseband, RdsProgramme, rds_baseband_period
from openwave.sdr.device import IqSamples


@dataclass(frozen=True, slots=True)
class Tone:
    """One sinusoidal component of a baseband multiplex signal.

    The amplitude may be negative, which is the same as a phase shift of pi but keeps the
    trigonometric expansions below readable.
    """

    freq_hz: float
    amplitude: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.freq_hz <= 0:
            raise ValueError(f"tone frequency must be positive, got {self.freq_hz}")


@dataclass(frozen=True, slots=True)
class SyntheticFmStation:
    """An FM broadcast transmission to synthesise.

    Satisfies the ``SignalSource`` protocol in :mod:`openwave.sdr.mock`, so a
    :class:`~openwave.sdr.mock.MockSdrDevice` can be handed a list of these and will produce the
    band they would make together.

    Example::

        station = SyntheticFmStation(freq_hz=98.0e6, power_dbfs=-20.0, stereo=True)

    Attributes:
        freq_hz: Carrier frequency.
        power_dbfs: Carrier level in dB relative to full scale. FM has a constant envelope, so
            this is both the peak and the mean power, whatever the modulation is doing.
        left_tone_hz: Frequency of the test tone in the left channel, or the mono audio.
        right_tone_hz: Frequency of the test tone in the right channel. ``None`` means the same
            tone as the left, which gives a silent difference channel -- a stereo transmission
            carrying identical audio on both sides.
        stereo: Whether to transmit a stereo multiplex, with the 19 kHz pilot and the 38 kHz
            difference subcarrier, rather than plain mono audio.
        deviation_hz: Peak carrier deviation.
        name: The station name to send over RDS. Setting it makes the transmission carry an
            RDS subcarrier, so a decoder can read the name back. ``None`` sends no RDS, which
            is what a station without it looks like.
        rds: Full control over what the RDS carries, for a test that needs a particular
            programme identification, programme type or RadioText. Takes precedence over
            :attr:`name`.
    """

    freq_hz: float
    power_dbfs: float = -20.0
    left_tone_hz: float = 1000.0
    right_tone_hz: float | None = None
    stereo: bool = False
    deviation_hz: float = FM_MAX_DEVIATION_HZ
    name: str | None = None
    rds: RdsProgramme | None = None

    #: Cached multiplex expansion, filled on first use. Excluded from equality and repr.
    _tones: list[Tone] = field(
        default_factory=list, init=False, repr=False, compare=False, hash=False
    )

    #: Cached RDS baseband, one repeat of the group sequence, keyed by sample rate.
    _rds_cache: dict[float, RdsBaseband] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if self.freq_hz <= 0:
            raise ValueError(f"carrier frequency must be positive, got {self.freq_hz}")
        if self.power_dbfs > 0:
            raise ValueError(
                f"power must not exceed full scale, got {self.power_dbfs} dBFS; "
                "0 dBFS is an amplitude of 1.0 and anything above it would clip"
            )
        if self.left_tone_hz <= 0:
            raise ValueError(f"left tone must be positive, got {self.left_tone_hz}")
        if self.right_tone_hz is not None and self.right_tone_hz <= 0:
            raise ValueError(f"right tone must be positive, got {self.right_tone_hz}")
        if self.deviation_hz <= 0:
            raise ValueError(f"deviation must be positive, got {self.deviation_hz}")

    # -- Derived properties -------------------------------------------------------------------

    @property
    def amplitude(self) -> float:
        """Carrier amplitude, where 1.0 is full scale."""
        return dbfs_to_amplitude(self.power_dbfs)

    @property
    def rds_programme(self) -> RdsProgramme | None:
        """What this station transmits over RDS, or ``None`` if it transmits none."""
        if self.rds is not None:
            return self.rds
        if self.name is not None:
            return RdsProgramme(programme_service=self.name)
        return None

    @property
    def carries_rds(self) -> bool:
        """Whether this transmission includes an RDS subcarrier."""
        return self.rds_programme is not None

    @property
    def occupied_bandwidth_hz(self) -> float:
        """Bandwidth the transmission occupies, by Carson's rule.

        Carson's rule is ``2 * (peak deviation + highest baseband frequency)``. It accounts for
        about 98% of the transmitted power, which is the usual engineering definition of
        occupied bandwidth.
        """
        if self.stereo or self.carries_rds:
            highest_baseband_hz = FM_MPX_BANDWIDTH_HZ
        else:
            highest_baseband_hz = self.left_tone_hz
        return 2.0 * (self.deviation_hz + highest_baseband_hz)

    @property
    def _tone_share(self) -> float:
        """How much of the deviation the tones may use, leaving room for RDS."""
        return 1.0 - RDS_DEVIATION_SHARE if self.carries_rds else 1.0

    def mpx_tones(self) -> tuple[Tone, ...]:
        """The baseband multiplex, expanded into sinusoids.

        Amplitudes are scaled so that the sum of their magnitudes is 1. That guarantees the
        multiplex never exceeds unity, and therefore that the peak deviation never exceeds
        :attr:`deviation_hz`.

        The trade-off is deliberate: a real transmitter deviates further than this, because it
        exploits the fact that the sum signal, the difference signal and the pilot almost never
        peak at the same instant, and limits the few occasions when they do. Scaling by the
        worst case instead keeps the generator exact and stateless, and preserves the *ratios*
        between the components -- which is what a stereo detector actually keys on.
        """
        if not self._tones:
            self._tones.extend(_normalise(self._raw_mpx_tones(), total=self._tone_share))
        return tuple(self._tones)

    def _raw_mpx_tones(self) -> list[Tone]:
        """The multiplex before normalisation."""
        if not self.stereo:
            return [Tone(freq_hz=self.left_tone_hz, amplitude=1.0)]

        left_hz = self.left_tone_hz
        right_hz = self.right_tone_hz if self.right_tone_hz is not None else self.left_tone_hz
        sum_share = STEREO_AUDIO_DEVIATION_SHARE
        diff_share = STEREO_AUDIO_DEVIATION_SHARE

        # ITU-R BS.450 sends the pilot as sin(wt) and the subcarrier as sin(2wt), so both are
        # cosines delayed by a quarter-turn. The phase matters: a decoder that assumed cosines
        # would pass against a generator that made the same assumption, and fail on the air.
        sine = -np.pi / 2.0
        tones = [
            # The sum signal, L+R, transmitted at baseband.
            Tone(freq_hz=left_hz, amplitude=sum_share),
            Tone(freq_hz=right_hz, amplitude=sum_share),
            # The pilot, which is the only thing that marks the transmission as stereo.
            Tone(freq_hz=STEREO_PILOT_HZ, amplitude=STEREO_PILOT_DEVIATION_SHARE, phase_rad=sine),
        ]

        # The difference signal, L-R, double-sideband suppressed-carrier on 38 kHz. Expanding
        # cos(a)sin(b) into [sin(b+a) + sin(b-a)] / 2 puts it in terms of plain sinusoids, which
        # is what keeps the FM phase integral in closed form.
        for audio_hz, sign in ((left_hz, 1.0), (right_hz, -1.0)):
            half = sign * diff_share / 2.0
            tones.append(
                Tone(freq_hz=STEREO_SUBCARRIER_HZ + audio_hz, amplitude=half, phase_rad=sine)
            )
            tones.append(
                Tone(freq_hz=STEREO_SUBCARRIER_HZ - audio_hz, amplitude=half, phase_rad=sine)
            )
        return tones

    # -- Generation ---------------------------------------------------------------------------

    def samples(
        self,
        *,
        center_freq_hz: float,
        sample_rate_hz: float,
        start_index: int,
        count: int,
    ) -> IqSamples:
        """Generate ``count`` complex samples of this transmission.

        Args:
            center_freq_hz: Frequency the receiver is tuned to. The carrier appears at the
                difference between this and :attr:`freq_hz`.
            sample_rate_hz: Sampling rate.
            start_index: Absolute sample index of the first sample, counted from the start of
                the receiver's timeline. Passing consecutive ranges produces a continuous
                signal; passing the same range twice produces identical samples.
            count: How many samples to generate.

        Returns:
            Complex baseband samples, not band-limited. The caller applies whatever the
            receiver's anti-aliasing filter would do.
        """
        if count <= 0:
            raise ValueError(f"sample count must be positive, got {count}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample rate must be positive, got {sample_rate_hz}")
        if start_index < 0:
            raise ValueError(f"start index must not be negative, got {start_index}")

        indices = start_index + np.arange(count, dtype=np.int64)
        t: npt.NDArray[np.float64] = indices.astype(np.float64) / sample_rate_hz
        offset_hz = self.freq_hz - center_freq_hz
        phase = 2.0 * np.pi * offset_hz * t + self._fm_phase(t)
        phase = phase + self._rds_phase(indices, sample_rate_hz=sample_rate_hz)
        iq: IqSamples = (self.amplitude * np.exp(1j * phase)).astype(np.complex64)
        return iq

    def _rds_phase(
        self, indices: npt.NDArray[np.int64], *, sample_rate_hz: float
    ) -> npt.NDArray[np.float64]:
        """Phase contributed by the RDS subcarrier, or zeros for a station without one.

        Unlike the tones, RDS data has no closed-form integral, so one period of the subcarrier
        and of its integral is generated once and looked into by sample index. That keeps the
        signal exactly reproducible: the same index always gives the same phase, however the
        reads are chunked.
        """
        programme = self.rds_programme
        if programme is None:
            return np.zeros(indices.size, dtype=np.float64)

        baseband = self._rds_baseband(programme, sample_rate_hz=sample_rate_hz)
        deviation_hz = self.deviation_hz * RDS_DEVIATION_SHARE
        phase: npt.NDArray[np.float64] = 2.0 * np.pi * deviation_hz * baseband.phase_at(indices)
        return phase

    def _rds_baseband(self, programme: RdsProgramme, *, sample_rate_hz: float) -> RdsBaseband:
        """One period of this station's RDS subcarrier, generated once and kept."""
        cached = self._rds_cache.get(sample_rate_hz)
        if cached is None:
            cached = rds_baseband_period(programme, sample_rate_hz=sample_rate_hz)
            self._rds_cache[sample_rate_hz] = cached
        return cached

    def _fm_phase(self, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Phase contributed by the modulation, evaluated directly from the closed-form integral.

        For each multiplex tone ``a*cos(2*pi*f*t + p)``, the integral is
        ``a*sin(2*pi*f*t + p) / (2*pi*f)``, and the phase is ``2*pi*deviation`` times the sum of
        those. The constant of integration is dropped: a fixed phase offset is not observable.
        """
        phase: npt.NDArray[np.float64] = np.zeros_like(t)
        for tone in self.mpx_tones():
            phase += (tone.amplitude / tone.freq_hz) * np.sin(
                2.0 * np.pi * tone.freq_hz * t + tone.phase_rad
            )
        return self.deviation_hz * phase


def _normalise(tones: list[Tone], *, total: float = 1.0) -> list[Tone]:
    """Scale tones so the magnitudes of their amplitudes sum to ``total``.

    ``total`` is less than one for a station carrying RDS, which needs its own share of the
    deviation. Without that, adding RDS would push the peak deviation past what was asked for.
    """
    current = sum(abs(tone.amplitude) for tone in tones)
    if current == 0:
        raise ValueError("multiplex has no energy")
    scale = total / current
    return [
        Tone(freq_hz=tone.freq_hz, amplitude=tone.amplitude * scale, phase_rad=tone.phase_rad)
        for tone in tones
    ]
