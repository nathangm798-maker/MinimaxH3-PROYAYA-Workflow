"""H3 adaptation of CustomNodeKit's intra-step x0 rolling.

Based on the idea/formula in user2318/ComfyUI-CustomNodeKit,
wan_scail2_rolling.py at 5099660055cfe96a26366150f171129fd5578c39 (MIT).
See THIRD_PARTY_NOTICES.md. Unlike the Wan wrapper, this implementation uses
post-CFG predictions, H3's packed AV grid, and the original global NOISE tensor.
"""

from dataclasses import dataclass
import math

import torch

from .windows import CPUFusion, unpack


ROLLING_MODES = ("resample", "hard", "front", "front_resample")

# Front-curve window: rolling is at full strength at sigma >= FRONT_SIGMA_HIGH,
# decays linearly to zero at FRONT_SIGMA_LOW, and stays off below it. Rationale:
# composition/framing forms at high noise, detail work happens at low noise, so
# context inheritance should act early and exit before the detail phase.
FRONT_SIGMA_HIGH = 0.9
FRONT_SIGMA_LOW = 0.63


@dataclass(frozen=True)
class RollingConfig:
    enabled: bool = True
    strength: float = 1.0
    mode: str = ROLLING_MODES[0]
    front_high: float = FRONT_SIGMA_HIGH
    front_low: float = FRONT_SIGMA_LOW

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("rolling_enabled must be a boolean.")
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise ValueError("rolling_strength must be a number between 0 and 1.")
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError("rolling_strength must be finite and between 0 and 1.")
        if self.mode not in ROLLING_MODES:
            raise ValueError(f"rolling_mode must be one of {ROLLING_MODES}, got {self.mode!r}.")
        for name, value in (("front_high", self.front_high), ("front_low", self.front_low)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"rolling_{name} must be a number between 0 and 1.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"rolling_{name} must be finite and between 0 and 1.")
        if self.front_low >= self.front_high:
            raise ValueError("rolling_front_low must be smaller than rolling_front_high.")

    @property
    def active(self):
        return self.enabled and self.strength > 0

    def transfer(self, clean, epsilon, sigma, noise_scale):
        """Reconstruct the donor x0 at the current noise level.

        'resample' keeps the donor at full strength but rescales it exactly like
        the sampler would (x = (1-sigma)*x0 + sigma*eps), so a donor share of only
        (1-sigma) survives and the step at sigma=1 carries nothing at all.
        'hard' skips the rescale and injects the donor x0 itself, which is the
        only way to transfer more than (1-sigma) per step.
        'front' acts only at high sigma (composition phase), where the same
        rescale objection applies, so it injects the donor x0 like 'hard'.
        'front_resample' combines the front sigma curve with resample
        semantics: the donor is re-noised to the current sigma, so the latent
        amplitude always matches the schedule (no washed-out band), at the
        cost of a (1-sigma) per-step transfer bound.
        """
        if self.mode in ("hard", "front"):
            return clean
        return (1.0 - sigma) * clean + sigma * (noise_scale * epsilon)

    def donor_share(self, sigma):
        """Fraction of the donor x0 that survives the transfer at this sigma."""
        if self.mode in ("hard", "front"):
            return 1.0
        return 1.0 - sigma

    def beta(self, sigma):
        # Flat curve: the configured strength at every sigma. CustomNodeKit's
        # 'late' curve (strength * min(1, 2*(1-sigma))) silently multiplies the
        # transfer by (1-sigma) on top of the (1-sigma) already present in the
        # resampling formula, so it decays as (1-sigma)**2 and leaves a short
        # schedule (4 steps at shift 12, all of them at sigma >= 0.8) with
        # almost no transfer at all. Flat removes the extra factor.
        # Note: in resample mode the donor value at sigma == 1 is sigma*epsilon,
        # i.e. pure noise, so step 1 stays a no-op by construction; see
        # donor_share(). Hard mode is the only way past that.
        if self.mode in ("front", "front_resample") and self.active:
            if sigma >= self.front_high:
                return self.strength
            if sigma <= self.front_low:
                return 0.0
            span = self.front_high - self.front_low
            return self.strength * (sigma - self.front_low) / span
        return self.strength if self.active else 0.0


def _time_slice(stream, start, length, kind):
    dim = 2 if kind == 0 else 3
    return stream.narrow(dim, start, length)


def _start(window, kind):
    return window.video_start if kind == 0 else window.audio_start


class CPURollingFusion(CPUFusion):
    """Forward-only commit to local input copies; global state is never changed.

    CPUFusion validates increasing starts AND increasing ends. Therefore every
    frame shared with any earlier window is covered by the immediate predecessor.
    Keeping its overlap with the next window gives the donor's latest-writer
    rule without retaining a complete window or a full-timeline cache.
    """

    def __init__(self, windows, config, blend=0.0):
        super().__init__(windows, blend=blend)
        self.config = config
        self.noise = None
        self.sigma = None
        self.noise_scale = 1.0
        self.steps = []

    def start_run(self, noise, noise_scale=1.0):
        if noise.device.type != "cpu" or noise.dtype != torch.float32:
            raise ValueError("Rolling noise must remain CPU float32 in H3 solver space.")
        unpack(noise, self.shapes)
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("H3 rolling noise_scale must be finite and nonnegative.")
        self.noise_scale = float(noise_scale)
        self.noise = noise  # borrow the sampler noise; do not regenerate/copy it
        self.sigma = None
        self.steps = []

    def finish_run(self):
        self.noise = None
        self.sigma = None

    def predict(self, state, predictor, order=None):
        sigma = self.sigma
        if sigma is None or not math.isfinite(sigma) or not 0 <= sigma <= 1:
            raise ValueError("Rolling requires the current H3 sigma in [0, 1].")
        beta = self.config.beta(sigma)
        # donor_share is how much of the donor window's x0 reaches this step, so
        # beta * donor_share bounds the usable transfer.
        stats = {"sigma": float(sigma), "effective_strength": beta,
                 "donor_share": self.config.donor_share(sigma),
                 "mode": self.config.mode,
                 "window_predictions": 0, "video_latent_positions": 0,
                 "audio_latent_positions": 0}
        # Exact V1 path: no extra arithmetic or RNG, including single-window runs.
        if beta == 0 or len(self.windows) == 1:
            result = super().predict(state, predictor, order=order)
            stats["window_predictions"] = len(self.windows)
            self.steps.append(stats)
            return result
        if order is not None and list(order) != list(range(len(self.windows))):
            raise ValueError("Intra-step rolling requires chronological window order.")
        if self.noise is None:
            raise ValueError("Rolling requires the original global noise for this run.")
        noise_streams = unpack(self.noise, self.shapes)
        previous = None

        def rolling_predict(index, local):
            nonlocal previous
            window = self.windows[index]
            if previous is not None:
                local_streams = unpack(local, window.shapes)
                for kind, (current, old, eps) in enumerate(zip(local_streams, previous, noise_streams)):
                    start = _start(window, kind)
                    dim = 2 if kind == 0 else 3
                    length = old.shape[dim]
                    if length <= 0:
                        continue
                    shared = _time_slice(current, 0, length, kind)
                    clean = old
                    epsilon = _time_slice(eps, start, length, kind)
                    target = self.config.transfer(clean, epsilon, sigma, self.noise_scale)
                    shared.copy_((1.0 - beta) * shared + beta * target)
                    key = "video_latent_positions" if kind == 0 else "audio_latent_positions"
                    stats[key] += length
            prediction = predictor(index, local)
            predicted = unpack(prediction.detach(), window.shapes)
            previous = None
            if index + 1 < len(self.windows):
                next_window = self.windows[index + 1]
                previous = []
                for kind, value in enumerate(predicted):
                    dim = 2 if kind == 0 else 3
                    offset = _start(next_window, kind) - _start(window, kind)
                    length = min(value.shape[dim] - offset, next_window.shapes[kind][dim])
                    # clone: a view would retain the entire packed AV storage.
                    previous.append(_time_slice(value, offset, length, kind).clone())
            stats["window_predictions"] += 1
            return prediction

        try:
            result = super().predict(state, rolling_predict)
            self.steps.append(stats)
            return result
        finally:
            previous = None

    def report(self):
        return {
            "version": "1.1",
            "mechanism": "intra_step_x0_rolling",
            "enabled": self.config.enabled,
            "strength": self.config.strength,
            "mode": self.config.mode,
            "curve": ("front: strength at sigma>=%.2f, linear to 0 at %.2f"
                      % (self.config.front_high, self.config.front_low)
                      if self.config.mode in ("front", "front_resample")
                      else "flat: strength at every sigma"),
            "transfer_bound": ("beta" if self.config.mode in ("hard", "front")
                               else "beta * (1 - sigma)"),
            "noise": "reuse_original_global_av_noise",
            "noise_scale": self.noise_scale,
            "commit": "previous_window_post_cfg_x0; reset_each_sigma",
            "fusion": "V1_linear_weighted_average",
            "streams": "video_and_audio_in_packed_solver_space",
            "steps": list(self.steps),
            "video_latent_positions": sum(s["video_latent_positions"] for s in self.steps),
            "audio_latent_positions": sum(s["audio_latent_positions"] for s in self.steps),
        }
