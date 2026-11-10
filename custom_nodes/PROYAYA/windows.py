"""H3's native AV time grid and CPU overlap fusion."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Window:
    start: int
    frames: int
    height: int
    width: int

    def __post_init__(self):
        if self.start < 0 or self.start % 51:
            raise ValueError("H3 joint AV windows must start at a multiple of 51 frames (0, 102, 204 for the three-window preset).")
        if self.frames < 5 or self.frames % 17 != 5:
            raise ValueError("Window length must be 17k+5 frames.")
        if self.height < 1 or self.width < 1:
            raise ValueError("Invalid latent spatial dimensions.")

    @property
    def video_start(self):
        return self.start // 17 * 5

    @property
    def video_length(self):
        return 2 + (self.frames - 5) // 17 * 5

    @property
    def audio_start(self):
        return self.start * 5 // 3

    @property
    def audio_length(self):
        return round(self.frames * 5 / 3)

    @property
    def shapes(self):
        return [(1, 24, self.video_length, self.height, self.width), (1, 32, 2, self.audio_length)]


def shapes_for_windows(windows):
    if not windows or windows[0].start != 0:
        raise ValueError("The first joint window must start at source frame 0.")
    previous = None
    for window in windows:
        if (window.height, window.width) != (windows[0].height, windows[0].width):
            raise ValueError("All windows must have the same output resolution.")
        if previous is not None:
            if not previous.start < window.start < previous.start + previous.frames:
                raise ValueError("Windows must be in increasing order with a nonempty overlap and no gap.")
            if window.start + window.frames <= previous.start + previous.frames:
                raise ValueError("Each additional window must extend the output timeline.")
        previous = window
    last = windows[-1]
    return [(1, 24, last.video_start + last.video_length, last.height, last.width),
            (1, 32, 2, last.audio_start + last.audio_length)]


def unpack(packed, shapes):
    streams = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        streams.append(packed.reshape(-1)[offset:offset + size].view(shape))
        offset += size
    if offset != packed.numel():
        raise ValueError("Packed AV size does not match its declared shapes.")
    return streams


def pack(streams):
    return torch.cat([stream.reshape(1, 1, -1) for stream in streams], dim=-1)


def crop(streams, window):
    video, audio = streams
    return [video[:, :, window.video_start:window.video_start + window.video_length],
            audio[..., window.audio_start:window.audio_start + window.audio_length]]


BLEND_MIN = -10.0
BLEND_MAX = 10.0


def _blend_bias(blend):
    """Map the UI slider [-10, 10] onto a [-1, 1] bias for the newer-window ramp.

    Positive values push the hand-out toward the newer window, negative values
    keep the overlap with the older one.
    """
    value = float(blend)
    if not math.isfinite(value) or value < BLEND_MIN or value > BLEND_MAX:
        raise ValueError(f"blend_bias must be within [{BLEND_MIN:g}, {BLEND_MAX:g}], got {blend!r}.")
    return value / BLEND_MAX


def _weights(windows, index, stream, blend=0.0):
    """Linear cross-window ramps, optionally biased toward one side of each seam.

    The stock ramp hands the seam from the older window to the newer one: the
    head of window i is ``(j+1)/(o+1)`` and its tail is ``(o-j)/(o+1)``, so
    adjacent windows are exactly complementary. ``blend`` shifts the two ramps
    in opposite directions and clamps, which keeps the pair complementary while
    moving where the hand-out happens:

    * ``blend = -10`` -> the older window keeps 100% of the overlap for its whole
      length; the newer window contributes nothing before the seam ends.
    * ``blend = 0``   -> the stock linear cross-fade.
    * ``blend = +10`` -> the newer window takes 100% of the overlap from its very
      first frame; the older window contributes nothing inside it.

    Clamping both sides at once is safe: for ``a + c == 1``, ``c + b >= 1``
    implies ``a - b <= 0``, so an overlap frame is either untouched (sum 1) or
    saturated to (1, 0) / (0, 1). CPUFusion divides by the summed weights, so
    coverage is never lost at any slider position.
    """
    bias = _blend_bias(blend)
    starts = [w.video_start if stream == 0 else w.audio_start for w in windows]
    lengths = [w.video_length if stream == 0 else w.audio_length for w in windows]
    start, length = starts[index], lengths[index]
    weight = torch.ones(length, dtype=torch.float32, device="cpu")
    if index:
        # Head: this window is the newer side, so +bias favours it.
        overlap = min(length, starts[index - 1] + lengths[index - 1] - start)
        ramp = torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)
        weight[:overlap] *= (ramp + bias).clamp(0.0, 1.0)
    if index + 1 < len(windows):
        # Tail: this window is the older side, so -bias yields the overlap away.
        overlap = min(length, start + length - starts[index + 1])
        ramp = torch.arange(overlap, 0, -1, dtype=torch.float32) / (overlap + 1)
        weight[-overlap:] *= (ramp - bias).clamp(0.0, 1.0)
    return weight.view(1, 1, length, 1, 1) if stream == 0 else weight.view(1, 1, 1, length)


class CPUFusion:
    def __init__(self, windows, blend=0.0):
        self.windows = tuple(windows)
        self.blend = float(blend)
        self.shapes = shapes_for_windows(windows)
        self.weights = [(_weights(windows, i, 0, self.blend), _weights(windows, i, 1, self.blend))
                        for i in range(len(windows))]
        self.counts = [torch.zeros((1, 1, self.shapes[0][2], 1, 1), device="cpu"),
                       torch.zeros((1, 1, 1, self.shapes[1][-1]), device="cpu")]
        for window, weights in zip(windows, self.weights):
            self.counts[0][:, :, window.video_start:window.video_start + window.video_length].add_(weights[0])
            self.counts[1][..., window.audio_start:window.audio_start + window.audio_length].add_(weights[1])
        if any(not bool((count > 0).all()) for count in self.counts):
            raise ValueError("Joint windows leave an uncovered AV time position.")

    def predict(self, state, predictor, order=None):
        if state.device.type != "cpu" or state.dtype != torch.float32:
            raise ValueError("The global sampler state must remain FP32 on CPU.")
        streams = unpack(state, self.shapes)
        result = torch.zeros_like(state, device="cpu")
        totals = unpack(result, self.shapes)
        indices = range(len(self.windows)) if order is None else order
        for index in indices:
            window = self.windows[index]
            # Packing copies this crop, so a model cannot mutate the global snapshot.
            local = pack(crop(streams, window))
            prediction = predictor(index, local)
            if prediction.device.type != "cpu" or prediction.dtype != torch.float32:
                raise ValueError("Window predictions must return FP32 CPU tensors.")
            if prediction.shape != local.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError(f"Window {index + 1} returned a wrong shape or non-finite prediction.")
            predicted = unpack(prediction, window.shapes)
            destinations = crop(totals, window)
            for destination, value, weight in zip(destinations, predicted, self.weights[index]):
                destination.addcmul_(value, weight)
            del prediction, predicted, destinations, local
        for total, count in zip(totals, self.counts):
            total.div_(count)
        return result
