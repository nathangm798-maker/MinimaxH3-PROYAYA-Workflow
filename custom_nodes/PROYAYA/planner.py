"""Pure-Python planning for H3 windows on the native joint AV time grid.

Lengths include overlap.  ``real_frames`` also includes overlap, while the
tail threshold counts only source frames not already covered by a full window.
Planning does not read video frames, load models, or allocate latent tensors.
"""

from dataclasses import asdict, dataclass
import itertools
from numbers import Integral
import re


MIN_WINDOW_FRAMES = 90
FRAME_PERIOD = 17
START_PERIOD = 51


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _window_length(value, name):
    value = _integer(value, name, MIN_WINDOW_FRAMES)
    if (value - MIN_WINDOW_FRAMES) % FRAME_PERIOD:
        raise ValueError(
            f"{name}={value} is not a valid H3 length. Use 90 + 17*n frames "
            "(90, 107, 124, 141, 158, 175, ...); there is no maximum window length."
        )
    return value


def parse_manual_lengths(value):
    """Parse comma/space/newline separated lengths, or an integer sequence."""
    if isinstance(value, str):
        tokens = [token for token in re.split(r"[\s,;，；]+", value.strip()) if token]
        values = []
        for index, token in enumerate(tokens, 1):
            if not re.fullmatch(r"[+-]?[0-9]+", token):
                raise ValueError(
                    f"manual_lengths item {index} ({token!r}) must be an integer frame count."
                )
            values.append(int(token))
    elif isinstance(value, (list, tuple)):
        values = value
    elif value is None:
        values = []
    else:
        raise ValueError("manual_lengths must be text or a list/tuple of integer frame counts.")
    return tuple(_window_length(length, f"manual_lengths item {index}")
                 for index, length in enumerate(values, 1))


def _ceil_window_length(frames):
    return MIN_WINDOW_FRAMES + FRAME_PERIOD * max(
        0, (frames - MIN_WINDOW_FRAMES + FRAME_PERIOD - 1) // FRAME_PERIOD
    )


def reachable_overlaps(frames):
    """Overlap values a boundary can actually land on after ``frames`` frames.

    Starts are aligned to ``START_PERIOD``, so the achievable overlap is
    ``frames - 51k``. A 124-frame window therefore offers only 73 and 22.
    """
    frames = _integer(frames, "frames", 1)
    values = []
    stride = START_PERIOD
    while frames - stride >= 0:
        values.append(frames - stride)
        stride += START_PERIOD
    return tuple(sorted(values))


@dataclass(frozen=True)
class PlannedWindow:
    start: int
    frames: int
    real_frames: int
    pad_frames: int
    overlap: int
    requested_frames: int

    @property
    def end(self):
        return self.start + self.frames


@dataclass(frozen=True)
class Plan:
    total_frames: int
    mode: str
    windows: tuple[PlannedWindow, ...]
    output_frames: int
    sampled_frames: int
    dropped_frames: int
    pad_frames: int
    minimum_overlap: int
    tail_keep_threshold: int
    tail_action: str
    source_frames: int = 0
    target_frames: int = 0

    @property
    def shortfall_frames(self):
        # Target frames the source cannot supply. They are never invented by
        # repeating the last source frame.
        return max(0, self.target_frames - self.source_frames)

    @property
    def window_count(self):
        return len(self.windows)

    @property
    def overlaps(self):
        """Actual overlap per window, starting at the timeline head.

        Item 0 is always 0 (the first window has no predecessor); item i is the
        overlap window i shares with window i-1. Derived from the windows so a
        mixed-overlap plan needs no extra stored state.
        """
        return tuple(window.overlap for window in self.windows)

    def as_dict(self):
        result = asdict(self)
        result["windows"] = [asdict(window) for window in self.windows]
        result["window_count"] = self.window_count
        result["overlaps"] = list(self.overlaps)
        result["shortfall_frames"] = self.shortfall_frames
        result["fps"] = 24
        return result


def _next_start(previous, next_length, overlap_hint, index, label="overlap"):
    stride = START_PERIOD * ((previous.frames - overlap_hint) // START_PERIOD)
    if stride < START_PERIOD:
        raise ValueError(
            f"Window {index}: {label}={overlap_hint} is too large after a "
            f"{previous.frames}-frame window. At least 51 new start frames are required; "
            f"{label} must be <= {previous.frames - START_PERIOD} here."
        )
    overlap = previous.frames - stride
    if next_length <= overlap:
        raise ValueError(
            f"Window {index}: requested length {next_length} cannot advance the timeline "
            f"after the required aligned overlap of {overlap} frames. Its length must "
            f"exceed {overlap}; reducing the stride would only increase overlap."
        )
    return previous.start + stride, overlap


def _check_unused(lengths, used, allow_dropped_tail=False):
    unused = len(lengths) - used
    allowed = 1 if allow_dropped_tail else 0
    if unused > allowed:
        raise ValueError(
            f"manual_lengths has {unused} unused item(s) after {used} generated window(s). "
            "Remove extra lengths; only one final item may correspond to a discarded tail."
        )


def make_plan(source_frames, mode="uniform", window_frames=124, manual_lengths="",
              minimum_overlap=22, tail_keep_threshold=40, target_frames=None,
              overlaps=None):
    """Plan all windows before encoding, including final crop and tail padding.

    ``target_frames`` is the requested output length. Planning always uses
    min(target_frames, source_frames): a target longer than the source never
    creates extra windows from repeated last frames. The unmet part is
    reported as ``shortfall_frames``.

    ``overlaps`` optionally fixes the overlap of every boundary individually:
    item 0 is the boundary between window 1 and 2, item 1 between 2 and 3, and
    so on, so it must hold exactly ``window_count - 1`` items. Every value must
    be reachable (see ``reachable_overlaps``); an unreachable value raises
    instead of being silently snapped to the nearest legal stride. When it is
    omitted, ``minimum_overlap`` drives every boundary exactly as before.

    Manual lengths specify one target length per retained window.  Only the
    actual source tail may shorten a target (and repeat its last source frame to
    reach the native grid).  A retained tail needs its own manual list item;
    the final length is never repeated implicitly.  One trailing unused item is
    allowed only when the source tail is discarded by the threshold.
    """
    source_frames = _integer(source_frames, "source_frames", 1)
    target_frames = source_frames if target_frames is None else _integer(target_frames, "target_frames", 1)
    total_frames = min(source_frames, target_frames)
    minimum_overlap = _integer(minimum_overlap, "minimum_overlap")
    tail_keep_threshold = _integer(tail_keep_threshold, "tail_keep_threshold")
    if overlaps is not None:
        overlaps = tuple(_integer(value, f"overlaps item {index}")
                         for index, value in enumerate(overlaps, 1))
    if mode not in ("uniform", "manual"):
        raise ValueError(f"mode must be 'uniform' or 'manual', got {mode!r}.")
    if mode == "manual":
        lengths = parse_manual_lengths(manual_lengths)
        if not lengths:
            raise ValueError("manual mode requires at least one length in manual_lengths.")
    else:
        window_frames = _window_length(window_frames, "window_frames")
        lengths = ()

    windows = []
    dropped_frames = 0
    tail_action = "none"
    while True:
        previous = windows[-1] if windows else None
        covered_end = previous.end if previous else 0
        uncovered = total_frames - covered_end
        if uncovered <= 0:
            if mode == "manual":
                _check_unused(lengths, len(windows))
            break

        if mode == "manual":
            if len(windows) >= len(lengths):
                if uncovered < tail_keep_threshold:
                    dropped_frames = uncovered
                    tail_action = "dropped"
                    break
                raise ValueError(
                    f"manual_lengths is exhausted after {len(windows)} window(s), with "
                    f"{uncovered} uncovered source frames remaining. Add a target length "
                    "for the next window; the last length is not repeated automatically."
                )
            requested = lengths[len(windows)]
        else:
            requested = window_frames

        if previous is None:
            start, overlap = 0, 0
        else:
            boundary = len(windows) - 1
            if overlaps is None:
                hint, label = minimum_overlap, "minimum_overlap"
            elif boundary >= len(overlaps):
                # Same escape hatch as a rejected stride: a short remainder is a
                # discarded tail, not a boundary the list has to describe.
                if uncovered < tail_keep_threshold:
                    if mode == "manual":
                        _check_unused(lengths, len(windows), allow_dropped_tail=True)
                    dropped_frames = uncovered
                    tail_action = "dropped"
                    break
                raise ValueError(
                    f"overlaps has {len(overlaps)} item(s), which implies "
                    f"{len(overlaps) + 1} window(s), but window {len(windows) + 1} needs "
                    f"item {boundary + 1}: {uncovered} frame(s) of the {total_frames} "
                    f"timeline would stay uncovered.")
            else:
                hint, label = overlaps[boundary], f"overlaps[{boundary}]"
            try:
                start, overlap = _next_start(previous, requested, hint, len(windows) + 1, label)
            except ValueError:
                # No boundary needs to be constructed for a discarded tail.
                if uncovered >= tail_keep_threshold:
                    raise
                if mode == "manual":
                    _check_unused(lengths, len(windows), allow_dropped_tail=True)
                dropped_frames = uncovered
                tail_action = "dropped"
                break
            if overlaps is not None and overlap != hint:
                raise ValueError(
                    f"Boundary {boundary + 1} asks for an overlap of {hint} frames, but after "
                    f"a {previous.frames}-frame window only {overlap} is reachable "
                    f"(reachable: {list(reachable_overlaps(previous.frames))}).")

        if start + requested <= total_frames:
            windows.append(PlannedWindow(start, requested, requested, 0, overlap, requested))
            continue

        # This is the actual video tail, rather than an intermediate short item.
        if uncovered < tail_keep_threshold:
            if previous is None:
                raise ValueError(
                    f"No windows remain: the source has {total_frames} frames, shorter than "
                    f"the first target window, and its {uncovered} new frames are below "
                    f"tail_keep_threshold={tail_keep_threshold}. Lower the tail threshold."
                )
            if mode == "manual":
                _check_unused(lengths, len(windows), allow_dropped_tail=True)
            dropped_frames = uncovered
            tail_action = "dropped"
            break
        real_frames = total_frames - start
        frames = _ceil_window_length(real_frames)
        windows.append(PlannedWindow(start, frames, real_frames, frames - real_frames,
                                     overlap, requested))
        if mode == "manual":
            _check_unused(lengths, len(windows))
        tail_action = "kept"
        break

    if overlaps is not None and len(overlaps) != len(windows) - 1:
        raise ValueError(
            f"overlaps has {len(overlaps)} item(s), which implies {len(overlaps) + 1} "
            f"window(s), but the plan produced {len(windows)} over {total_frames} frames. "
            f"Pass exactly one item per boundary (window count - 1).")

    return Plan(
        total_frames=total_frames, mode=mode, windows=tuple(windows),
        output_frames=total_frames - dropped_frames,
        sampled_frames=windows[-1].end, dropped_frames=dropped_frames,
        pad_frames=sum(window.pad_frames for window in windows),
        minimum_overlap=minimum_overlap, tail_keep_threshold=tail_keep_threshold,
        tail_action=tail_action, source_frames=source_frames, target_frames=target_frames,
    )


@dataclass(frozen=True)
class OverlapSolution:
    """One concrete per-boundary overlap assignment chosen by ``solve_overlaps``."""

    gaps: tuple
    window_count: int
    matches: int
    pad_frames: int
    sampled_frames: int
    considered: int
    window_frames_total: int = 0

    @property
    def boundaries(self):
        return len(self.gaps)

    def summary(self):
        if not self.gaps:
            return "单窗，无边界"
        joined = "、".join(str(gap) for gap in self.gaps)
        return (f"{len(self.gaps)} 条边界 → {joined}；"
                f"符合提示词偏好 {self.matches}/{len(self.gaps)}；"
                f"共采样 {self.window_frames_total} 帧，补帧 {self.pad_frames} 帧"
                f"（试算 {self.considered} 组）")


def solve_overlaps(total_frames, window_frames, window_count, same_flags,
                   same_overlap=22, diff_overlap=73, tail_keep_threshold=40,
                   max_combinations=20000):
    """Pick one overlap per boundary so exactly ``window_count`` windows fit.

    ``same_flags[i]`` is True when window i and window i+1 carry the same prompt
    text. A boundary between identical prompts prefers ``same_overlap``: the
    window does not have to re-synthesize a new state there, so the cheaper
    handoff is enough. A boundary between different prompts prefers
    ``diff_overlap``. Returns ``None`` when no legal layout produces exactly
    ``window_count`` windows, which is a real possibility: with a 124-frame
    window on 328 frames, five windows only fit with every boundary at 73.
    """
    total_frames = _integer(total_frames, "total_frames", 1)
    window_count = _integer(window_count, "window_count", 1)
    tail_keep_threshold = _integer(tail_keep_threshold, "tail_keep_threshold")
    same_overlap = _integer(same_overlap, "same_overlap")
    diff_overlap = _integer(diff_overlap, "diff_overlap")
    same_flags = tuple(bool(flag) for flag in same_flags)
    if len(same_flags) != window_count - 1:
        raise ValueError(
            f"same_flags has {len(same_flags)} item(s) for {window_count} window(s); "
            f"exactly {window_count - 1} boundaries are needed.")
    if not same_flags:
        return OverlapSolution((), 1, 0, 0, min(total_frames, window_frames), 1)

    reachable = reachable_overlaps(window_frames)
    preferred = tuple(sorted({same_overlap, diff_overlap} & set(reachable)))
    options = preferred or reachable
    if len(options) ** len(same_flags) > max_combinations:
        options = reachable
    if len(options) ** len(same_flags) > max_combinations:
        return None

    wanted = [same_overlap if flag else diff_overlap for flag in same_flags]
    best = None
    considered = 0
    for gaps in itertools.product(options, repeat=len(same_flags)):
        considered += 1
        try:
            plan = make_plan(total_frames, "uniform", window_frames, "", 0,
                             tail_keep_threshold, overlaps=gaps)
        except ValueError:
            continue
        if len(plan.windows) != window_count or plan.dropped_frames:
            continue
        # Every window must use the configured length; only the real tail may be
        # shorter and padded up to the native grid. That keeps the temporal
        # meaning of each prompt segment where the caller placed it.
        if any(window.frames != window_frames for window in plan.windows[:-1]):
            continue
        if plan.windows[-1].frames > window_frames:
            continue
        matches = sum(1 for gap, target in zip(gaps, wanted) if gap == target)
        key = (-matches, plan.pad_frames, -sum(gaps), gaps)
        if best is None or key < best[0]:
            best = (key, tuple(gaps), matches, plan)
    if best is None:
        return None
    _, gaps, matches, plan = best
    return OverlapSolution(gaps=gaps, window_count=len(plan.windows), matches=matches,
                           pad_frames=plan.pad_frames, sampled_frames=plan.sampled_frames,
                           considered=considered,
                           window_frames_total=sum(window.frames for window in plan.windows))
