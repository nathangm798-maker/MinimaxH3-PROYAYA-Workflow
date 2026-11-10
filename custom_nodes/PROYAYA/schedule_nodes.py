"""Exact start-sigma on native scheduler curves.

The `PROYYBasicScheduler` node that used to wrap this helper has been removed:
the official **BasicScheduler** (`comfy_extras/nodes_custom_sampler.py`) covers
the scheduler/steps case, and **ManualSigmas** covers the exact-start case by
taking the sigma list directly. Keeping a third node with the same job only
gave two ways to build the same SIGMAS payload.

`schedule_from_start` stays because it is the tested implementation of
"resample the remaining native curve from a chosen sigma" and is the reference
for hand-built ManualSigmas lists.
"""

import math
from numbers import Integral, Real

import torch
import comfy.samplers

from .common import validate_sigmas


def schedule_from_start(model_sampling, scheduler, steps, start_sigma):
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if isinstance(start_sigma, bool) or not isinstance(start_sigma, Real) or not 0 < start_sigma <= 1:
        raise ValueError("H3 start_sigma must be in (0, 1].")
    if steps == 1:
        # There is no intermediate curve to preserve. Some native schedulers
        # require at least two steps to compute their interior knots.
        return torch.tensor([start_sigma, 0.0], dtype=torch.float32)
    native = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).detach().cpu().float()
    values = native.tolist()
    if (len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values)
            or values[-1] != 0 or any(a < b for a, b in zip(values, values[1:]))):
        raise ValueError("The selected scheduler must produce a finite, descending H3 schedule ending at zero.")
    start = float(torch.tensor(start_sigma, dtype=torch.float32))
    if start == values[0] and len(values) == steps + 1 and all(a > b for a, b in zip(values, values[1:])):
        return native.clone()
    if start > values[0]:
        values.insert(0, start)
    index = next(i for i in range(len(values) - 1) if values[i] >= start > values[i + 1])
    start_position = index + (values[index] - start) / (values[index] - values[index + 1])
    end_position = len(values) - 1
    result = []
    for step in range(steps + 1):
        position = start_position + (end_position - start_position) * step / steps
        left = min(math.floor(position), len(values) - 2)
        fraction = position - left
        result.append(values[left] * (1 - fraction) + values[left + 1] * fraction)
    result[0], result[-1] = start, 0.0
    # Low-resolution native grids (e.g. simple) may repeat values at huge step
    # counts. Interpolate their distinct knots instead of emitting zero dt.
    schedule = torch.tensor(result, dtype=torch.float32)
    if not bool((schedule[:-1] > schedule[1:]).all()):
        knots = [start] + [s for i, s in enumerate(values) if s < start and (i == 0 or s != values[i - 1])]
        result = []
        for step in range(steps + 1):
            position = (len(knots) - 1) * step / steps
            left = min(math.floor(position), len(knots) - 2)
            fraction = position - left
            result.append(knots[left] * (1 - fraction) + knots[left + 1] * fraction)
        result[0], result[-1] = start, 0.0
        schedule = torch.tensor(result, dtype=torch.float32)
    return validate_sigmas(schedule)
