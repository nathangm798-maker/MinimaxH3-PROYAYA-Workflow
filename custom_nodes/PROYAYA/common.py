"""Shared helpers for the PROYAYA nodes."""

import torch
import torch.nn.functional as functional
from comfy_api.latest import io

from .windows import Window, shapes_for_windows


def cpu_tensors(value):
    """Detach nested conditioning from the GPU; keeps tensor structure."""
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tensors(item) for item in value)
    return value


def group(value):
    # An empty V3 Autogrow group is wrapped during lazy-input discovery.
    return value[0] if isinstance(value, tuple) else (value or {})


def ordered(values):
    return sorted(values, key=lambda name: int(name.rsplit("_", 1)[-1]))


def autogrow(name, item, prefix):
    return io.Autogrow.Input(name, optional=True, template=io.Autogrow.TemplatePrefix(
        item, prefix=prefix, min=0, max=100))


def audio_slice(audio, start_frame, frames):
    """Slice paired audio on absolute 24 fps boundaries; silence-pad the tail."""
    sr = audio["sample_rate"]
    start = round(start_frame * sr / 24)
    end = round((start_frame + frames) * sr / 24)
    wave = audio["waveform"][..., start:end]
    if wave.shape[-1] < end - start:
        wave = functional.pad(wave, (0, end - start - wave.shape[-1]))
    return {"waveform": wave, "sample_rate": sr}


def validate_sigmas(sigmas):
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("SIGMAS must be a one-dimensional tensor containing at least two values.")
    if not sigmas.is_floating_point():
        raise ValueError("SIGMAS must use a floating-point dtype.")
    # Keep the caller's schedule untouched; the full solver always runs in FP32
    # on CPU and only its small model-side copy is sent to the inference device.
    schedule = sigmas.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(schedule).all()):
        raise ValueError("SIGMAS must contain only finite values.")
    if not bool((schedule >= 0).all()) or not bool((schedule <= 1).all()):
        raise ValueError("Native H3 flow SIGMAS must stay in the interval [0, 1].")
    if float(schedule[0]) <= 0.0:
        raise ValueError("SIGMAS must start above 0 and at or below 1.")
    if float(schedule[-1]) != 0.0:
        raise ValueError("SIGMAS must end at 0; unfinished denoising schedules are not supported by this node.")
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("SIGMAS must strictly decrease after conversion to CPU FP32.")
    return schedule


def append_window(positive, negative, latent, start_frame, previous_windows=None,
                  *, require_source_video=True):
    """Validate one encoded REF2VA window and append it to the window list."""
    samples = latent["samples"]
    if not samples.is_nested or len(samples.unbind()) != 2:
        raise ValueError("Connect the empty AV LATENT output of native MiniMaxH3ReferenceToVideo.")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[:2] != (1, 24) or audio.shape[:3] != (1, 32, 2):
        raise ValueError("Expected native H3 video [1,24,T,H,W] and audio [1,32,2,T].")
    if video.shape[2] < 2 or video.shape[2] % 5 != 2:
        raise ValueError("The window latent is not on H3's native temporal grid.")
    if "noise_mask" in latent or any(bool(torch.count_nonzero(t)) for t in (video, audio)):
        raise ValueError("Use an empty REF2VA latent. This sampler does not accept masked or pre-generated target latents.")
    for condition in positive + negative:
        metadata = condition[1]
        if metadata.get("minimax_keyframes") or metadata.get("control") is not None or "mask" in metadata:
            raise ValueError("Use independent native REF2VA conditioning without MotionContext, Guide, ControlNet or regional masks.")
    frames = (video.shape[2] - 2) // 5 * 17 + 5
    window = Window(start_frame, frames, video.shape[3], video.shape[4])
    if tuple(audio.shape) != window.shapes[1]:
        raise ValueError("Window audio length does not match the native 24/40 Hz AV grid.")
    if require_source_video and not any(
            any(ref.get("kind") in ("video", "video_audio") and "latent" in ref
                and ref["latent"].shape[2] == window.video_length
                for ref in cond[1].get("minimax_refs", [])) for cond in positive):
        raise ValueError("Each window needs an encoded source video of the full window length. Check its source slice and connect the video VAE to native REF2VA.")
    entries = list(previous_windows or [])
    entries.append({"window": window, "positive": cpu_tensors(positive), "negative": cpu_tensors(negative)})
    shapes_for_windows([entry["window"] for entry in entries])
    return entries