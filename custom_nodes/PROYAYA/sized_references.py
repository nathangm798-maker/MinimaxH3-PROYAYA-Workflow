"""Native H3 REF2VA assembly with reference sizes owned by the caller.

Only the native 32-pixel spatial alignment remains. Images are not matched to
the output area or capped at 2048; videos do not pass through adapt_canvas.
The tokenizer retains its own native vision preprocessing and pixel limits.

Optional ``cache`` / ``provenance`` arguments let the caller reuse VAE encodes
across windows and across executions. Provenance is supplied by the caller
because the frames handed in are already sliced copies: only the caller knows
which source video and which offset they came from.
"""

import node_helpers
import torch
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE, FPS, _empty_av_latent, _encode_ref_audio, _resize,
)

from .ref_cache import MISSING, Identity, model_identity


def _aligned_pixels(frames):
    height, width = frames.shape[1:3]
    aligned_width = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    aligned_height = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    if (width, height) == (aligned_width, aligned_height):
        return frames[..., :3], aligned_width, aligned_height
    return (_resize(frames, aligned_width, aligned_height, "disabled"),
            aligned_width, aligned_height)


def _source(provenance, group, name, fallback):
    """Stable cache key for one material; falls back to the tensor itself."""
    table = (provenance or {}).get(group) or {}
    known = table.get(name)
    return known if known is not None else (group, Identity(fallback))


def _detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    return value


def _cached(cache, key, produce):
    """Encode once, then reuse.

    The stored copy lives on the CPU: reference latents are consumed as CPU
    tensors by ``append_window`` anyway, so caching on the GPU would only pin
    VRAM for the whole run.
    """
    if cache is None:
        return produce()
    hit = cache.get(key, "l1")
    if hit is not MISSING:
        return hit
    value = _detach_to_cpu(produce())
    cache.put(key, value)
    return value


def encode_sized_references(clip, prompt, width, height, length, vae=None, audio_vae=None,
                            ref_images=None, ref_videos=None, ref_video_audios=None,
                            ref_audios=None, cache=None, provenance=None) -> io.NodeOutput:
    """Use the native REF2VA presentation and payload without reference downscaling.

    Aligned reference dimensions are preserved independently of the output
    canvas. Other dimensions round to the nearest multiple of 32, at least 32.
    Image batches contribute their first image, and videos are trimmed to the
    target window and then down to 5+17k frames, matching native REF2VA behavior.

    ``provenance`` maps the input group and key (``images``, ``videos``,
    ``video_audios``, ``audios``) to a hashable description of where that
    material came from — typically ``("video", Identity(video), window_start)``.
    Entries without provenance are keyed on the tensor they were given, which is
    still correct but never reaches across windows.
    """
    latent, frame_count = _empty_av_latent(width, height, length)
    ref_items = []
    ref_blocks = []
    vae_key = model_identity(vae)
    audio_key = model_identity(audio_vae)

    for name, image in (ref_images or {}).items():
        if image is None:
            continue
        pixels, image_width, image_height = _aligned_pixels(image[:1])
        ref_items.append({"type": "image", "data": pixels})
        if vae is not None:
            encoded = _cached(
                cache, (vae_key, "image", _source(provenance, "images", name, image),
                        image_width, image_height),
                lambda pixels=pixels: vae.encode(pixels))
            ref_blocks.append({"kind": "image", "latent_h": image_height // 16,
                               "latent_w": image_width // 16, "latent": encoded})

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        frame_length = min(video_frames.shape[0], frame_count)
        if frame_length < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        frame_length = 5 + ((frame_length - 5) // 17) * 17
        frames, video_width, video_height = _aligned_pixels(video_frames[:frame_length])
        if soundtrack is not None:
            ref_items.append({"type": "audio"})
        sample_indices = list(range(0, frame_length, FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_indices],
                          "timestamps": [index / 2.0 for index in range(len(sample_indices))]})
        if vae is None:
            continue
        source = _source(provenance, "videos", name, video_frames)
        encoded = _cached(
            cache, (vae_key, "video", source, frame_length, video_width, video_height),
            lambda frames=frames: vae.encode(frames))
        audio_latent, ref_audio_t = None, 0
        if soundtrack is not None and audio_vae is not None:
            paired = _cached(
                cache, (audio_key, "video_audio", source, frame_length),
                lambda soundtrack=soundtrack: _encode_ref_audio(audio_vae, soundtrack))
            audio_latent, ref_audio_t = paired
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": encoded.shape[2], "latent_h": video_height // 16,
                           "latent_w": video_width // 16, "ref_audio_t": ref_audio_t,
                           "latent": encoded, "audio_latent": audio_latent})

    for name, audio in (ref_audios or {}).items():
        if audio is None:
            continue
        ref_items.append({"type": "audio"})
        if audio_vae is not None:
            wave = audio["waveform"]
            paired = _cached(
                cache, (audio_key, "ref_audio",
                        _source(provenance, "audios", name, wave),
                        audio["sample_rate"], int(wave.shape[-1])),
                lambda audio=audio: _encode_ref_audio(audio_vae, audio))
            audio_latent, ref_audio_t = paired
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
    return io.NodeOutput(conditioning, latent)
