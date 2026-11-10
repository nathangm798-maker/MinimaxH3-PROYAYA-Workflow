"""Experimental, single-pass modality gains for native H3 REF2VA.

One preserves the native path; zero removes an input before either encoder.
Other values scale selected, already contextualized Qwen hidden rows and the
corresponding VAE latents. They are not calibrated or independent attention
weights: Qwen has already mixed reference information into other hidden rows.
No token-weight tuples, model patches, attention overrides or extra denoising
branches are used. A near-zero gain is not equivalent to removing a modality.
"""

import math
from numbers import Integral

import torch
from comfy_api.latest import io
from comfy.text_encoders.minimax import VISION_END, VISION_START
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo


def _checked_weight(value, name):
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite, nonnegative number.") from exc
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    return weight


def _native_entries(tokens):
    """Refuse another tokenizer format instead of guessing hidden-row offsets."""
    if not isinstance(tokens, dict) or set(tokens) != {"qwen3vl_32b"}:
        raise ValueError("Experimental H3 gains require the native Qwen3-VL tokenizer.")
    batches = tokens["qwen3vl_32b"]
    if not isinstance(batches, (list, tuple)) or len(batches) != 1:
        raise ValueError("Experimental H3 gains require one native token sequence.")
    entries = batches[0]
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Unsupported native H3 token sequence.")
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[1] != 1.0:
            raise ValueError("Experimental H3 gains cannot be combined with token tuple weights.")
    return entries


def _visual_spans(tags, sequence_length):
    if not isinstance(tags, torch.Tensor) or tags.ndim != 1 or tags.numel() != sequence_length:
        raise ValueError("Native H3 token tags are missing or do not match the encoded sequence.")
    values = tags.detach().to(device="cpu").tolist()
    if any(value not in (0, 1) for value in values):
        raise ValueError("Unsupported H3 token tags; experimental gains were not applied.")
    spans = []
    start = None
    for index, value in enumerate(values + [1]):
        if value == 0 and start is None:
            start = index
        elif value != 0 and start is not None:
            spans.append((start, index))
            start = None
    return spans


class _WeightedClip:
    """Per-call proxy. The shared CLIP and its tokenizer remain untouched."""

    def __init__(self, clip, text_weight, image_weight, video_weight):
        self._clip = clip
        self._text_weight = text_weight
        self._visual_weights = {"image": image_weight, "video": video_weight}
        self._tokens = None
        self._entries = None
        self._visual_entries = None
        self._prompt_count = 0

    def __getattr__(self, name):
        return getattr(self._clip, name)

    def tokenize(self, text, *args, **kwargs):
        if not isinstance(text, str):
            raise ValueError("Experimental H3 gains require a string prompt.")
        tokens = self._clip.tokenize(text, *args, **kwargs)
        entries = _native_entries(tokens)
        visual_entries = []
        for index, entry in enumerate(entries):
            token = entry[0]
            if isinstance(token, Integral):
                continue
            if not isinstance(token, dict) or token.get("type") != "image":
                raise ValueError("Unsupported embedded token in native H3 conditioning.")
            if (index == 0 or index + 1 >= len(entries)
                    or entries[index - 1][0] != VISION_START
                    or entries[index + 1][0] != VISION_END):
                raise ValueError("H3 vision block delimiters changed; experimental gains were not applied.")
            kind = "video" if token.get("minimax_video_block", False) else "image"
            visual_entries.append((index, kind))

        # Native REF2VA appends the user's raw prompt after all reference labels.
        # Tokenize only that suffix, without references; this does not encode it.
        prompt_entries = _native_entries(self._clip.tokenize(text)) if text else []
        prompt_ids = [entry[0] for entry in prompt_entries]
        if any(not isinstance(token, Integral) for token in prompt_ids):
            raise ValueError("Embedded prompt tokens are not supported by experimental H3 gains.")
        if len(prompt_ids) > len(entries):
            raise ValueError("Cannot locate the user prompt in native H3 conditioning.")
        if prompt_ids:
            suffix_ids = [entry[0] for entry in entries[-len(prompt_ids):]]
            if any(not isinstance(token, Integral) for token in suffix_ids) or suffix_ids != prompt_ids:
                raise ValueError("Native H3 prompt suffix changed; experimental gains were not applied.")

        self._tokens = tokens
        self._entries = entries
        self._visual_entries = visual_entries
        self._prompt_count = len(prompt_ids)
        return tokens

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        if tokens is not self._tokens or self._entries is None:
            raise ValueError("H3 gain proxy must encode the exact sequence it tokenized.")
        conditioning = self._clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)
        weighted = []
        for hidden, metadata in conditioning:
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("Unsupported native H3 hidden-state shape.")
            spans = _visual_spans(metadata.get("minimax_token_tags"), hidden.shape[1])
            if len(spans) != len(self._visual_entries):
                raise ValueError("H3 visual token groups do not match their expanded spans.")

            delta = 0
            gains = []
            for (raw_index, kind), (start, stop) in zip(self._visual_entries, spans):
                # A raw image dict expands to N rows; its two delimiters remain.
                if stop - start < 3 or start != raw_index + delta - 1:
                    raise ValueError("H3 visual token offsets changed; refusing ambiguous gains.")
                delta += stop - start - 3
                gain = self._visual_weights[kind]
                if gain != 1.0:
                    # Keep both vision delimiters, labels and timestamps native.
                    gains.append((start + 1, stop - 1, gain))
            if len(self._entries) + delta != hidden.shape[1]:
                raise ValueError("H3 expanded token length changed; refusing ambiguous gains.")

            if self._prompt_count and self._text_weight != 1.0:
                start = hidden.shape[1] - self._prompt_count
                if spans and start < spans[-1][1]:
                    raise ValueError("User prompt overlaps an H3 vision block.")
                gains.append((start, hidden.shape[1], self._text_weight))

            if gains:
                hidden = hidden.detach().to(device="cpu").clone()
                for start, stop, gain in gains:
                    try:
                        hidden[:, start:stop].mul_(gain)
                    except (RuntimeError, OverflowError) as exc:
                        raise ValueError("H3 hidden-state gain exceeds the tensor's supported range.") from exc
                if not torch.isfinite(hidden).all().item():
                    raise ValueError("Experimental H3 gains produced nonfinite hidden states.")
            weighted.append([hidden, metadata])
        return weighted


def _scaled_latent(latent, gain):
    if not isinstance(latent, torch.Tensor) or not latent.is_floating_point():
        raise ValueError("Experimental H3 gains require floating-point reference latents.")
    result = latent.detach().to(device="cpu").clone()
    try:
        result.mul_(gain)
    except (RuntimeError, OverflowError) as exc:
        raise ValueError("H3 reference gain exceeds the latent tensor's supported range.") from exc
    if not torch.isfinite(result).all().item():
        raise ValueError("Experimental H3 gains produced a nonfinite reference latent.")
    return result


def _weighted_reference_latents(conditioning, image_weight, video_weight):
    cache = {}
    weighted = []
    for hidden, metadata in conditioning:
        refs = metadata.get("minimax_refs")
        if refs:
            key = id(refs)
            if key not in cache:
                updated = []
                for ref in refs:
                    kind = ref.get("kind")
                    gain = image_weight if kind == "image" else video_weight if kind in ("video", "video_audio") else 1.0
                    if gain != 1.0:
                        if gain == 0.0:
                            raise ValueError("A disabled H3 reference unexpectedly survived native encoding.")
                        ref = ref.copy()
                        if ref.get("latent") is None:
                            raise ValueError("Native H3 visual reference has no latent to weight.")
                        ref["latent"] = _scaled_latent(ref["latent"], gain)
                        # A video's paired soundtrack follows its video group;
                        # independent audio references remain unchanged.
                        if kind == "video_audio" and ref.get("audio_latent") is not None:
                            ref["audio_latent"] = _scaled_latent(ref["audio_latent"], gain)
                    updated.append(ref)
                cache[key] = updated
            metadata = {**metadata, "minimax_refs": cache[key]}
        weighted.append([hidden, metadata])
    return weighted


def encode_weighted_references(clip, text_weight=1.0, image_weight=1.0, video_weight=1.0,
                               *, reference_encoder=None, reference_kwargs=None,
                               **native_kwargs) -> io.NodeOutput:
    """Call native REF2VA with optional experimental single-pass gains.

    ``native_kwargs`` is the native ``MiniMaxH3ReferenceToVideo.execute`` input
    dictionary without ``clip``. A video weight of zero removes *all* entries
    in ``ref_videos`` and their paired soundtracks; the caller retains ownership
    of the separate source used for window planning. Standalone audio is not
    weighted. Disabled references are removed before Qwen and VAE encoding.

    ``reference_encoder`` optionally selects a caller-owned REF2VA assembly
    function; omitted, the native node's execute method is used unchanged.
    ``reference_kwargs`` carries extra arguments for that assembly function
    (caches, provenance) without routing them through the gain proxy.

    All-one inputs return the selected encoder's NodeOutput, with no proxy or
    tensor changes. Other binary values are native input removal only. Nonbinary
    values are experimental gains, including above-one extrapolation; quality
    and monotonic adherence are not guaranteed, particularly for distilled H3.
    """
    text_weight = _checked_weight(text_weight, "text_weight")
    image_weight = _checked_weight(image_weight, "image_weight")
    video_weight = _checked_weight(video_weight, "video_weight")
    if reference_encoder is None:
        reference_encoder = MiniMaxH3ReferenceToVideo.execute

    def assemble(clip_object, inputs):
        arguments = dict(inputs)
        if reference_kwargs:
            arguments.update(reference_kwargs)
        return reference_encoder(clip=clip_object, **arguments)

    if text_weight == image_weight == video_weight == 1.0:
        return assemble(clip, native_kwargs)

    inputs = dict(native_kwargs)
    if text_weight == 0.0:
        inputs["prompt"] = ""
    if image_weight == 0.0:
        inputs["ref_images"] = {}
    if video_weight == 0.0:
        inputs["ref_videos"] = {}
        inputs["ref_video_audios"] = {}

    if all(weight in (0.0, 1.0) for weight in (text_weight, image_weight, video_weight)):
        return assemble(clip, inputs)

    proxy = _WeightedClip(clip, text_weight, image_weight, video_weight)
    output = assemble(proxy, inputs)
    positive, latent = output.result
    positive = _weighted_reference_latents(positive, image_weight, video_weight)
    return io.NodeOutput(positive, latent, ui=output.ui, expand=output.expand,
                         block_execution=output.block_execution)
