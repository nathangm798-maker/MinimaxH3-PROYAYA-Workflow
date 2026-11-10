"""Bounded window reads from the installed official ComfyUI VIDEO types.

File videos (including temporal Video Slice/Trim outputs) are sampled on their
original PTS timeline. Only timestamp metadata is cached; decoding retains two
source frames plus the requested output window. The first metadata scan visits
the source video without converting or retaining RGB frames.

VideoFromComponents already owns a complete IMAGE tensor upstream. Its public
get_components() returns references, so this adapter selects only the requested
frames without encoding or copying that entire tensor. It cannot release the
upstream tensor. VideoFromList and unknown VIDEO implementations are rejected:
their public stream-source/default metadata methods can materialize the whole
video. The installed public API has no generic get_frame() method and exposes
no spatial crop coordinates, so file-backed spatial crops are also rejected.
"""

from contextlib import contextmanager
from fractions import Fraction
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import itertools
import logging
import math
from numbers import Integral
import sys
from weakref import WeakKeyDictionary

import av
import numpy as np
import torch
import torch.nn.functional as functional
import nodes as comfy_nodes
from comfy_api.latest import Input, InputImpl, Types, VideoInput


FPS = 24
# 原生 ImageScale 之外的第三种裁剪语义：按**绝对像素**取一个固定尺寸的居中窗口，
# 不缩放；窗口大于源画面时空出的部分填黑。见 _absolute_crop_pad。
ABSOLUTE_CROP = "absolute"
CROP_METHODS = tuple(comfy_nodes.ImageScale.crop_methods) + (ABSOLUTE_CROP,)


def video_fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Source video is not a file: {path}")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def source_frame_rate(path):
    """只读容器头拿源视频帧率（平均帧率优先，猜不准用 guessed_rate）。

    不做全片时间戳扫描 —— 与 read_audio_track 一样只付一次容器探测的成本。
    拿不到时返回 None，由调用方决定回退值。
    """
    try:
        with av.open(path, mode="r") as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return None
            rate = stream.average_rate or stream.guessed_rate
            return float(rate) if rate and rate > 0 else None
    except Exception:                                   # noqa: BLE001
        return None


def _timestamp(frame, stream):
    if frame.pts is None:
        raise ValueError("Source video has frames without timestamps; remux it before window sampling.")
    return Fraction(frame.pts) * (frame.time_base or stream.time_base)

_BUFFER_INFO = WeakKeyDictionary()


class WindowResizedVideo(VideoInput):
    """A VIDEO resize recipe; H3 reads it one window at a time at 24 FPS.

    Explicit get_components/save_to/get_stream_source calls materialize the
    whole clip. They are full-video API fallbacks, never the H3 reader path.
    Temporal trimming keeps the recipe lazy, including for tensor sources.
    """

    def __init__(self, video, width, height, upscale_method="area", crop="disabled",
                 *, start_offset=Fraction(0), duration=None, read_limit=0):
        _kind(video)
        if any(not isinstance(value, Integral) or isinstance(value, bool) or value < 32
               for value in (width, height)):
            raise ValueError("VIDEO resize width and height must be integers of at least 32 pixels.")
        if upscale_method not in comfy_nodes.ImageScale.upscale_methods or crop not in CROP_METHODS:
            raise ValueError("Unsupported native ImageScale interpolation or crop mode.")
        if not isinstance(read_limit, Integral) or isinstance(read_limit, bool) or read_limit < 0:
            raise ValueError("VIDEO read_limit must be a nonnegative integer (0 = no limit).")
        self.video = video
        self.width, self.height = int(width), int(height)
        self.upscale_method, self.crop = upscale_method, crop
        self.start_offset, self.duration = start_offset, duration
        # 只影响「把整片materialize出来」的路径（get_components / save_to）。
        # 元数据与逐窗读取不受它影响 —— 窗口计划的边界仍然按源片真实长度校验，
        # 这样设小了会得到明确的越界报错，而不是悄悄截断。
        self.read_limit = int(read_limit)

    def frame_limit(self, total_frames):
        """本包装最多向外交付多少帧。``0`` 表示不限制。"""
        return min(int(total_frames), self.read_limit) if self.read_limit else int(total_frames)

    def get_dimensions(self):
        return self.width, self.height

    def get_frame_rate(self):
        return Fraction(FPS)

    def get_frame_count(self):
        return video_info(self)["total_frames"]

    def get_duration(self):
        return video_info(self)["duration"]

    def get_bit_depth(self):
        # File windows currently use RGB24; do not advertise source precision
        # that the bounded RGB reader does not retain.
        return 8

    def get_color_space(self):
        return self.video.get_color_space()

    def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
        info = video_info(self)
        available = info["end_time"] - info["start_time"]
        start = _seconds(start_time or 0)
        if start < 0:
            start = max(available + start, Fraction(0))
        remaining = available - start
        requested = _seconds(duration or 0)
        if remaining <= 0 or requested < 0 or (strict_duration and requested > remaining):
            return None
        kept = min(requested, remaining) if requested else remaining
        return WindowResizedVideo(self.video, self.width, self.height, self.upscale_method, self.crop,
                                  start_offset=self.start_offset + start, duration=kept,
                                  read_limit=self.read_limit)

    def get_components(self):
        logging.warning("H3 VIDEO resize: full VIDEO extraction/save materializes the clip; use H3 window encoding for bounded reads.")
        info = video_info(self)
        count = self.frame_limit(info["total_frames"])
        # 以前这里先 self.video.get_components() 把**整片**解出来，只为了拿音频和
        # 元数据，然后把图丢掉再重新按目标尺寸读一遍。长源片（几千帧）光这一下就是
        # 几十秒白工。音频有只解音频轨的路子，元数据读容器头就有，都没必要付全片解码。
        source = self.video.get_stream_source()
        if isinstance(source, str):
            audio = read_audio_track(source)
            metadata = _file_metadata(source)
        else:
            components = self.video.get_components()
            audio, metadata = components.audio, components.metadata
            del components
        if audio is not None:
            rate = audio["sample_rate"]
            begin = round(self.start_offset * rate)
            end = begin + round(Fraction(count, FPS) * rate)
            audio = {**audio, "waveform": audio["waveform"][..., begin:end].clone()}
        images = read_video_window(self, 0, count, self.width, self.height)
        return Types.VideoComponents(images=images, frame_rate=Fraction(FPS), audio=audio, metadata=metadata)

    def save_to(self, path, format=Types.VideoContainer.AUTO, codec=Types.VideoCodec.AUTO,
                metadata=None, bit_depth=None, crf=None, color_space=None, preset=None):
        source_color = self.get_color_space()
        video = InputImpl.VideoFromComponents(self.get_components(), bit_depth=self.get_bit_depth(),
                                              color_space="sRGB" if source_color == "auto" else source_color)
        return video.save_to(path, format=format, codec=codec, metadata=metadata, bit_depth=bit_depth,
                             crf=crf, color_space=color_space, preset=preset)

    # The inherited stream-source fallback encodes the already-trimmed result;
    # its public active trim is therefore (0, 0), not the underlying offset.


def _seconds(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("VIDEO time values must be finite.")
    # Recover normal rational frame boundaries from the API's float seconds;
    # e.g. 86 / 24 must not become 86.000000000000004 output frames.
    return Fraction(str(number)).limit_denominator(1_000_000_000)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _rotation(frame):
    angle = float(frame.rotation or 0.0)
    quarter = round(angle / 90.0)
    if not math.isfinite(angle) or abs(angle - quarter * 90.0) > 0.001:
        raise ValueError("VIDEO rotation must be a multiple of 90 degrees; export this source first.")
    return int(quarter) % 4


@contextmanager
def _open_source(source):
    position = source.tell() if isinstance(source, BytesIO) else None
    if position is not None:
        source.seek(0)
    try:
        with av.open(source, mode="r") as container:
            yield container
    finally:
        if position is not None and not source.closed:
            source.seek(position)


def _scan_source(source):
    """Collect exact video-only duration without accumulating decoded frames."""
    with _open_source(source) as container:
        if not container.streams.video:
            raise ValueError("VIDEO source contains no video stream.")
        stream = container.streams.video[0]
        first = previous = last = None
        last_duration = Fraction(0)
        source_frames = 0
        geometry = None
        for frame in container.decode(stream):
            timestamp = _timestamp(frame, stream)
            if last is not None and timestamp < last:
                raise ValueError("VIDEO timestamps run backwards; remux this source first.")
            current_geometry = (frame.width, frame.height, _rotation(frame))
            if geometry is None:
                geometry = current_geometry
            elif geometry != current_geometry:
                raise ValueError("VIDEO dimensions or rotation change midstream; split/export this source first.")
            if first is None:
                first = timestamp
            previous, last = last, timestamp
            last_duration = Fraction(frame.duration or 0) * (frame.time_base or stream.time_base)
            source_frames += 1
        if not source_frames:
            raise ValueError("VIDEO source contains no decodable frames.")
        rate = stream.average_rate or stream.guessed_rate
        if last_duration <= 0:
            last_duration = (last - previous if previous is not None and last > previous
                             else Fraction(1) / Fraction(rate or FPS))
        raw_width, raw_height, rotation = geometry
        width, height = (raw_height, raw_width) if rotation % 2 else (raw_width, raw_height)
        return {
            "raw_width": raw_width, "raw_height": raw_height,
            "width": width, "height": height, "rotation": rotation,
            "start_time": first, "end_time": last + last_duration,
            "source_frames": source_frames, "source_fps": float(rate) if rate else None,
        }


@lru_cache(maxsize=8)
def _scan_path(fingerprint):
    info = _scan_source(fingerprint[0])
    if video_fingerprint(fingerprint[0]) != fingerprint:
        raise ValueError("VIDEO file changed while its timestamps were being read; run again.")
    return info


def _file_description(video):
    source = video.get_stream_source()
    if isinstance(source, str):
        fingerprint = video_fingerprint(source)
        source = fingerprint[0]
        raw = _scan_path(fingerprint)
        source_kind = "file"
    elif isinstance(source, BytesIO):
        # Official file-backed BytesIO objects are treated as immutable inputs.
        # Do not copy their compressed contents into a second buffer.
        with source.getbuffer() as view:
            size = view.nbytes
        cached = _BUFFER_INFO.get(source)
        if cached is None or cached[0] != size:
            cached = (size, _scan_source(source))
            _BUFFER_INFO[source] = cached
        raw = cached[1]
        fingerprint = None
        source_kind = "encoded_buffer"
    else:
        raise ValueError("Official file VIDEO must expose a local path or an existing BytesIO stream.")

    published_dimensions = tuple(video.get_dimensions())
    if published_dimensions != (raw["raw_width"], raw["raw_height"]):
        raise ValueError(
            "Spatially cropped file VIDEO cannot be read safely through this installed public API: "
            "crop coordinates are not exposed. Save the cropped video to a file and LoadVideo it, "
            "or use the uncropped VIDEO. Temporal Video Slice/Trim is supported."
        )

    start_value, duration_value = video.get_active_trim_window()
    trim_start, trim_duration = _seconds(start_value), _seconds(duration_value)
    if trim_start < 0 or trim_duration < 0:
        raise ValueError("VIDEO must expose a normalized, nonnegative active trim window.")
    # The native file implementation interprets active trim times as absolute
    # stream timestamps. An untrimmed stream starts at its first video PTS.
    start = max(raw["start_time"], trim_start)
    stop = min(raw["end_time"], trim_start + trim_duration) if trim_duration else raw["end_time"]
    duration = stop - start
    if duration <= 0:
        raise ValueError("The active VIDEO trim contains no video time.")
    info = {
        **raw, "fps": FPS, "total_frames": _ceil(duration * FPS),
        "duration": float(duration), "start_time": start, "end_time": stop,
        "source_kind": source_kind, "path": source if isinstance(source, str) else None,
        "fingerprint": fingerprint,
        "trim_start": float(trim_start), "trim_duration": float(trim_duration),
    }
    return source, info


def _components_description(video):
    # For exactly the official implementation this only exposes existing
    # tensor references. Calling this on an arbitrary VIDEO could decode all.
    components = video.get_components()
    images = components.images
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Component VIDEO must contain an NHWC RGB IMAGE tensor.")
    if not images.is_floating_point() or min(images.shape[:3]) <= 0:
        raise ValueError("Component VIDEO must contain nonempty floating-point IMAGE frames.")
    rate = Fraction(components.frame_rate)
    if rate <= 0:
        raise ValueError("Component VIDEO frame rate must be positive.")
    duration = Fraction(images.shape[0]) / rate
    info = {
        "fps": FPS, "total_frames": _ceil(duration * FPS),
        "source_frames": images.shape[0], "source_fps": float(rate),
        "width": images.shape[2], "height": images.shape[1],
        "duration": float(duration), "start_time": Fraction(0), "end_time": duration,
        "source_kind": "components", "path": None, "fingerprint": None,
        "trim_start": 0.0, "trim_duration": 0.0,
    }
    return images, rate, info


def _is_window_resized(video):
    # Exact-class check only: arbitrary VIDEO wrappers may materialize a clip.
    return type(video) is WindowResizedVideo


def _kind(video):
    # Restrict dispatch to known implementations: overridden public methods on
    # arbitrary subclasses may hide full-video decoding, export or transforms.
    if type(video) is InputImpl.VideoFromFile:
        return "file"
    if type(video) is InputImpl.VideoFromComponents:
        return "components"
    if _is_window_resized(video):
        return "resize"
    name = type(video).__name__
    raise ValueError(
        f"Unsupported bounded VIDEO input: {name}. Use official LoadVideo / temporal Video Slice "
        "or CreateVideo. VideoFromList and custom VIDEO objects have no guaranteed bounded "
        "frame-access interface in this ComfyUI version; export them to a file first."
    )


def _resized_description(video):
    source = video_info(video.video)
    start = source["start_time"] + video.start_offset
    stop = source["end_time"]
    if video.duration is not None:
        stop = min(stop, start + video.duration)
    duration = stop - start
    if duration <= 0:
        raise ValueError("The resized VIDEO trim contains no video time.")
    return {**source, "width": video.width, "height": video.height,
            "source_kind": "window_resize", "fps": FPS,
            "start_time": start, "end_time": stop,
            "duration": float(duration), "total_frames": _ceil(duration * FPS)}


def video_info(video):
    """Return small metadata; ``total_frames`` always uses the 24 fps timeline.

    Compare this count across main videos before allocating/encoding windows.
    Audio tails do not extend file-video lengths. Fraction start/end times are
    deliberately retained so independent window reads share exactly one origin.
    """
    kind = _kind(video)
    if kind == "resize":
        return _resized_description(video)
    if kind == "file":
        return dict(_file_description(video)[1])
    return dict(_components_description(video)[2])


def initial_video_target(video, width, height):
    """Read the first main source at target size without reference downscales.

    H3's lazy resize recipes are only reference presentation. Bypass their
    spatial transforms for the target latent, but retain the final effective
    time interval, including nested temporal trims. Already materialized
    component pixels are the best source available and cannot be recovered.
    """
    selected = video_info(video)
    source = video
    bypassed = 0
    while _is_window_resized(source):
        source = source.video
        bypassed += 1
    original = video_info(source)
    target = WindowResizedVideo(
        source, width, height, "lanczos", "center",
        start_offset=selected["start_time"] - original["start_time"],
        duration=selected["end_time"] - selected["start_time"],
    )
    return target, {
        "source_width": original["width"], "source_height": original["height"],
        "width": width, "height": height, "bypassed_resize_recipes": bypassed,
        "source_kind": original["source_kind"],
    }


def _bounds(start_frame, count, width, height, total_frames):
    if any(not isinstance(value, Integral) or isinstance(value, bool)
           for value in (start_frame, count, width, height)):
        raise ValueError("VIDEO window start/count and dimensions must be integers.")
    if start_frame < 0 or count <= 0 or width <= 0 or height <= 0:
        raise ValueError("VIDEO start must be nonnegative; count and dimensions must be positive.")
    if start_frame + count > total_frames:
        raise ValueError(f"VIDEO window [{start_frame}, {start_frame + count}) exceeds {total_frames} frames at 24 fps.")


def _apply_resize_ops(frame, resize_ops):
    """Apply native recipes to one NHWC frame, never a full source window."""
    for method, width, height, crop in resize_ops:
        if crop == ABSOLUTE_CROP:
            frame = _absolute_crop_pad(frame, width, height)
        else:
            frame = comfy_nodes.ImageScale().upscale(frame, method, width, height, crop)[0]
    return frame


def _absolute_crop_pad(frame, width, height):
    """Cut a fixed ``width x height`` pixel window out of ``frame``, 1:1, center aligned.

    Unlike ``crop="center"`` this never scales: the output window is ``width`` by
    ``height`` *source* pixels. When the window is larger than the frame (or
    hangs over an edge) the missing area is filled with black, so a 1080x1920
    source asked for 720x1280 returns the middle 720x1280 untouched.
    """
    src_h, src_w = int(frame.shape[1]), int(frame.shape[2])
    channels = int(frame.shape[3])
    offset_x = (src_w - width) // 2
    offset_y = (src_h - height) // 2
    output = frame.new_zeros((1, height, width, channels))

    # Intersection between the requested window and the source frame.
    src_x = max(offset_x, 0)
    src_y = max(offset_y, 0)
    dst_x = max(-offset_x, 0)
    dst_y = max(-offset_y, 0)
    keep_w = min(src_w - src_x, width - dst_x)
    keep_h = min(src_h - src_y, height - dst_y)
    if keep_w > 0 and keep_h > 0:
        output[:, dst_y:dst_y + keep_h, dst_x:dst_x + keep_w, :] = \
            frame[:, src_y:src_y + keep_h, src_x:src_x + keep_w, :]
    return output


def _file_metadata(path):
    """只读容器头拿元数据，不解码视频轨。

    与 ``VideoFromFile.get_components().metadata`` 同源（都是 ``container.metadata``），
    但不需要把整片 RGB 解出来。读不到时返回 ``None``，与「没有元数据」等价。
    """
    try:
        with av.open(path, mode="r") as container:
            return container.metadata or None
    except Exception:                                   # noqa: BLE001 - 头坏了不影响出帧
        return None


def read_audio_track(path):
    """只解音频轨，返回 ``AudioInput``；源文件没有音频轨时返回 ``None``。

    与官方 ``VideoFromFile.get_components_internal`` 的音频分支同构：``fltp``
    重采样、逐包解码、按 sample 轴 concat 成 ``(channels, samples)`` 再升到
    ``(1, channels, samples)``，因此波形与 ``GetVideoComponents`` 抽出来的是同一份。

    **不动视频轨**：整个调用只 demux 音频流，1080p 长片的耗时与内存都跟画幅无关。
    没有音频流时在打开容器后立刻返回，成本只有一次容器探测。
    """
    with av.open(path, mode="r") as container:
        stream = _decodable_audio_stream(container)
        if stream is None:
            return None
        sample_rate = int(stream.codec_context.sample_rate or 0)
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        chunks = []
        for packet in container.demux(stream):
            try:
                decoded = packet.decode()
            except av.error.FFmpegError:
                continue
            for frame in itertools.chain.from_iterable(map(resampler.resample, decoded)):
                if not sample_rate:
                    sample_rate = int(frame.sample_rate)
                chunks.append(frame.to_ndarray())
        # 冲出重采样器内部残留的尾包（容器末尾不足一帧的部分）
        for frame in resampler.resample(None):
            if not sample_rate:
                sample_rate = int(frame.sample_rate)
            chunks.append(frame.to_ndarray())

    chunks = [chunk for chunk in chunks if chunk.shape[-1] > 0]
    if not chunks:
        return None
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)).unsqueeze(0)
    return Input.Audio({"waveform": waveform, "sample_rate": sample_rate or 1})


def _decodable_audio_stream(container):
    """取最后一条能解码的音频流；FFmpeg 无解码器的流会没有 codec_context。"""
    stream = next(
        (s for s in reversed(container.streams.audio) if s.codec_context is not None),
        None,
    )
    if stream is None and len(container.streams.audio):
        logging.warning("PROYAYA: 视频里的音频流没有可用解码器，按无音频处理。")
    return stream


def _read_file(source, info, offset, count, width, height, resize_ops=()):
    origin = info["start_time"]
    target = origin + offset
    result = torch.empty((count, height, width, 3), dtype=torch.float32, device="cpu")
    with _open_source(source) as container:
        stream = container.streams.video[0]
        container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
        frames = container.decode(stream)
        current = next(frames, None)
        if current is not None and _timestamp(current, stream) > target:
            container.seek(0, stream=stream, backward=True, any_frame=False)
            frames = container.decode(stream)
            current = next(frames, None)
        if current is None or _timestamp(current, stream) > target:
            raise ValueError(f"Cannot locate VIDEO time {float(offset):.6f}s on its original PTS timeline.")
        following = next(frames, None)
        rgb_frame = rgb = None
        for index in range(count):
            target = origin + offset + Fraction(index, FPS)
            while following is not None and _timestamp(following, stream) <= target:
                current = following
                following = next(frames, None)
            if target >= info["end_time"]:
                raise ValueError("VIDEO ended before the requested output window.")
            if current is not rgb_frame:
                rotation = _rotation(current)
                if resize_ops:
                    # Keep source pixels intact for native crop/interpolation,
                    # but materialize only this source frame at full resolution.
                    rgb = current.reformat(format="rgb24").to_ndarray()
                else:
                    scaled_width, scaled_height = (height, width) if rotation % 2 else (width, height)
                    rgb = current.reformat(width=scaled_width, height=scaled_height, format="rgb24").to_ndarray()
                if rotation:
                    rgb = np.rot90(rgb, k=rotation, axes=(0, 1)).copy()
                rgb = torch.from_numpy(rgb)
                if resize_ops:
                    rgb = _apply_resize_ops(rgb.to(dtype=torch.float32).div_(255.0).unsqueeze(0), resize_ops)[0]
                rgb_frame = current
            result[index].copy_(rgb)
    if info["fingerprint"] is not None and video_fingerprint(source) != info["fingerprint"]:
        raise ValueError("VIDEO file changed while a window was being decoded; run again.")
    return result if resize_ops else result.div_(255.0)


def _read_components(images, rate, offset, count, width, height, resize_ops=()):
    result = torch.empty((count, height, width, 3), dtype=torch.float32, device="cpu")
    previous_index = None
    resized = None
    for index in range(count):
        position = (offset + Fraction(index, FPS)) * rate
        source_index = position.numerator // position.denominator
        if source_index != previous_index:
            # Slice before moving/casting: never copy the complete upstream IMAGE.
            frame = images[source_index:source_index + 1, :, :, :3].detach().to(device="cpu", dtype=torch.float32)
            if resize_ops:
                resized = _apply_resize_ops(frame, resize_ops)[0]
            elif frame.shape[1:3] != (height, width):
                resized = functional.interpolate(frame.permute(0, 3, 1, 2), size=(height, width),
                                                  mode="bilinear", align_corners=False).permute(0, 2, 3, 1)[0]
            else:
                resized = frame[0]
            previous_index = source_index
        result[index].copy_(resized)
    return result


def _read_at_time(video, offset, count, width, height, resize_ops=()):
    kind = _kind(video)
    if kind == "resize":
        info = _resized_description(video)
        if offset < 0 or offset + Fraction(count - 1, FPS) >= info["end_time"] - info["start_time"]:
            raise ValueError("Requested window exceeds the resized VIDEO trim.")
        # Unwrap recipes into their original inner-to-outer order. The base
        # decoder applies them one source frame at a time into the final-size
        # result, avoiding a source-resolution tensor for the whole window.
        current_ops = ((video.upscale_method, video.width, video.height, video.crop),)
        if not resize_ops and (width, height) != (video.width, video.height):
            current_ops += ((video.upscale_method, width, height, "disabled"),)
        return _read_at_time(video.video, video.start_offset + offset, count,
                             width, height, current_ops + resize_ops)
    if kind == "file":
        source, info = _file_description(video)
        if offset < 0 or offset + Fraction(count - 1, FPS) >= info["end_time"] - info["start_time"]:
            raise ValueError("Requested window exceeds the source VIDEO timeline.")
        return _read_file(source, info, offset, count, width, height, resize_ops)
    images, rate, info = _components_description(video)
    if offset < 0 or offset + Fraction(count - 1, FPS) >= info["end_time"]:
        raise ValueError("Requested window exceeds the component VIDEO timeline.")
    return _read_components(images, rate, offset, count, width, height, resize_ops)


def read_video_window(video, start_frame, count, width, height):
    """Read exactly ``count`` CPU float RGB frames or reject invalid bounds.

    The selection rule is previous-frame hold at origin + k/24 (VFR files) or
    floor(k * source_fps / 24) (component tensors). Each window uses the same
    original origin, so splitting cannot accumulate timing drift. No tail pad
    is implicit: the encoder owns padding to the legal H3 window length.
    Auxiliary videos can use start_frame=0 on every window; main videos use
    the shared window start. Short references must be capped by the caller.
    """
    info = video_info(video)
    _bounds(start_frame, count, width, height, info["total_frames"])
    return _read_at_time(video, Fraction(start_frame, FPS), count, width, height)
