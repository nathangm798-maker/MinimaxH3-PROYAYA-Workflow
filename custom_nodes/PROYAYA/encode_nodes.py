"""PROYYMiniMaxH3ReferenceToVideo: multi-window REF2VA encoding on one 24 fps timeline."""

import math

import torch
import comfy.model_management as model_management
import comfy.utils
import nodes as comfy_nodes
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker
from contextlib import ExitStack
from typing import NamedTuple
from server import PromptServer

from .common import append_window, audio_slice, autogrow, cpu_tensors, group, ordered
from .conditioning_weights import encode_weighted_references
from .native_video import initial_video_target, read_video_window, video_info
from .planner import Plan, PlannedWindow, make_plan, reachable_overlaps, solve_overlaps
from .ref_cache import MISSING, Identity, model_identity, tensor_fingerprint
from . import ref_cache as ref_cache_module
from .sized_references import encode_sized_references
from .windows import shapes_for_windows


def _main_sources(main_ref_videos, *, lazy=False):
    sources = group(main_ref_videos)
    if not sources:
        raise ValueError("请至少连接一路主视频：官方 Load Video → 主视频 <video 1>。")
    ordered_sources = [(key, sources[key][0] if lazy else sources[key]) for key in ordered(sources)]
    infos = [video_info(video) for _, video in ordered_sources]
    lengths = [info["total_frames"] for info in infos]
    if len(set(lengths)) != 1:
        details = "，".join(f"<video {i + 1}>={length}帧" for i, length in enumerate(lengths))
        raise ValueError(f"所有主视频统一到24 FPS后长度必须一致：{details}。请在上游裁到相同长度。")
    return ordered_sources, infos


def _read_padded(video, info, start, count, width, height):
    """Read ``count`` frames; only a legal window's tail padding repeats the last frame."""
    available = max(0, min(count, info["total_frames"] - start))
    frames = read_video_window(video, min(start, info["total_frames"] - 1),
                               max(1, available), width, height)
    if len(frames) < count:
        frames = torch.cat((frames, frames[-1:].expand(count - len(frames), -1, -1, -1)), dim=0)
    return frames


def _paired_indices(audios, video_count):
    for key in ordered(audios):
        index = int(key.rsplit("_", 1)[-1])
        if index >= video_count:
            raise ValueError(f"配套音轨 {key} 对应 <video {index + 1}>，但只连接了 {video_count} 路视频。")
        yield index, key


def _validate_weights(*gains):
    for gain in gains:
        try:
            valid = math.isfinite(gain) and gain >= 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError("条件权重必须为有限的非负数。")


def _validate_length(length):
    if isinstance(length, bool) or not isinstance(length, int) or length < 5:
        raise ValueError("length 必须是不小于5的整数帧数。")
    return length


def _validate_materials(images, auxiliary, paired, standalone, main_count):
    """Inspect already-resolved references before requesting any encoder."""
    for key, image in images.items():
        if (not isinstance(image, torch.Tensor) or image.ndim != 4
                or min(image.shape[:3]) <= 0 or image.shape[-1] < 3
                or not image.is_floating_point()):
            raise ValueError(f"参考图片 {key} 必须为非空的浮点 RGB IMAGE。")
    aux_infos = [video_info(video) for _, video in auxiliary]
    for index, info in enumerate(aux_infos, main_count + 1):
        if info["total_frames"] < 5:
            raise ValueError(f"辅助视频 <Video {index}> 至少需要5帧（24 FPS）。")
    for key, audio in list(paired.items()) + list(standalone.items()):
        if not isinstance(audio, dict):
            raise ValueError(f"参考音频 {key} 不是有效的 AUDIO。")
        rate, wave = audio.get("sample_rate"), audio.get("waveform")
        try:
            valid_rate = math.isfinite(rate) and rate > 0 and int(rate) == rate
        except (TypeError, ValueError, OverflowError):
            valid_rate = False
        if (not valid_rate or not isinstance(wave, torch.Tensor) or wave.ndim != 3
                or min(wave.shape) <= 0 or not wave.is_floating_point()):
            raise ValueError(f"参考音频 {key} 需要正整数采样率和非空的浮点 [batch, channels, samples] 波形。")
    return aux_infos


class PlanOutcome(NamedTuple):
    """Plan plus everything the report needs to explain a wiring problem."""

    plan: object
    selected: tuple
    notice: tuple = ()
    solution: object = None


def _plan(source, length, auto_segment, window_frames, manual_lengths, overlap,
          keep_final, auto_prompt, prompts):
    """Plan on min(length, actual main-video frames); never invent frames to reach length.

    Wiring problems that a user can fix by editing the graph (no prompt
    connected, prompt count that cannot line up with the generated windows, an
    overlap request no layout satisfies) are returned in ``notice`` instead of
    raised: the caller still emits the plan report, so the message appears in
    the 分段信息 preview rather than only in the execution log.

    ``auto_prompt`` defaults to true and means "one prompt for the whole clip":
    only the first connected prompt is used and every window shares it. Turning
    it off switches to manual per-window prompts, and each window then consumes
    one prompt in connection order.

    Overlap auto-solve is no longer a separate switch: it is implied by
    ``auto_segment``. Manual per-segment lengths cannot auto-solve, so the
    solve only runs in uniform mode.
    """
    target = _validate_length(length)
    total = min(source["total_frames"], target)
    keys = ordered(prompts)
    texts = {key: _prompt_text(prompts[key]) for key in keys}
    solution = None
    notice = ()

    def build(overlaps=None):
        return make_plan(source["total_frames"], "uniform" if auto_segment else "manual",
                         window_frames, manual_lengths, overlap,
                         keep_final, target_frames=target, overlaps=overlaps)

    if len(keys) > 1 and not auto_prompt and auto_segment:
        # Only boundaries can be solved: with a single prompt there is nothing
        # to pick, and solve_overlaps would hand back a one-window layout that
        # cannot cover a longer timeline (make_plan then legitimately rejects it).
        same_flags = tuple(texts[keys[i]] == texts[keys[i + 1]] for i in range(len(keys) - 1))
        solution = solve_overlaps(total, window_frames, len(keys), same_flags,
                                  tail_keep_threshold=keep_final)
        if solution is not None:
            plan = build(solution.gaps)
        else:
            plan = build()
            notice = _unsolvable_overlap_notice(total, window_frames, len(keys),
                                                keep_final)
    else:
        plan = build()

    if not keys:
        notice = ("没有连接任何提示词：请在 prompts 上至少连接一个文本节点（空文本也需要一个节点）。",)
    elif not auto_prompt and len(keys) != len(plan.windows) and not notice:
        # A failed overlap solve already explained itself in more detail.
        notice = _prompt_count_notice(plan, len(keys), auto_segment, window_frames,
                                      overlap, total)
    return PlanOutcome(plan, tuple(keys if not auto_prompt else keys[:1]), notice, solution)


def _prompt_text(value):
    """Unwrap an autogrow prompt entry during lazy discovery."""
    return value[0] if isinstance(value, tuple) else value


def _prompt_count_notice(plan, connected, auto_segment, window_frames, overlap, total):
    lines = [
        f"提示词数量不匹配：当前参数生成 {plan.window_count} 段，连接了 {connected} 个提示词。"
        "开启逐段提示词时，每个窗口必须按顺序对应一个提示词，数量必须相等。",
    ]
    if auto_segment and connected > 1:
        stride = 51 * max(1, (window_frames - overlap) // 51)
        needed = (connected - 1) * stride + window_frames
        lines.append(
            f"按当前重叠（{window_frames} 帧窗、最低重叠 {overlap} → 步长 {stride} 帧），"
            f"{connected} 段需要 {needed} 帧；当前可用 {total} 帧"
            + ("（主视频不够长）。" if needed > total else "。"))
    suggestions = ["把 length 调成能生成 " + str(connected) + " 段的帧数，或改 window_frames / overlap 改变段数"]
    suggestions.append("增删提示词，使其数量等于 " + str(plan.window_count))
    suggestions.append("打开 auto_prompt，整段只用第一个提示词")
    lines.append("可选做法：" + "；".join(suggestions) + "。")
    lines.append("本次不会继续编码；下方表格是按当前参数算出的分段，可用来核对。")
    return tuple(lines)


def _unsolvable_overlap_notice(total, window_frames, connected, keep_final):
    feasible = _feasible_window_counts(total, window_frames, keep_final)
    lines = [
        f"无法为 {connected} 个提示词求解重叠：{total} 帧、每窗 {window_frames} 帧时，"
        f"不存在刚好 {connected} 段且每段都落在原生网格上的起点布局。",
        "窗口起点必须按 51 帧对齐，所以可用的实际重叠只有 "
        f"{list(reachable_overlaps(window_frames))} 帧，能排出的段数是离散的。",
        "当前帧数下可行的段数：" + ("、".join(str(count) for count in feasible) if feasible else "无")
        + "。",
        "可选做法：增删提示词到上述数量；或改 window_frames / length / keep_final。",
        "本次不会继续编码；下方是按 overlap 算出的分段，可用来核对。",
    ]
    return tuple(lines)


def _feasible_window_counts(total, window_frames, keep_final, limit=8):
    counts = [1] if total >= 1 else []
    for count in range(2, limit + 1):
        if solve_overlaps(total, window_frames, count, [False] * (count - 1),
                          tail_keep_threshold=keep_final) is not None:
            counts.append(count)
    return counts


def plan_report(plan, path, prompt_keys, auto_prompt, notice=(), solution=None):
    lines = []
    if notice:
        lines += ["!" * 60]
        lines += [f"⚠⚠⚠ {line}" for line in notice]
        lines += ["!" * 60]
    if plan.shortfall_frames:
        lines += [
            "!" * 60,
            f"⚠⚠⚠ 警告：目标帧数 {plan.target_frames} 超过主视频实际帧数 {plan.source_frames}（缺 {plan.shortfall_frames} 帧）",
            f"⚠⚠⚠ 已按实际帧数规划：{len(plan.windows)} 段，输出 {plan.output_frames} 帧；"
            f"丢弃尾帧 {plan.dropped_frames} 帧。不会复制末帧补段。",
            "⚠⚠⚠ 如需更长输出，请更换更长的主视频，或把 length 改为不超过实际帧数。",
            "!" * 60,
        ]
    lines += [f"源视频：{path}",
              f"目标帧数 {plan.target_frames}；主视频实际 {plan.source_frames} 帧；24 FPS；{len(plan.windows)} 窗",
              f"有效输出：{plan.output_frames} 帧 / {plan.output_frames / 24:.3f} 秒；"
              f"丢弃尾帧：{plan.dropped_frames} 帧（阈值 {plan.tail_keep_threshold}）；"
              f"补帧：{plan.pad_frames} 帧（解码后裁掉）"]
    if solution is not None:
        lines.append(f"重叠自动求解：{solution.summary()}")
    lines.append("窗 | 帧区间 [起,止) | 采样帧 | 实际重叠 | 有效帧 | 补帧 | 提示词接口")
    for i, window in enumerate(plan.windows):
        if not auto_prompt:
            key = prompt_keys[i] if i < len(prompt_keys) else "（缺提示词）"
        else:
            key = prompt_keys[0] if prompt_keys else "（未连接）"
        lines.append(f"{i + 1} | [{window.start}, {window.start + window.real_frames}) | "
                     f"{window.frames} | {window.overlap} | {window.real_frames} | {window.pad_frames} | {key}")
    return "\n".join(lines)


def _video_source(video, info):
    """Cache identity for one video input.

    The reader's own fingerprint (path + mtime + size) is preferred: it survives
    an upstream node rebuilding the VIDEO object with identical content, which
    plain object identity would treat as a brand new source. Videos assembled
    from components have no fingerprint, so those fall back to identity.
    """
    fingerprint = info.get("fingerprint") if isinstance(info, dict) else None
    if fingerprint:
        return ("video", fingerprint, info.get("trim_start"), info.get("trim_duration"),
                info.get("total_frames"))
    return ("video", Identity(video))


def _with_cache_line(report, cache):
    """Append the cache accounting, if this run touched a cache at all."""
    if cache is None:
        return report
    line = cache.summary()
    return report + "\n" + line if line else report


def _second_pass_plan(entries, source_latent, target_length, keep_final):
    """用一采的 windows 复刻时间轴，只把初始 latent 换成二采的内容。

    起点、长度、补帧、重叠全部照抄一采，所以两条 pass 的接缝逐帧重合；这里
    不重新 solve 重叠，也不看 auto_segment / window_frames / overlap 三个控件。

    返回 ``(plan, streams)``；``streams`` 是已搬到 CPU FP32 的 [video, audio]。
    """
    if not entries:
        raise ValueError("second_pass_windows 是空的：请接一采的 windows 输出。")
    if not isinstance(source_latent, dict) or source_latent.get("samples") is None:
        raise ValueError("二采需要 source_latent：接一采 joint_latent，或它放大后的版本。")
    samples = source_latent["samples"]
    if not getattr(samples, "is_nested", False) or len(samples.unbind()) != 2:
        raise ValueError("source_latent 必须是 H3 的 joint AV latent（video + audio 两路）。")
    video, audio = samples.unbind()
    if video.ndim != 5 or video.shape[:2] != (1, 24):
        raise ValueError("source_latent 的视频分支必须是 [1,24,T,H,W]。")
    if audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
        raise ValueError("source_latent 的音频分支必须是 [1,32,2,T]。")

    windows = [entry["window"] for entry in entries]
    shapes_for_windows(windows)
    last = windows[-1]
    covered = last.video_start + last.video_length
    if int(video.shape[2]) != covered:
        raise ValueError(
            "source_latent 与 second_pass_windows 的时间轴对不上：latent 视频 %d 帧，"
            "windows 覆盖 %d 帧。请确认两者来自同一次一采，中途没有裁剪或拼接。"
            % (int(video.shape[2]), covered))

    total = min(_validate_length(target_length), last.start + last.frames)
    if total <= last.start:
        raise ValueError(
            "length=%d 比最后一段的起点 %d 还小，二采会少掉一整段。请把一采的 length 原样接过来。"
            % (total, last.start))

    planned = []
    previous_end = 0
    for index, window in enumerate(windows):
        if index == len(windows) - 1:
            real = max(1, min(window.frames, total - window.start))
            pad = window.frames - real
        else:
            real, pad = window.frames, 0
        overlap = 0 if index == 0 else max(0, previous_end - window.start)
        planned.append(PlannedWindow(window.start, window.frames, real, pad, overlap, window.frames))
        previous_end = window.start + window.frames

    plan = Plan(total_frames=total, mode="reuse", windows=tuple(planned),
                output_frames=total, sampled_frames=previous_end, dropped_frames=0,
                pad_frames=sum(window.pad_frames for window in planned),
                minimum_overlap=0, tail_keep_threshold=keep_final, tail_action="reused",
                source_frames=total, target_frames=total)
    streams = [stream.detach().to(device="cpu", dtype=torch.float32) for stream in (video, audio)]
    return plan, streams


class PROYYMiniMaxH3ReferenceToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        gain_tip = "实验性条件增益，非原生独立权重。1沿用原生；0移除此组条件；其它值不保证影响程度成正比。"
        return io.Schema(
            node_id="PROYYMiniMaxH3ReferenceToVideo", display_name="۞▷Reference to Window◁۞",
            category="PROYAYA/H3",
            description="MiniMax H3 Reference to Window 的分窗版本：按实际主视频帧数分段并逐窗编码；video_encode 可选。",
            inputs=[
                io.Clip.Input("clip", lazy=True),
                io.Vae.Input("vae", lazy=True),
                io.Boolean.Input("auto_segment", default=True, label_on="true", label_off="false",
                                 tooltip="true自动分段并禁用手动长度，同时开启重叠自动求解；false使用manual_lengths。"),
                io.Boolean.Input("auto_prompt", default=True,
                                 tooltip="true（默认）整段只用第一个连接的提示词，所有窗口共用。"
                                         "false 时手动逐段：提示词数必须等于最终段数，按连接顺序一一对应。"),
                io.Int.Input("window_frames", default=124, min=90, step=17,
                             tooltip="90、107、124、141…；满足5+17k，无人为最大时长限制。"),
                io.String.Input("manual_lengths", default="124,124,124", multiline=False,
                                tooltip="手动逐段长度，逗号或空格分隔；每项至少90帧且满足5+17k。auto_segment=true时禁用。"),
                io.Int.Input("overlap", default=22, min=1,
                             tooltip="最低重叠；起点按51帧对齐，实际重叠可能更大。"
                                     "auto_segment=true 且 auto_prompt=false 时改由提示词自动求解，此项不再生效。"),
                io.Int.Input("keep_final", default=40, min=0,
                             tooltip="尾部未被完整窗口覆盖的新帧数：大于等于此值则补成一窗（补帧解码后裁掉），小于此值则丢弃。"),
                io.Int.Input("width", default=768, min=32, step=32),
                io.Int.Input("height", default=1344, min=32, step=32),
                io.Int.Input("length", default=328, min=5,
                             tooltip="目标输出帧数上限。实际规划取 min(length, 主视频实际帧数)；主视频不足时不会复制末帧补段，plan_report 会给出醒目警告。"),
                io.Float.Input("text_weight", default=1.0, min=0.0, step=0.05, tooltip=gain_tip),
                io.Float.Input("image_weight", default=1.0, min=0.0, step=0.05, tooltip=gain_tip),
                io.Float.Input("video_weight", default=1.0, min=0.0, step=0.05,
                               tooltip=gain_tip + " 对主视频、辅助视频及其配套音轨一起生效。"),
                io.Boolean.Input("video_encode", default=False,
                                 tooltip="启用后从第一路主视频原始画面逐窗编码初始视频latent，独立于视频参考权重。"
                                         "按输出尺寸等比填满中心裁切，并绕过参考用的空间Resize；保留时间裁剪。"
                                         "音频初始latent仍为空。关闭保持空latent路径。"),
                io.Boolean.Input("unload", default=True,
                                 tooltip="所有窗口编码完成后卸载CLIP；参考编码使用的VAE也会卸载。"),
                io.Boolean.Input("preview_only", default=False,
                                 tooltip="只检查分段并输出报告，不请求本节点CLIP/VAE，也不采样。警告不会截断生成，除非启用此项。"),
                io.Vae.Input("audio_vae", optional=True, lazy=True),
                autogrow("ref_images", io.Image.Input("ref_image", lazy=True,
                    tooltip="参考图片；提示词使用原生标签 <Picture 1>、<Picture 2>…"), "ref_image_"),
                autogrow("main_ref_videos", io.Video.Input("main_ref_video",
                    tooltip="主视频：连接官方Load Video的VIDEO。多路长度必须相同，按同一时间轴逐窗读取。"), "main_ref_video_"),
                autogrow("ref_videos", io.Video.Input("ref_video", lazy=True,
                    tooltip="辅助视频：VIDEO类型；每窗从头使用，超长部分截断，与主视频共用<Video n>编号。"), "ref_video_"),
                autogrow("main_ref_video_audios", io.Audio.Input("main_ref_video_audio", lazy=True,
                    tooltip="接口后缀对应主视频+辅助视频的总编号。AUDIO必须在上游与对应VIDEO裁到同一时间段；不会自动叠加视频裁剪起点。主音轨同步切片并在视频有效时长后补静音，辅助音轨从头裁剪。"), "main_ref_video_audio_"),
                autogrow("ref_audios", io.Audio.Input("ref_audio", lazy=True,
                    tooltip="独立参考音频，每窗完整复用；排在所有已接入的视频配套音轨之后。需audio_vae。"), "ref_audio_"),
                autogrow("prompts", io.String.Input("prompt", lazy=True), "prompt_"),
                io.Custom("H3_JOINT_WINDOWS").Input("second_pass_windows", optional=True,
                    tooltip="二采：接一采的 windows。接入后不重新规划，起点/长度/补帧/重叠全部照抄，"
                            "保证两条 pass 的接缝逐帧重合。此时 auto_segment / window_frames / "
                            "manual_lengths / overlap 四个控件不再生效，length 请接一采的 length。"),
                io.Latent.Input("source_latent", optional=True,
                    tooltip="二采的初始 latent：接一采 joint_latent（或它放大后的版本）。"
                            "每个窗口从它裁出自己那一段，视频与音频都作为初始值，跳过 VAE 往返。"
                            "接入后本节点的 width/height 会被忽略，改由 latent 尺寸 × 16 反推。"
                            "windows 会带 lock_audio，采样器据此保留音频、只重绘画面。"),
                io.Boolean.Input("ref_cache", default=True,
                                 tooltip="复用参考编码结果：改时长时已有窗口直接复用，只补算新增窗口；"
                                         "改提示词时参考图/参考视频/参考音频的 VAE 编码不重算。"
                                         "关掉后每次执行都从头编码。"),
                io.Int.Input("ref_cache_mb", default=1024, min=0, step=64,
                             tooltip="缓存内存上限（MB），0 关闭。超出后淘汰最久未使用的条目；"
                                     "只影响速度，不影响结果。"),
            ],
            outputs=[io.Custom("H3_JOINT_WINDOWS").Output(display_name="windows"),
                     io.Int.Output(display_name="length"),
                     io.String.Output(display_name="plan_report")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def check_lazy_status(cls, auto_segment, window_frames, manual_lengths, overlap,
                          keep_final, auto_prompt, preview_only,
                          text_weight, image_weight, video_weight, length,
                          main_ref_videos=None, prompts=None,
                          clip=None, vae=None, **kwargs):
        _validate_weights(text_weight, image_weight, video_weight)
        main, infos = _main_sources(main_ref_videos, lazy=True)
        prompts = group(prompts)
        if kwargs.get("second_pass_windows") is not None:
            # 二采：分段完全由一采的 windows 决定，规划控件不参与，这里不再试算。
            keys = ordered(prompts)
            selected = tuple(keys[:1] if auto_prompt else keys)
            notice = ()
        else:
            outcome = _plan(infos[0], length, auto_segment, window_frames, manual_lengths, overlap,
                            keep_final, auto_prompt, prompts)
            selected, notice = outcome.selected, outcome.notice
        if notice:
            # Nothing else is needed: execution emits the report that explains it.
            return []
        paired = group(kwargs.get("main_ref_video_audios")) if video_weight else {}
        list(_paired_indices(paired, len(main) + len(group(kwargs.get("ref_videos")))))
        if preview_only:
            return []
        requested = [prompts[key][1] for key in selected if prompts[key][0] is None]
        if requested:
            return requested
        requested = []
        groups = ["ref_audios"]
        if image_weight:
            groups.append("ref_images")
        if video_weight:
            groups.extend(("ref_videos", "main_ref_video_audios"))
        if paired or group(kwargs.get("ref_audios")):
            if "audio_vae" not in kwargs:
                raise ValueError("接入视频配套音轨或独立参考音频时，请连接audio_vae。")
        for name in groups:
            requested.extend(value[1] for value in group(kwargs.get(name)).values() if value[0] is None)
        if requested:
            return requested
        images = {key: value[0] for key, value in group(kwargs.get("ref_images")).items()} if image_weight else {}
        auxiliary = group(kwargs.get("ref_videos")) if video_weight else {}
        standalone = group(kwargs.get("ref_audios"))
        _validate_materials(
            images, [(key, auxiliary[key][0]) for key in ordered(auxiliary)],
            {key: value[0] for key, value in paired.items()},
            {key: value[0] for key, value in standalone.items()}, len(main))
        requested = [name for name, value in (("clip", clip), ("vae", vae)) if value is None]
        if (paired or standalone) and kwargs["audio_vae"] is None:
            requested.append("audio_vae")
        return requested

    @classmethod
    def execute(cls, auto_segment, auto_prompt, window_frames, manual_lengths,
                overlap, keep_final, width, height, length, text_weight,
                image_weight, video_weight, unload, preview_only,
                clip=None, vae=None, audio_vae=None, ref_images=None, main_ref_videos=None,
                ref_videos=None, main_ref_video_audios=None, ref_audios=None,
                prompts=None, video_encode=False, second_pass_windows=None,
                source_latent=None, ref_cache=True, ref_cache_mb=1024):
        try:
            if not isinstance(video_encode, bool):
                raise ValueError("video_encode must be a boolean.")
            if source_latent is not None and second_pass_windows is None:
                raise ValueError("接 source_latent 时请同时接一采的 windows（second_pass_windows）。")
            if source_latent is not None and video_encode:
                raise ValueError("二采已经直接用 source_latent 当初始 latent，请关闭 video_encode。")
            _validate_weights(text_weight, image_weight, video_weight)
            cache = ref_cache_module.configure(ref_cache_mb if ref_cache else 0)
            main, infos = _main_sources(main_ref_videos)
            prompts = prompts or {}
            source_streams = None
            if second_pass_windows is not None:
                plan, source_streams = _second_pass_plan(second_pass_windows, source_latent,
                                                         length, keep_final)
                # 二采的宽高直接由 source_latent 反推（H3 视频 VAE 压缩率 16），
                # 避免用户手填 width/height 与放大后的 latent 不一致。
                height = int(source_streams[0].shape[3]) * 16
                width = int(source_streams[0].shape[4]) * 16
                keys = ordered(prompts)
                selected = tuple(keys[:1] if auto_prompt else keys)
                notice = ()
                if not keys:
                    notice = ("没有连接任何提示词：请在 prompts 上至少连接一个文本节点（空文本也需要一个节点）。",)
                elif not auto_prompt and len(keys) != len(plan.windows):
                    notice = _prompt_count_notice(plan, len(keys), False, window_frames,
                                                  overlap, plan.total_frames)
                outcome = PlanOutcome(plan, selected, notice, None)
            else:
                outcome = _plan(infos[0], length, auto_segment, window_frames, manual_lengths,
                                overlap, keep_final, auto_prompt, prompts)
            plan, selected = outcome.plan, outcome.selected
            paired = dict(main_ref_video_audios or {}) if video_weight else {}
            auxiliary = ref_videos or {}
            pairs = list(_paired_indices(paired, len(main) + len(auxiliary)))
            report = plan_report(plan, infos[0]["path"] or "原生 VIDEO", selected,
                                 auto_prompt, outcome.notice, outcome.solution)
            report += f"\n主视频：{len(main)} 路，同步切片；辅助视频：{len(auxiliary)} 路，每窗从头裁剪。"
            report += "\n参考标签：主视频排在辅助视频前；提示词使用 <Picture n>、<Video n>、<Audio n>。"
            if pairs:
                report += "\n音轨配对：" + "；".join(
                    f"<Audio {j}> ↔ <Video {index + 1}>" for j, (index, _) in enumerate(pairs, 1))
                report += "\n配套AUDIO必须已在上游与对应VIDEO裁到同一时间段；不会再次叠加视频裁剪起点。主视频尾段补帧使用静音。"
            if ref_audios:
                report += f"\n独立音频从 <Audio {len(pairs) + 1}> 开始，共 {len(ref_audios)} 路。"
            report += f"\n条件增益：文本={text_weight:g}；图片={image_weight:g}；所有视频={video_weight:g}（1为原生；0移除并重新编号）"
            if source_streams is not None:
                report += (
                    f"\n二采：复用一采 plan（{plan.window_count} 窗，起点/长度/补帧/重叠逐帧重合，未重新求解重叠）；"
                    f"初始 latent 取自 source_latent（video {int(source_streams[0].shape[2])}×"
                    f"{int(source_streams[0].shape[3])}×{int(source_streams[0].shape[4])}，audio "
                    f"{int(source_streams[1].shape[-1])} 步），跳过 VAE 解码/编码往返；"
                    "音频随窗口原样保留，采样器只重绘画面（lock_audio）。"
                    "\n注意：auto_segment / window_frames / manual_lengths / overlap 在二采模式下不生效，"
                    "分段完全由一采的 windows 决定。"
                    f"\n尺寸：width/height 控件值已忽略，按 source_latent 反推为 {width}×{height}；"
                    "建议把放大节点的「宽」「高」输出接到这两个输入上，倍率一改自动跟，不会脱节。")
            if outcome.notice:
                # The report carries the explanation; block only the tensors so
                # the preview node still runs and can display it.
                return io.NodeOutput(ExecutionBlocker(None), plan.output_frames,
                                      _with_cache_line(report, cache))
            target_video = target_info = None
            if video_encode:
                target_video, target_description = initial_video_target(main[0][1], width, height)
                target_info = video_info(target_video)
                target_info["total_frames"] = min(target_info["total_frames"], plan.output_frames)
                report += (
                    f"\nvideo_encode（主视频初始latent）：启用；第一路主视频原始画面 "
                    f"{target_description['source_width']}×{target_description['source_height']} → {width}×{height}；"
                    "等比缩放填满并中心裁切（lanczos）。视频条件增益不影响此开关；音频初始latent仍为空。"
                    f"\n初始latent读取绕过 {target_description['bypassed_resize_recipes']} 层参考视频空间Resize，"
                    "保留时间裁剪；已物化的低清源只能使用现有像素，无法恢复原始细节。")
            if preview_only:
                progress = comfy.utils.ProgressBar(len(plan.windows))
                for i in range(len(plan.windows)):
                    model_management.throw_exception_if_processing_interrupted()
                    progress.update_absolute(i + 1, len(plan.windows))
                return io.NodeOutput(ExecutionBlocker(None), plan.output_frames,
                                      _with_cache_line(report, cache))
            images = {key: (ref_images or {})[key] for key in ordered(ref_images or {})} if image_weight else {}
            standalone = {key: (ref_audios or {})[key] for key in ordered(ref_audios or {})}
            if (standalone or paired) and audio_vae is None:
                raise ValueError("参考音频需要连接audio_vae。")
            aux = [(key, auxiliary[key]) for key in ordered(auxiliary)] if video_weight else []
            aux_infos = _validate_materials(images, aux, paired, standalone, len(main))
            sizes = [(info["width"], info["height"]) for info in infos + aux_infos]
            # Each paired AUDIO starts on the already-trimmed VIDEO timeline.
            # Bound main audio before audio_slice pads a legal final window,
            # so a longer upstream waveform cannot leak into padded video time.
            for index, audio_key in pairs:
                if index < len(main):
                    audio = paired[audio_key]
                    cap = round(min(infos[index]["total_frames"], plan.output_frames) * audio["sample_rate"] / 24)
                    paired[audio_key] = {**audio, "waveform": audio["waveform"][..., :cap]}
            entries = []
            progress = comfy.utils.ProgressBar(len(plan.windows))
            progress.update_absolute(0, len(plan.windows))
            # negative_prompt 已从接口移除：节点内部固定用空文本，
            # 负面条件仍走 append_window，行为与过去「未连接」时完全一致。
            # 空文本这一次 32B 前向与窗口无关，改时长时没必要重跑。
            clip_key, vae_key, avae_key = (model_identity(clip), model_identity(vae),
                                           model_identity(audio_vae))
            negative = MISSING
            if cache is not None:
                negative = cache.get(("negative", clip_key), "l2")
            if negative is MISSING:
                negative = cpu_tensors(clip.encode_from_tokens_scheduled(clip.tokenize("")))
                if cache is not None:
                    cache.put(("negative", clip_key), negative)
            # 素材身份：参考图与独立音频在所有窗口间不变；主视频按 window.start 同步
            # 切片，所以它的身份里必须带上起点，否则不同窗口会互相命中对方的编码。
            # 图片/音频/source_latent 用内容指纹（跨运行稳定），运行重跑时 L2 整窗
            # 缓存可以直接命中，省掉整段重编码——短视频二采的迭代刚需。
            image_keys = tuple((name, ("image", tensor_fingerprint(image)))
                               for name, image in images.items())
            audio_keys = tuple((name, ("audio", tensor_fingerprint(audio["waveform"])))
                               for name, audio in standalone.items())
            main_keys = tuple(("main", j, _video_source(video, infos[j]))
                              for j, (_, video) in enumerate(main))
            aux_keys = tuple(("aux", j, _video_source(video, aux_infos[j - len(main)]))
                             for j, (_, video) in enumerate(aux, len(main)))
            source_marker = (tensor_fingerprint(source_latent["samples"])
                             if source_latent is not None else None)
            for i, window in enumerate(plan.windows):
                model_management.throw_exception_if_processing_interrupted()
                window_videos, window_audio, video_sources, slice_sources = {}, {}, {}, {}
                if video_weight:
                    lengths = []
                    for j, (_, video) in enumerate(main):
                        read_info = {**infos[j], "total_frames": min(infos[j]["total_frames"], plan.output_frames)}
                        frames = _read_padded(video, read_info, window.start, window.frames, *sizes[j])
                        window_videos[f"ref_video_{j}"] = frames
                        video_sources[f"ref_video_{j}"] = ("main", _video_source(video, infos[j]),
                                                           window.start)
                        lengths.append(window.frames)
                    for j, (_, video) in enumerate(aux, len(main)):
                        count = min(aux_infos[j - len(main)]["total_frames"], window.frames)
                        count = 5 + ((count - 5) // 17) * 17
                        window_videos[f"ref_video_{j}"] = read_video_window(video, 0, count, *sizes[j])
                        video_sources[f"ref_video_{j}"] = ("aux", _video_source(
                            video, aux_infos[j - len(main)]))
                        lengths.append(count)
                    for j, audio_key in pairs:
                        start = window.start if j < len(main) else 0
                        window_audio[f"ref_video_audio_{j}"] = audio_slice(paired[audio_key], start, lengths[j])
                        slice_sources[f"ref_video_audio_{j}"] = video_sources[f"ref_video_{j}"]
                    del frames
                key = selected[0] if auto_prompt else selected[i]
                # 整窗缓存：key 里出现的东西全部决定了这一窗的编码结果。
                window_key = None
                if cache is not None:
                    window_key = (
                        "window", clip_key, vae_key, avae_key,
                        _prompt_text(prompts[key]), text_weight, image_weight, video_weight,
                        width, height, window.start, window.frames, bool(video_encode),
                        source_marker, image_keys, audio_keys, main_keys, aux_keys,
                        tuple((j, audio_key) for j, audio_key in pairs))
                cached_window = cache.get(window_key, "l2") if window_key is not None else MISSING
                if cached_window is MISSING:
                    encoded = encode_weighted_references(
                        clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompts[key],
                        width=width, height=height, length=window.frames,
                        text_weight=text_weight, image_weight=image_weight, video_weight=video_weight,
                        ref_images=images, ref_videos=window_videos,
                        ref_video_audios=window_audio, ref_audios=standalone,
                        reference_encoder=encode_sized_references,
                        reference_kwargs={"cache": cache,
                                          "provenance": {"images": dict(image_keys or ()),
                                                         "videos": video_sources,
                                                         "video_audios": slice_sources,
                                                         "audios": dict(audio_keys or ())}})
                    positive, latent = encoded.result
                    del encoded
                    if window_key is not None:
                        cached_window = (cpu_tensors(positive), cpu_tensors(latent))
                        cache.put(window_key, cached_window)
                    else:
                        cached_window = (positive, latent)
                positive, latent = cached_window
                entries = append_window(positive, negative, latent, window.start, entries,
                                        require_source_video=video_weight != 0)
                window_videos.clear()
                window_audio.clear()
                video_sources.clear()
                slice_sources.clear()
                del positive, latent
                if source_streams is not None:
                    window = entries[-1]["window"]
                    initial_v = source_streams[0][:, :, window.video_start:
                                                  window.video_start + window.video_length]
                    initial_a = source_streams[1][..., window.audio_start:
                                                  window.audio_start + window.audio_length]
                    if (tuple(initial_v.shape) != tuple(window.shapes[0])
                            or tuple(initial_a.shape) != tuple(window.shapes[1])):
                        raise ValueError(
                            "二采的 source_latent 与反推出的像素尺寸编码结果不一致：窗口期望 video %s / audio %s，"
                            "实际切出 %s / %s。宽高本应由 latent ×16 反推，出现这个错误通常说明"
                            "H3 视频 VAE 的空间压缩率不是 16，或 source_latent 不是同一个 VAE 产生的。"
                            % (tuple(window.shapes[0]), tuple(window.shapes[1]),
                               tuple(initial_v.shape), tuple(initial_a.shape)))
                    entries[-1]["initial_video_latent"] = initial_v.detach().to(
                        device="cpu", dtype=torch.float32).clone()
                    entries[-1]["initial_audio_latent"] = initial_a.detach().to(
                        device="cpu", dtype=torch.float32).clone()
                    entries[-1]["lock_audio"] = True
                    del initial_v, initial_a
                if video_encode:
                    if vae is None:
                        raise ValueError("video_encode 需要连接视频VAE。")
                    expected = tuple(entries[-1]["window"].shapes[0])
                    source_key = (vae_key, "source_video", _video_source(main[0][1], infos[0]),
                                  window.start, window.frames, width, height)
                    initial = cache.get(source_key, "l1") if cache is not None else MISSING
                    if initial is MISSING:
                        target_frames = _read_padded(
                            target_video, target_info, window.start, window.frames, width, height)
                        try:
                            initial = vae.encode(target_frames)
                            if not isinstance(initial, torch.Tensor) or tuple(initial.shape) != expected:
                                raise ValueError(f"video_encode 的初始latent形状必须为 {expected}。")
                            if not initial.is_floating_point() or not bool(torch.isfinite(initial).all()):
                                raise ValueError("video_encode 的初始latent必须为有限浮点数。")
                            # Raw VAE space; the sampler performs process_latent_in.
                            # Own storage even when encode returns a view or reused buffer.
                            initial = initial.detach().to(device="cpu", dtype=torch.float32).clone()
                        finally:
                            del target_frames
                        if cache is not None:
                            cache.put(source_key, initial)
                    elif not isinstance(initial, torch.Tensor) or tuple(initial.shape) != expected:
                        raise ValueError(f"video_encode 的初始latent形状必须为 {expected}。")
                    entries[-1]["initial_video_latent"] = initial
                    del initial
                progress.update_absolute(i + 1, len(plan.windows))
                cls._progress_text(i + 1, len(plan.windows), "编码")
            return io.NodeOutput(entries, plan.output_frames, _with_cache_line(report, cache))
        finally:
            if not preview_only:
                # LIFO retains CLIP -> video VAE -> audio VAE cleanup order,
                # while still trying every unload if an earlier one fails.
                with ExitStack() as cleanup:
                    for encoder in (audio_vae, vae, clip if unload else None):
                        patcher = getattr(encoder, "patcher", None)
                        if patcher is not None:
                            cleanup.callback(model_management.unload_model_and_clones, patcher)

    @classmethod
    def _progress_text(cls, completed, total, action):
        PromptServer.instance.send_progress_text(
            f"{action} {completed}/{total} · {completed * 100 // total}%", cls.hidden.unique_id)