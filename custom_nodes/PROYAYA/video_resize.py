"""在节点内上传视频，并暴露一个原生 ImageScale 配方，不解码整片。

改造说明（相对上一版）：
- **不再从外部接入 VIDEO**：节点内直接上传/选择 ``input`` 目录下的视频，与官方
  Load Video 同一套机制（``io.UploadType.video`` + ``InputImpl.VideoFromFile``）。
- **``width`` / ``height`` 现在是最终输出尺寸**。此前它们只是「缩放目标」，输出比例
  永远跟着源视频走；现在填多少就是多少，比例不一致时由 ``crop`` 决定怎么处理。
- ``crop``：
  * ``absolute``：按**源画面绝对像素**取一个「宽×高」的居中窗口，1:1 不缩放。1080x1920
    的源要 720x1280，得到的就是中间那块 720x1280 原像素，方框不贴边；
  * ``center``：从源画面中心裁出最大的 w:h 区域，**框内即输出、框外丢弃**，再缩放到目标尺寸；
  * ``disabled``：整帧拉伸到 w:h，不裁切（画面会变形）。
  输出宽高大于源画面时（只可能发生在 ``absolute``），没有画面的部分填黑。
- ``single_side`` 开启时只认 ``width``，把它作用在长边（横屏是宽、竖屏是高），另一条边
  按源比例自动算。此时输出比例必然等于源比例，``crop`` 不起作用 —— 方框等于整帧。
- 前端会在节点内画出上传视频的画面，并把上面算出来的取景框叠在上面；框内区域就是
  实际输出的画面。框不可拖动，完全由宽/高/crop/single_side 推导。

两套尺寸互相独立：
- **视频缩放尺寸**：由 ``width`` / ``height`` / ``single_side`` / ``crop`` 决定，作用在输出的 VIDEO 上。
- **目标视频尺寸**：由 ``target_mp``（单位 M 像素，1.0 = 100 万像素）反推，按参考视频
  宽高比拆成宽高，再按横竖屏分配到「目标视频宽」「目标视频高」两个 INT 输出，两个值都对 32 取整。

音频输出直接取源视频的音频轨（缩放/裁切不改变声音），只 demux 音频流、不解任何视频帧；
源片本身没有音频轨时该输出为 ``None``，下游即使连了线也等于没接。
"""

import logging
import math
import os
from fractions import Fraction

import folder_paths
import nodes as comfy_nodes
from comfy_api.latest import InputImpl, io

from .native_video import (ABSOLUTE_CROP, CROP_METHODS, FPS as TIMELINE_FPS,
                          WindowResizedVideo, read_audio_track, source_frame_rate)

# 输出尺寸对齐粒度：生成模型硬性要求宽高都是 32 的倍数，所有路径统一 32。
SIZE_ALIGN = 32
MIN_SIDE = 32
# 目标生成尺寸的对齐栅格：长短边统一按 32 对齐（生成模型硬性要求）。
# 标定点（参考片为竖屏 9:16 时）：
#   0.25M -> 416x704   1.00M -> 768x1344   2.00M -> 1888x1088
#   3.68M -> 2560x1440   8.29M -> 2176x3840
# 可用环境变量覆盖（PROYY_LONG_ALIGN / PROYY_SHORT_ALIGN）。
LONG_ALIGN = int(os.environ.get("PROYY_LONG_ALIGN", "32"))
SHORT_ALIGN = int(os.environ.get("PROYY_SHORT_ALIGN", "32"))

# 视频下拉框里的「不选择」项。放第一项 = 新建节点的默认值。
NO_FILE = ""


def _align(value, align=SIZE_ALIGN, minimum=MIN_SIDE):
    """向下对齐到 align 的整数倍，并保证不小于 minimum。"""
    aligned = int(value) // align * align
    return max(minimum, aligned)


def _ceil_to(value, align=LONG_ALIGN):
    """向上对齐到 align 的整数倍。"""
    return max(align, int(math.ceil(float(value) / align)) * align)


def source_dimensions(video):
    """读取源视频宽高。

    优先走 ``get_dimensions()``：只打开容器读流信息，不解码任何帧
    （比"取一帧"更省，且与 PROYAYA 内部对尺寸的一致性校验口径相同）。
    只有在拿不到元数据时才回退到物化首帧。
    """
    getter = getattr(video, "get_dimensions", None)
    if callable(getter):
        try:
            size = getter()
            width, height = int(size[0]), int(size[1])
            if width > 0 and height > 0:
                return width, height
        except Exception as exc:  # noqa: BLE001 - 任何异常都回退
            logging.warning("PROYYImageScale: get_dimensions() 失败(%s)，回退到取首帧。", exc)

    components = video.get_components()
    images = components.images
    logging.warning("PROYYImageScale: 已物化整片 VIDEO 用于取尺寸，建议改用含元数据的文件源。")
    return int(images.shape[2]), int(images.shape[1])


def resolve_output_size(src_width, src_height, width, height, single_side):
    """把「宽 / 高 / 单边开关」翻译成最终输出尺寸。

    ``single_side=True``：只认 ``width``，作用在源视频长边上。两条边都向上
    取整到 ``SHORT_ALIGN``（默认 32）：生成模型要求宽高都是 32 的倍数，
    单边缩放不能例外。比例因此可能偏离源比例一点点（例如 9:16 源在长边
    640 时短边 360 -> 384），偏差交给 ``crop`` 处理。
    ``single_side=False``：``width`` / ``height`` 就是输出尺寸，不做比例修正；
    比例不一致时由 ``crop`` 决定裁切还是拉伸。
    """
    if src_width <= 0 or src_height <= 0:
        raise ValueError("源视频尺寸无效：%sx%s" % (src_width, src_height))

    src_w, src_h = int(src_width), int(src_height)
    if single_side:
        src_long = max(src_w, src_h)
        src_short = min(src_w, src_h)
        ratio = src_short / float(src_long)
        target_long = _ceil_to(max(int(width), MIN_SIDE), SHORT_ALIGN)
        short_side = _ceil_to(target_long * ratio, SHORT_ALIGN)
        if src_h > src_w:   # 竖屏：长边在高 -> (宽=短边, 高=长边)
            return short_side, target_long
        return target_long, short_side

    return _align(int(width)), _align(int(height))


def crop_region(src_width, src_height, out_width, out_height, crop):
    """返回取景框矩形 ``(x, y, w, h)``，单位为源像素。

    三种语义：
    * ``absolute``：取一个**宽×高 源像素**的居中窗口，1:1 不缩放。窗口可以比画面大，
      此时 ``x/y`` 为负、``w/h`` 大于源尺寸，超出的部分在输出里填黑；
    * ``center``：从中心裁出最大的「宽:高」区域再缩放到目标尺寸（会缩放）；
    * ``disabled``：整帧拉伸到目标尺寸，不裁切。

    ``center`` 分支与 ``comfy.utils.common_upscale`` 的 ``crop="center"`` 逐行同构
    （同样的浮点运算、同样的 ``round``），所以框画在哪里，真正裁的就是哪里。
    前端预览的取景框按同一套算法绘制。
    """
    src_w, src_h = int(src_width), int(src_height)
    out_w, out_h = int(out_width), int(out_height)
    if out_w <= 0 or out_h <= 0 or src_w <= 0 or src_h <= 0:
        return 0, 0, src_w, src_h

    if crop == ABSOLUTE_CROP:
        # 与 _absolute_crop_pad 用的是同一个取整方式（向下取整），保证预览和实际一致。
        return (src_w - out_w) // 2, (src_h - out_h) // 2, out_w, out_h

    if crop != "center" or src_w * out_h == out_w * src_h:
        return 0, 0, src_w, src_h

    old_aspect = src_w / src_h
    new_aspect = out_w / out_h
    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((src_w - src_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((src_h - src_h * (old_aspect / new_aspect)) / 2)
    return x, y, src_w - x * 2, src_h - y * 2


def resolve_generation_size(ref_width, ref_height, target_mp):
    """算出「目标生成尺寸」。

    **比例取自裁切后的输出尺寸**，不是源视频：``crop=center`` 把 9:16 源裁成
    544x544 时，生成分辨率也必须是正方形（``target_mp`` 只决定像素总量）。

    ``target_mp > 0``（单位 M 像素，1.0 = 100 万像素）：按**像素总量**反推宽高。
    保持参考比例 r = 短边 / 长边：
        长边 = √(像素总量 / r)  → 向上取整到 32 的倍数
        短边 = 长边 × r         → 向上取整到 32 的倍数
    两条边都向上取整，最终像素总量恒不小于输入值；且宽高必然都是 32 的倍数。
    标定点（输出为竖屏 9:16 时）：
        0.25M -> 416x704
        1.00M -> 768x1344
        2.00M -> 1088x1888 即 1888x1088（横屏）
        3.68M -> 1440x2560 即 2560x1440（横屏）
        8.29M -> 2176x3840

    ``target_mp <= 0``（未启用）：按自定义尺寸，取本节点上填的宽高，
    同样按 32 向上对齐，并归到正确的轴上。
    """
    ref_long = max(int(ref_width), int(ref_height))
    ref_short = min(int(ref_width), int(ref_height))
    portrait = int(ref_height) > int(ref_width)

    try:
        mp = round(float(target_mp or 0), 2)
    except (TypeError, ValueError):
        mp = 0.0

    if ref_long <= 0 or ref_short <= 0:
        raise ValueError("参考尺寸无效：%sx%s" % (ref_width, ref_height))

    if mp > 0:
        ratio = ref_short / float(ref_long)          # 短/长，<= 1
        long_side = _ceil_to(math.sqrt(mp * 1_000_000.0 / ratio), LONG_ALIGN)
        short_side = min(_ceil_to(long_side * ratio, SHORT_ALIGN), long_side)
    else:
        long_side = _ceil_to(ref_long, LONG_ALIGN)
        short_side = min(_ceil_to(ref_short, SHORT_ALIGN), long_side)

    if portrait:
        return short_side, long_side
    return long_side, short_side


def _input_videos():
    """列出 input 目录下可用的视频，供上传下拉框使用。

    第一项固定是空字符串，代表「不选择任何视频」——ComfyUI 的组合框默认取
    列表第一项，所以新建节点时视频是空的，不会自动挑一个 input 目录里的片子。
    """
    try:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return [NO_FILE] + sorted(folder_paths.filter_files_content_types(files, ["video"]))
    except Exception as exc:  # noqa: BLE001 - 目录不可读时给空列表，前端仍可上传
        logging.warning("PROYYImageScale: 列举 input 视频失败(%s)。", exc)
        return [NO_FILE]


class PROYYImageScale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PROYYImageScale",
            display_name="۞▷Resize Video◁۞",
            category="PROYAYA/H3",
            description=(
                "在节点内上传视频，按宽高与 crop 输出缩放后的 VIDEO；H3分窗读取时才解码当前窗口。"
                "音频输出取源视频的音频轨，源片没有音频轨时为空。"
                "宽、高是最终输出尺寸。crop 有三种："
                "absolute 按源画面绝对像素取一个「宽×高」的居中窗口（1:1 不缩放，窗口比画面大时填黑）；"
                "center 从中心裁出最大的 w:h 区域再缩放（不变形、不填黑）；"
                "disabled 整帧拉伸（不裁切，会变形）。"
                "单边缩放开启时只认「宽」并作用在长边（横屏作用在宽、竖屏作用在高），另一条边按源比例自动算，"
                "此时三种 crop 都不起作用，取景框等于整帧。"
                "目标像素（M）用于算下游生成尺寸：按像素总量反推宽高，比例跟参考视频一致，"
                "长边向上对齐 64、短边向上对齐 32（1.0→768x1344，2.0→1920x1088，3.68→2560x1440，8.29→2176x3840），"
                "再按横竖屏分配到「目标视频宽」「目标视频高」两个整数输出；填 0 时按自定义宽高输出。"
            ),
            inputs=[
                io.Combo.Input("file", options=_input_videos(), default=NO_FILE,
                               upload=io.UploadType.video,
                               tooltip="在节点内上传或选择视频。默认留空不选；空着时无法出帧，需先选一个。"
                                       "选好后下方会显示画面，并叠出按当前宽高/crop 算出的取景框"
                                       "——框内区域就是输出画面。"),
                io.Int.Input("width", display_name="宽", default=480, min=32, step=32,
                             tooltip="单边缩放开启时：此值作用在视频长边上——横屏是宽，竖屏是高。\n"
                                     "关闭时：直接就是输出宽度。"),
                io.Int.Input("height", display_name="高", default=832, min=32, step=32,
                             tooltip="单边缩放开启时此框会被锁定（灰显），数值由长边按源比例自动算。\n"
                                     "关闭时：直接就是输出高度。"),
                io.Combo.Input("upscale_method", options=comfy_nodes.ImageScale.upscale_methods, default="area"),
                io.Combo.Input("crop", options=list(CROP_METHODS), default="center",
                               tooltip="absolute：按**源画面绝对像素**取一个「宽×高」的居中窗口，1:1 不缩放；"
                                       "窗口比画面大时，没有画面的部分填黑。\n"
                                       "center：从中心裁出最大的「宽:高」区域再缩放到「宽×高」，画面不变形、不填黑。\n"
                                       "disabled：整帧拉伸到「宽×高」，不裁切，比例不一致时会变形。\n"
                                       "单边缩放开启时输出比例恒等于源比例，三种模式都不起作用。"),
                io.Boolean.Input("single_side", display_name="单边缩放", default=False,
                                 tooltip="开：只填宽，高自动（锁定灰显），保持源比例、不裁切；"
                                         "关：宽高都可填，比例不一致时按 crop 处理。"),
                io.Float.Input("target_mp", display_name="目标像素（M，1.0=100万）", default=0.98,
                               min=0.0, max=200.0, step=0.01, round=0.01,
                               tooltip="按像素总量自动反推宽高：比例跟参考视频一致，长边向上对齐 64、短边向上对齐 32，"
                                       "保证输出的宽高都是 32 的倍数且像素不少于输入值。1.0 = 100 万像素，支持两位小数；"
                                       "0 表示不启用，改按本节点填的宽高输出（同样按 64/32 向上对齐）。"),
                io.Int.Input("read_frames", display_name="读取帧数（0 = 全部）", default=0,
                             min=0, max=1000000, step=1,
                             tooltip="只把前 N 帧交给下游。\n"
                                     "GetVideoComponents 这类「整片解码」的节点会因此只解 N 帧，"
                                     "几千帧的长源片能省掉绝大部分解码时间。\n"
                                     "H3 的逐窗读取本身就有界，不受此项影响；窗口计划的总长是多少就填多少"
                                     "（例如总长 328 帧就填 328）。0 = 不限制。"),
                io.Int.Input("skip_frames", display_name="跳过前 N 帧", default=0,
                             min=0, max=1000000, step=1,
                             tooltip="丢弃源视频最前面的 N 帧（按**源片帧率**数帧：30fps 片子填 24 就是从第 25 帧开始），"
                                     "从第 N+1 帧开始输出。\n"
                                     "音频同步裁掉对应开头，避免声画错位。\n"
                                     "与「读取帧数」可叠加：先跳过再读 N 帧。\n"
                                     "预览播放器的进度条会同步定位到跳过后的起点。0 = 不跳过。"),
            ],
            outputs=[
                io.Video.Output(display_name="视频"),
                io.Audio.Output(display_name="音频"),
                io.Int.Output(display_name="目标视频宽"),
                io.Int.Output(display_name="目标视频高"),
            ],
        )

    @classmethod
    def execute(cls, file, width, height, upscale_method="area", crop="center",
                single_side=False, target_mp=0.0, read_frames=0, skip_frames=0):
        if not str(file or "").strip():
            # validate_inputs 会先拦住排队；这里再兜一层，防止走 API 绕过前端校验。
            raise ValueError("PROYYImageScale: 未选择视频，无法出帧。")
        if not isinstance(read_frames, int) or isinstance(read_frames, bool) or read_frames < 0:
            raise ValueError("读取帧数必须是不小于 0 的整数（0 = 全部）。")
        if not isinstance(skip_frames, int) or isinstance(skip_frames, bool) or skip_frames < 0:
            raise ValueError("跳过帧数必须是不小于 0 的整数。")
        video_path = folder_paths.get_annotated_filepath(file)
        source = InputImpl.VideoFromFile(video_path)

        src_width, src_height = source_dimensions(source)
        video_width, video_height = resolve_output_size(
            src_width, src_height, width, height, bool(single_side)
        )
        gen_width, gen_height = resolve_generation_size(
            video_width, video_height, target_mp
        )
        box_x, box_y, box_w, box_h = crop_region(
            src_width, src_height, video_width, video_height, crop
        )
        # 跳过 N 帧 = 从源片的第 N+1 帧开始（按**源片帧率**数帧，不是 24fps 时间轴）。
        # 起点 = 第 N 帧的 PTS = N / 源fps；之后 H3 仍按 24fps 时间轴取帧。
        skip_offset = Fraction(0)
        if int(skip_frames):
            source_fps = source_frame_rate(video_path) or float(TIMELINE_FPS)
            fps_fraction = Fraction(source_fps).limit_denominator(1001)
            skip_offset = Fraction(int(skip_frames)) / fps_fraction
        # 只解音频轨，不解视频；没有音频轨时立刻返回 None（下游拿到空值＝没接）。
        audio = read_audio_track(video_path)
        if audio is not None and skip_offset > 0:
            # 视频起点后移了，音频必须跟着裁掉同样的开头，否则声画错位。
            rate = int(audio["sample_rate"])
            begin = round(skip_offset * rate)
            if begin > 0:
                audio = {**audio, "waveform": audio["waveform"][..., begin:].clone()}
        logging.info(
            "PROYYImageScale: 源 %sx%s（%s）单边=%s crop=%s | 取景 %sx%s+%s+%s -> 输出 %sx%s | "
            "目标视频 %.2fM -> %sx%s | 跳过前 %s 帧 | 读取上限 %s | 音频 %s",
            src_width, src_height,
            "横屏" if src_width >= src_height else "竖屏",
            bool(single_side), crop, box_w, box_h, box_x, box_y, video_width, video_height,
            float(target_mp or 0.0), gen_width, gen_height,
            int(skip_frames),
            "全部" if not int(read_frames) else "{} 帧".format(int(read_frames)),
            "无" if audio is None else "{}Hz {} 声道 {:.2f}s".format(
                audio["sample_rate"], audio["waveform"].shape[1],
                audio["waveform"].shape[-1] / float(audio["sample_rate"]),
            ),
        )
        return io.NodeOutput(
            WindowResizedVideo(source, video_width, video_height, upscale_method, crop,
                               start_offset=skip_offset,
                               read_limit=int(read_frames)),
            audio,
            gen_width,
            gen_height,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # 基类签名就是 (**kwargs)，ComfyUI 会把所有输入都传进来；
        # 只按 file 的修改时间做指纹，避免大文件反复重算内容哈希。
        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(kwargs.get("file", "")))
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def validate_inputs(cls, file):
        if not str(file or "").strip():
            return "未选择视频：请在节点内上传，或从下拉框里选一个 input 目录下的视频。"
        if not folder_paths.exists_annotated_filepath(file):
            return "无效的视频文件：{}".format(file)
        return True
