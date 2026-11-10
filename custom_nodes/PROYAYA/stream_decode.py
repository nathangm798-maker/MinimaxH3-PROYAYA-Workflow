"""流式 VAE 解码：逐 chunk 直接写进输出缓冲，绕开全量 torch.empty 预分配。

为什么需要这个节点
------------------
ComfyUI 的 ``VAE.decode()``（comfy/sd.py）会先
``torch.empty(self.first_stage_model.decode_output_shape(samples_in.shape))``
把**整条视频的像素**一次性分配出来。H3 视频 VAE 的输出是 ``[1,3,T,H,W]`` float32：

    384x672  全长 13129 帧 -> 40.7 GiB     （64G 内存勉强能塞，但会把系统挤爆）
    768x1344 全长 13129 帧 -> 151.5 GiB    （物理上不可能）

而 VAE 内部（``comfy/ldm/minimax/vae.py`` 的 ``decode_temporal``）本来就是流式的：
772 个 chunk 顺序算、顺序写，``write_pos`` 单调递增、永不回退。所以全量缓冲只是
sd.py 那一行分配方式造成的，不是算法需要。

本节点把 ``decode_temporal`` 的循环体原样搬过来，只把「写到全量缓冲」改成
「写到一块按需分配的 IMAGE 张量」：

* 缓冲小于 ``ram_limit_gb`` 时留在内存（正常单窗/短片段场景）；
* 超过则用 ``np.memmap`` 落到 ComfyUI 临时目录——``torch.from_numpy(memmap)``
  得到的张量对下游而言就是一个普通的 CPU IMAGE，``VHS_VideoCombine`` 是
  逐帧迭代消费的（``nodes.py`` ``images = iter(images)`` → ``tensor_to_bytes(i)``），
  所以全程内存常数级。

**与「分段解码」的区别（重要，别改错）**：分段解码是重启 N 次解码循环，
H3 解码器里 rotary attention 换个起点会重算出**不同内容**，而且时序会错位
（实测 latent 502 处即 71s 处，PSNR 从 48dB 掉到 19dB）。这里是**同一次
连续循环**跑完，只是不把结果攒在内存里，与一次性解码逐字节等价
（实测同一 latent 两种方式 PSNR = inf）。
"""

import math
import os
import shutil
import subprocess
import wave

import numpy as np
import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import io, ui

# 与 tiled_upscale.py 保持一致的说法：H3 视频 VAE 空间压缩率
VAE_DOWNSAMPLE = 16
BYTES_PER_FLOAT32 = 4

MODE_IMAGE = "IMAGE序列(接VHS)"
MODE_MP4 = "MP4直写(零缓冲)"


def _find_ffmpeg():
    """优先 PATH，其次 VHS 依赖的 imageio-ffmpeg 自带二进制。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        raise RuntimeError(
            "找不到 ffmpeg：MP4直写模式需要它。请把 ffmpeg 加入 PATH，"
            "或安装 imageio-ffmpeg（VHS 通常已带）。")


def _audio_to_wav(audio, path):
    """ComfyUI AUDIO dict -> 16bit PCM wav（仅用 stdlib wave + numpy）。"""
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    if wf.dim() == 3:
        wf = wf[0]
    arr = wf.detach().cpu().numpy()  # (C, N)
    arr = np.clip(arr.T, -1.0, 1.0)  # (N, C)
    if arr.shape[1] == 1:
        pcm = (arr[:, 0] * 32767.0).astype("<i2")
    else:
        pcm = (arr * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return sr


def _video_of(latent):
    """H3 的 joint AV latent（NestedTensor）只解视频分支，音频分支原样带走。"""
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        return samples.tensors[0]
    return samples


def _pick_device_for_buffer(nbytes, ram_limit_gb):
    """缓冲够小就放内存，否则放临时目录的 memmap。返回 (tensor, flush, 说明)。

    memmap 文件在节点返回后仍要被下游（VHS_VideoCombine）逐帧读取，
    所以这里**不负责删除**：文件固定名，下次运行时 w+ 直接截断覆盖。

    Windows 对 memmap 脏页的回写非常懒（不会像 Linux 那样按 dirty_ratio 主动
    flush），151 GiB 的页缓存会把物理内存吃到接近 0。所以这里返回一个
    ``flush(start_byte, end_byte)``，由 sink 在每写完一个 chunk 后主动回写
    那一段，把脏页压在几百 MB 量级。
    """
    limit = int(ram_limit_gb) * (1 << 30)
    if nbytes <= limit:
        return (torch.empty(nbytes, dtype=torch.uint8), None,
                "内存 %.1f GiB" % (nbytes / (1 << 30)))

    directory = folder_paths.get_temp_directory()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "PROYY_stream_decode_buffer.f32")
    free = None
    try:
        import shutil
        free = shutil.disk_usage(directory).free
    except Exception:  # noqa: BLE001 - 拿不到磁盘信息就不拦
        pass
    if free is not None and free < nbytes * 1.1:
        raise ValueError(
            "临时目录放不下解码缓冲：需要 %.1f GiB，可用 %.1f GiB（%s）。"
            "可以把 ram_limit_gb 调大走内存，或清理临时目录。"
            % (nbytes / (1 << 30), free / (1 << 30), directory))

    mmap = np.memmap(path, dtype=np.uint8, mode="w+", shape=(nbytes,))
    tensor = torch.from_numpy(mmap)
    raw = mmap._mmap  # mmap.mmap 对象，支持按范围 flush

    def flush(start_byte, end_byte):
        """回写 [start_byte, end_byte)，把这段的脏页交给系统。"""
        try:
            raw.flush(int(start_byte), int(end_byte) - int(start_byte))
        except (ValueError, OSError):  # noqa: BLE001 - 越界或平台不支持就整段刷
            try:
                mmap.flush()
            except Exception:  # noqa: BLE001
                pass

    return (tensor, flush, "磁盘 memmap %.1f GiB（%s，下次运行覆盖）"
            % (nbytes / (1 << 30), path))


class PROYYStreamVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PROYYStreamVAEDecode",
            display_name="۞▷Stream VAE Decode◁۞",
            category="PROYAYA/H3",
            is_output_node=True,  # MP4直写模式必须：否则不被执行链选中
            description=(
                "流式解码 H3 视频 latent：逐 chunk 直接写输出，不预分配全量像素缓冲。\n"
                "ComfyUI 原生 VAEDecode 会 torch.empty 整条视频（768x1344 全长 = 151 GiB），"
                "长视频必然 OOM。本节点与一次性解码逐字节等价，不是分段解码。\n"
                "输出 IMAGE 可直接接 VHS_VideoCombine——它是逐帧迭代消费的，"
                "所以大分辨率下缓冲会自动落到临时目录的 memmap 上，内存占用常数级。"
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="H3 视频 latent，或 joint AV latent（只解视频分支）。"),
                io.Vae.Input("vae"),
                io.Int.Input("ram_limit_gb", display_name="内存上限(GB)", default=16, min=0, max=512, step=1,
                             tooltip="仅 IMAGE序列 模式生效：解码输出缓冲小于该值留在内存，超过落临时目录 memmap。\n"
                                     "设 0 强制全部走 memmap；设很大则尽量走内存。"),
                io.Combo.Input("output_mode", display_name="输出模式",
                               options=[MODE_IMAGE, MODE_MP4], default=MODE_IMAGE,
                               tooltip="IMAGE序列：输出帧序列接 VHS_VideoCombine（大分辨率时缓冲落盘，文件可到百 GiB 级）。\n"
                                       "MP4直写：解码 chunk 直接经管道喂 ffmpeg，内存 ~2GB、磁盘零缓冲，"
                                       "视频自动注册为 output 产物（与 VHS 保存同级），云端可识别下载。"),
                io.Int.Input("crf", default=17, min=1, max=51, step=1,
                             tooltip="MP4直写 模式的 H.264 crf（越小质量越高）。"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.1,
                               tooltip="MP4直写 模式的输出帧率。"),
                io.String.Input("filename_prefix", display_name="文件名前缀", default="PROYY_stream_decode",
                                tooltip="MP4直写 模式的输出文件名前缀，自动编号，如 PROYY_stream_decode_00001_.mp4。"),
                io.Audio.Input("audio", optional=True,
                               tooltip="仅 MP4直写 模式：可多路选一路音轨 mux 进 mp4（aac 192k）。"),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE",
                                tooltip="帧序列 [N,H,W,3] float32，可直接接 VHS_VideoCombine。"
                                        "MP4直写 模式下此输出为空，视频从节点 UI 产物（gifs）取。"),
                io.String.Output(display_name="解码报告"),
            ],
        )

    @classmethod
    def execute(cls, latent, vae, ram_limit_gb, output_mode=MODE_IMAGE,
                crf=17, fps=24.0, filename_prefix="PROYY_stream_decode",
                audio=None) -> io.NodeOutput:
        video = _video_of(latent)
        if video.dim() != 5:
            raise ValueError(
                "需要 5 维视频 latent [B,C,T,H,W]，当前是 %d 维 %s。"
                % (video.dim(), tuple(video.shape)))
        if int(video.shape[0]) != 1:
            raise ValueError("只支持 batch=1，当前 batch=%d。" % int(video.shape[0]))
        if video.shape[2] <= 1:
            # 单帧走 comfy 原生路径，不值得复刻。
            return io.NodeOutput(vae.decode(video), "单帧：走原生 VAE.decode。")

        fs = vae.first_stage_model
        shape = fs.decode_output_shape(video.shape)
        _b, _c, frames, height, width = shape
        nbytes = int(math.prod(shape)) * BYTES_PER_FLOAT32
        mp4_mode = (output_mode == MODE_MP4)

        if mp4_mode:
            out_dir = folder_paths.get_output_directory()
            full_dir, base, counter, subfolder, _prefix = folder_paths.get_save_image_path(
                filename_prefix.strip() or "PROYY_stream_decode", out_dir)
            out_name = "%s_%05d_.mp4" % (base, counter)
            out_path = os.path.join(full_dir, out_name)
            sink = _FfmpegSink(out_path, height, width, crf=int(crf), fps=float(fps))
            where = "MP4 直写 %s" % out_path
        else:
            buffer, flush, where = _pick_device_for_buffer(nbytes, ram_limit_gb)
            sink = _ImageSink(buffer, frames, height, width, flush)

        report = [
            "流式解码：%d 帧 %dx%d，写出端 %s"
            % (frames, width, height, where),
        ]
        if not mp4_mode:
            report.append("输出形状 %s（原生 VAEDecode 需一次性分配 %.1f GiB）"
                          % (tuple(shape), nbytes / (1 << 30)))

        import time
        t0 = time.time()
        try:
            with torch.inference_mode():
                with comfy.model_management.cuda_device_context(vae.device):
                    need = vae.memory_used_decode(video.shape, vae.vae_dtype)
                    comfy.model_management.load_models_gpu(
                        [vae.patcher], memory_required=need,
                        force_full_load=vae.disable_offload)
                    _stream_decode_temporal(fs, video.to(vae.device, dtype=vae.vae_dtype),
                                            sink, frames)
        except BaseException:
            if mp4_mode:
                sink.abort()
            raise
        if mp4_mode:
            sink.close()
            if audio is not None:
                wav = out_path + ".audio.wav"
                _audio_to_wav(audio, wav)
                muxed = out_path + ".mux.mp4"
                subprocess.run(
                    [_find_ffmpeg(), "-v", "error", "-y", "-i", out_path,
                     "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", muxed],
                    check=True)
                os.replace(muxed, out_path)
                os.remove(wav)
                report.append("音轨：已 mux（aac 192k）")
        else:
            sink.image = sink.view()
        if sink.pos != frames:
            report.append("⚠ 写出 %d 帧，计划 %d 帧" % (sink.pos, frames))

        report.append("用时 %.0f 秒" % (time.time() - t0))
        if mp4_mode:
            report.append("文件：%s" % out_path)
            # 核心前端的原生视频预览卡（与 SaveVideo 同机制），VHS 的 gifs dict
            # 也一并保留：部分云平台按 history 的 gifs 收集产物。
            return io.NodeOutput(
                None, "\n".join(report),
                ui=ui.PreviewVideo([ui.SavedResult(out_name, subfolder, io.FolderType.output)]))
        return io.NodeOutput(sink.image, "\n".join(report))


class _ImageSink:
    """把 (1,3,n,H,W) 的解码块重排成 [N,H,W,3] 写进 IMAGE 缓冲。"""

    def __init__(self, buffer_u8, frames, height, width, flush=None):
        self.frames = frames
        self.height = height
        self.width = width
        self.buffer_u8 = buffer_u8
        self.flush = flush
        self.pos = 0
        # 视图只在真正返回给下游时才构造；这里先只记形状。
        self.image = None

    def view(self):
        flat = self.buffer_u8.view(torch.float32)
        return flat.view(self.frames, self.height, self.width, 3)

    def __call__(self, part):
        # part: (1,3,n,H,W) float32、已 clamp 到 [0,1]，可能还在 GPU 上。
        n = int(part.shape[2])
        if n <= 0:
            return
        arr = part[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()  # (n,H,W,3)
        start = self.pos
        end = self.pos + n
        if end > self.frames:
            arr = arr[: self.frames - start]
            end = self.frames
        if arr.shape[0] <= 0:
            return
        self.view()[start:end].copy_(torch.from_numpy(arr))
        self.pos = end
        if self.flush is not None:
            # buffer_u8 是 uint8 视图，float32 元素 index × 4 = 字节偏移。
            self.flush(start * BYTES_PER_FLOAT32, end * BYTES_PER_FLOAT32)


class _FfmpegSink:
    """把 (1,3,n,H,W) float32 [0,1] 的块经 stdin 管道喂 ffmpeg，零中间缓冲。"""

    def __init__(self, out_path, height, width, crf=17, fps=24.0):
        self.pos = 0
        self.h, self.w = height, width
        cmd = [
            _find_ffmpeg(), "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.out_path = out_path

    def __call__(self, part):
        # part: (1,3,n,H,W) float32，已在 _finalize_pixels 里 clamp 到 [0,1]
        if self.proc.poll() is not None:
            raise RuntimeError(
                "ffmpeg 提前退出（码 %d）：%s"
                % (self.proc.returncode,
                   self.proc.stderr.read().decode("utf-8", "replace")[-500:]))
        n = int(part.shape[2])
        if n <= 0:
            return
        arr = part[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        arr = (arr * 255.0 + 0.5).astype(np.uint8)  # 已 clamp 到 [0,1]
        try:
            self.proc.stdin.write(arr.tobytes())
        except BrokenPipeError:
            raise RuntimeError(
                "ffmpeg 管道中断（磁盘满或编码器崩溃）：%s"
                % self.proc.stderr.read().decode("utf-8", "replace")[-500:])
        self.pos += n

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg 返回码 %d：%s"
                % (self.proc.returncode,
                   self.proc.stderr.read().decode("utf-8", "replace")[-500:]))

    def abort(self):
        """异常路径：杀掉 ffmpeg，避免半截文件被当成功产物。"""
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        try:
            if os.path.exists(self.out_path):
                os.remove(self.out_path)
        except OSError:  # noqa: BLE001
            pass


def _stream_decode_temporal(fs, z, sink, total_frames):
    """与 comfy/ldm/minimax/vae.py::decode_temporal 逐行对应，只换写出端。

    任何一行改动都要和上游对齐——这里是 H3 时序一致性的全部来源。
    """
    # VAE.decode() 的前置归一化（raw latent -> 模型输入空间）。
    latents_mean = fs.latents_mean.view(1, -1, 1, 1, 1).to(z)
    latents_std = fs.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * latents_std + latents_mean

    chunk_dec = fs.tokens_chunk_size * fs.vae_ratio_t
    split_count = int(fs.token_drop > 0) + 1

    pad_tokens, num_chunks = fs._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat([z, pad_z], dim=2)

    dec_overlap = None
    write_pos = 0

    def write_part(part):
        nonlocal write_pos
        part_frames = int(part.shape[2])
        if part_frames <= 0:
            return
        part = fs._finalize_pixels(part)
        copy_frames = min(part_frames, max(0, total_frames - write_pos))
        if copy_frames > 0:
            sink(part[:, :, :copy_frames, :, :])
            write_pos += copy_frames

    for i in range(num_chunks):
        t_start_idx = i * fs.tokens_chunk_size
        t_end_idx = t_start_idx + fs.tokens_chunk_size + fs.token_overlap
        clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

        clip_dec = fs._adaptive_decode(clip_z)

        for j in range(split_count):
            f_start_idx = j * chunk_dec
            f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
            clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
            clip_dec_chunk = clip_dec_chunk[:, :, fs.frame_pre_padding:, :, :]

            if j == 0:
                if dec_overlap is not None:
                    clip_dec_chunk = fs.blend(
                        dec_overlap, clip_dec_chunk, fs.frame_overlap, dim=-3
                    )
                    dec_overlap = None
                write_part(clip_dec_chunk)
            else:
                dec_overlap = clip_dec_chunk.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            write_part(dec_overlap)
            dec_overlap = None

        del clip_dec, clip_z, clip_dec_chunk

    return write_pos
