"""Still-frame preview route for sources the browser cannot decode.

Chromium/Electron cannot decode many HEVC (H.265) files, so the ``<video>``
element inside ``web/video_crop_preview.js`` never fires ``loadedmetadata`` and
the node stays on "读取 …" forever. VHS Load Video looks fine because it does
not rely on the browser codec for its own preview path.

This module adds ``GET /proyaya/frame``: it decodes one frame with PyAV on the
server (same libav the sampler uses) and returns JPEG, so the frontend can fall
back to a static picture and still draw the crop box.

Nothing here touches node registration; import failure only costs the fallback.
"""

import asyncio
import io
import logging
import os
import subprocess
import tempfile

from aiohttp import web

import folder_paths
from server import PromptServer

ROUTE = "/proyaya/frame"
VIDEO_ROUTE = "/proyaya/preview_video"
INFO_ROUTE = "/proyaya/video_info"
FRAME_MAX_SIDE = 1920
JPEG_QUALITY = 88

# 浏览器内核能直接解的编码：命中就直接发原文件，不转码。
BROWSER_SAFE = {"h264", "vp8", "vp9", "av1"}
PROXY_DIR_NAME = "proyaya_preview"
PROXY_MAX_SIDE = 720
_transcoding = {}   # 目标文件 -> asyncio.Lock，避免同一文件被并发转码


def resolve_path(filename, subfolder, folder_type):
    """把 (filename, subfolder, type) 还原成磁盘路径。

    前端传的是下拉框里的原始值（可能带 ``[input]`` 这类后缀），所以先走
    ``get_annotated_filepath``，它本来就是为这种名字准备的；失败再按
    input/output/temp 目录手工拼一次。
    """
    name = str(filename or "").replace("\\", "/").strip()
    if not name:
        raise ValueError("filename is empty")
    if subfolder:
        name = str(subfolder).replace("\\", "/").strip("/") + "/" + name

    try:
        resolved = folder_paths.get_annotated_filepath(name)
        if resolved and os.path.isfile(resolved):
            return resolved
    except Exception:  # noqa: BLE001 - 注解解析失败就走下面的手工拼接
        pass

    roots = {
        "input": folder_paths.get_input_directory(),
        "output": folder_paths.get_output_directory(),
        "temp": folder_paths.get_temp_directory(),
    }
    root = roots.get(str(folder_type or "input").lower())
    if not root:
        raise ValueError("unknown folder type: %s" % folder_type)
    return os.path.join(root, *name.split("/"))


def first_frame_jpeg(path, max_side=FRAME_MAX_SIDE, quality=JPEG_QUALITY):
    """解第一帧并编码成 JPEG 字节串。"""
    import av

    with av.open(path, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError("no video stream: %s" % os.path.basename(path))
        frame = None
        for frame in container.decode(stream):
            break
        if frame is None:
            raise ValueError("no decodable frame: %s" % os.path.basename(path))
        image = frame.to_image()  # PyAV -> PIL，色彩空间转换由 libav 负责

    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


@PromptServer.instance.routes.get(ROUTE)
async def _proyaya_frame(request):
    filename = (request.query.get("filename") or "").strip()
    if not filename:
        return web.Response(status=400, text="filename is required")
    subfolder = (request.query.get("subfolder") or "").strip()
    folder_type = (request.query.get("type") or "input").strip()

    try:
        path = resolve_path(filename, subfolder, folder_type)
        if not os.path.isfile(path):
            logging.warning("PROYYImageScale: 取帧请求找不到文件: %s", path)
            return web.Response(status=404, text="file not found")
        loop = asyncio.get_running_loop()
        body = await loop.run_in_executor(None, first_frame_jpeg, path)
    except Exception as exc:  # noqa: BLE001 - 预览失败不该影响任何节点
        logging.warning("PROYYImageScale: 取帧预览失败(%s): %s", filename, exc)
        return web.Response(status=500, text=str(exc)[:200])

    # 成功也留一行：排障时能直接确认前端到底有没有发过请求。
    # 用 print 而不是 logging：ComfyUI 的根 logger 不一定放行本模块的 INFO。
    print("[PROYAYA] 取帧预览 %s -> %d bytes" % (os.path.basename(path), len(body)),
          flush=True)
    return web.Response(body=body, content_type="image/jpeg",
                        headers={"Cache-Control": "max-age=120"})


def _ffmpeg_exe():
    """找一个可用的 ffmpeg：先 imageio_ffmpeg 自带的，再 PATH。"""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:  # noqa: BLE001
        pass
    import shutil

    return shutil.which("ffmpeg")


def _video_codec(path):
    import av

    with av.open(path, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        return (stream.codec_context.name if stream else "") or ""


def _video_info(path):
    """容器头级别的轻量信息：源片帧率（跳帧预览定位要用它换算秒）。"""
    import av

    with av.open(path, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return {"fps": None}
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate and rate > 0 else None
        return {"fps": round(fps, 4) if fps else None}


@PromptServer.instance.routes.get(INFO_ROUTE)
async def _proyaya_video_info(request):
    filename = (request.query.get("filename") or "").strip()
    if not filename:
        return web.Response(status=400, text="filename is required")
    subfolder = (request.query.get("subfolder") or "").strip()
    folder_type = (request.query.get("type") or "input").strip()
    try:
        path = resolve_path(filename, subfolder, folder_type)
        if not os.path.isfile(path):
            return web.json_response({"fps": None}, status=404)
        info = await asyncio.get_running_loop().run_in_executor(None, _video_info, path)
        return web.json_response(info)
    except Exception as exc:  # noqa: BLE001 - 预览失败不该影响任何节点
        return web.json_response({"fps": None, "error": str(exc)[:200]})


def _proxy_path(path):
    """源文件 -> 缓存代理文件路径（按 mtime+size 判别，源变了自然重建）。"""
    stat = os.stat(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:60]
    name = "%s-%d-%d.mp4" % (safe, int(stat.st_mtime), int(stat.st_size))
    root = os.path.join(folder_paths.get_temp_directory(), PROXY_DIR_NAME)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name)


def transcode_proxy(src, dst):
    """把任意编码转成浏览器能播的 H.264（预览用，丢音频，长边 720）。"""
    exe = _ffmpeg_exe()
    if not exe:
        raise RuntimeError("未找到 ffmpeg，无法生成预览代理")
    tmp = dst + ".part.mp4"
    cmd = [
        exe, "-y", "-loglevel", "error",
        "-i", src,
        "-vf", "scale='if(gt(iw,ih),min(%d,iw),-2)':'if(gt(iw,ih),-2,min(%d,ih))'"
               % (PROXY_MAX_SIDE, PROXY_MAX_SIDE),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", tmp,
    ]
    subprocess.run(cmd, check=True, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.replace(tmp, dst)


async def ensure_proxy(src, dst):
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst
    lock = _transcoding.setdefault(dst, asyncio.Lock())
    async with lock:
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return dst
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, transcode_proxy, src, dst)
    return dst


@PromptServer.instance.routes.get(VIDEO_ROUTE)
async def _proyaya_preview_video(request):
    """给节点预览用的视频源：能直解就发原文件，否则发 H.264 代理。

    与 /proyaya/frame 的分工：这个让 <video> 真的能播；那条只是静帧兜底。
    """
    filename = (request.query.get("filename") or "").strip()
    if not filename:
        return web.Response(status=400, text="filename is required")
    subfolder = (request.query.get("subfolder") or "").strip()
    folder_type = (request.query.get("type") or "input").strip()

    try:
        path = resolve_path(filename, subfolder, folder_type)
        if not os.path.isfile(path):
            return web.Response(status=404, text="file not found")
        codec = await asyncio.get_running_loop().run_in_executor(
            None, _video_codec, path)
        if codec in BROWSER_SAFE:
            return web.FileResponse(path)
        proxy = await ensure_proxy(path, _proxy_path(path))
        print("[PROYAYA] 预览代理就绪 %s (%s -> h264)"
              % (os.path.basename(path), codec or "?"), flush=True)
        return web.FileResponse(proxy)
    except Exception as exc:  # noqa: BLE001 - 预览失败不该影响生成
        logging.warning("PROYYImageScale: 预览代理失败(%s): %s", filename, exc)
        return web.Response(status=500, text=str(exc)[:200])
