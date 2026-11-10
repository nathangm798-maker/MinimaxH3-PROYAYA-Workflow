import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * PROYYImageScale 的取景框预览。
 *
 * 三件事：
 *   1. 在节点里显示上传视频的画面（DOM <video>，可播放/拖动时间轴）；
 *   2. 按 宽/高/crop/single_side 算出「实际会输出的区域」，画一个框叠上去；
 *   3. 参数、文件、节点尺寸一变就重画。
 *
 * 框的算法必须和 video_resize.py 的 resolve_output_size / crop_region 完全一致，
 * 否则画出来的框和真正输出的画面不符。
 */

const NODE_NAME = "PROYYImageScale";
/** 未合并时那个独立小窗的高度。 */
const PREVIEW_H = 200;
/**
 * 预览区固定高度 = 小窗的 2 倍。
 *
 * 不这样做的话节点高度会跟着视频分辨率跑：ComfyUI 的原生视频预览把
 * computeSize()[1] 报成「视频按宽度等比后的高度」，竖屏视频立刻把节点撑高。
 * 固定后无论什么分辨率，预览框都是这个高度，画面在里面等比缩放。
 */
const PREVIEW_FIXED_H = PREVIEW_H * 2;
const ALIGN = 32;   // 与后端 video_resize.SIZE_ALIGN 一致（生成要求 32 的倍数）
const MIN_SIDE = 32;

/**
 * 底部控件条的固定高度（px）。
 *
 * 为什么不用 <video controls>：浏览器原生控件画在 video 的「内容盒」上，
 * 任何让内容盒变小的做法（padding / transform / 改 width）都会把控件一起缩小。
 * 所以这里关掉原生控件，自己画一条高度、字号全是固定像素的条，画面区域
 * 就是「容器高 - BAR_H」，两者彻底解耦。
 */
/* 控件条尺寸。全部是固定像素，和画面缩放完全无关。 */
const BAR_H = 42;
const BAR_PAD = 10;
const BAR_BUTTON = 28;
const BAR_FONT = 13;
const BAR_TRACK_H = 20;
const BAR_RAIL_H = 4;
const BAR_THUMB = 14;
const STYLE_ID = "proyaya-crop-preview-style";

/** 时间轴等样式只能靠注入 <style>。 */
function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // 关键：ComfyUI 的预览容器常常是 pointer-events:none，只让里面的 <video> 自己
    // 恢复 auto。这样控件条会继承 none —— 看着在、点着穿透。整条 bar 连同子孙
    // 一律强制 auto。
    style.textContent = [
        ".proyaya-bar,.proyaya-bar *{pointer-events:auto !important;}",
        ".proyaya-video{pointer-events:auto !important;}",
    ].join("");
    (document.head || document.body).appendChild(style);
}

function findWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function alignDown(value) {
    const v = Math.floor(Number(value) || 0);
    return Math.max(MIN_SIDE, Math.floor(v / ALIGN) * ALIGN);
}

function alignUp(value) {
    const v = Math.ceil(Number(value) || 0);
    return Math.max(ALIGN, Math.ceil(v / ALIGN) * ALIGN);
}

function isOn(widget) {
    const v = widget?.value;
    return v === true || v === "true" || v === 1;
}

/** 对应 resolve_output_size：单边模式长边向上取整到 32，短边按源比例同样向上取整到 32。 */
function outputSize(srcW, srcH, width, height, singleSide) {
    if (!srcW || !srcH) return null;
    if (singleSide) {
        const srcLong = Math.max(srcW, srcH);
        const srcShort = Math.min(srcW, srcH);
        const ratio = srcShort / srcLong;
        const long = alignUp(Math.max(Number(width) || 0, MIN_SIDE));
        const short = alignUp(long * ratio);
        if (srcH > srcW) return [short, long];   // 竖屏：长边在高
        return [long, short];
    }
    return [alignDown(width), alignDown(height)];
}

/** 对应 resolve_generation_size：按目标像素量(MP)与**裁切后输出比例**反推生成宽高，两边都 ceil 到 32。 */
function generationSize(refW, refH, mp) {
    const refLong = Math.max(refW, refH);
    const refShort = Math.min(refW, refH);
    if (!(mp > 0) || !refLong || !refShort) return null;
    const ratio = refShort / refLong;
    const longSide = alignUp(Math.sqrt((mp * 1000000) / ratio));
    const shortSide = Math.min(alignUp(longSide * ratio), longSide);
    if (refH > refW) return [shortSide, longSide];
    return [longSide, shortSide];
}

/** 对应 crop_region：absolute 取固定像素窗口，center 与 common_upscale 同构，其余取整帧。 */
function cropBox(srcW, srcH, outW, outH, crop) {
    if (!outW || !outH || !srcW || !srcH) return [0, 0, srcW, srcH];
    if (crop === "absolute") {
        return [Math.floor((srcW - outW) / 2), Math.floor((srcH - outH) / 2), outW, outH];
    }
    if (crop !== "center" || srcW * outH === outW * srcH) {
        return [0, 0, srcW, srcH];
    }
    const oldAspect = srcW / srcH;
    const newAspect = outW / outH;
    let x = 0;
    let y = 0;
    if (oldAspect > newAspect) {
        x = Math.round((srcW - srcW * (newAspect / oldAspect)) / 2);
    } else if (oldAspect < newAspect) {
        y = Math.round((srcH - srcH * (oldAspect / newAspect)) / 2);
    }
    return [x, y, srcW - x * 2, srcH - y * 2];
}

function viewUrl(value) {
    const cleaned = String(value || "").replace(/\s*\[(input|output|temp)\]\s*$/, "");
    const norm = cleaned.replace(/\\/g, "/");
    const cut = norm.lastIndexOf("/");
    const filename = cut >= 0 ? norm.slice(cut + 1) : norm;
    const subfolder = cut >= 0 ? norm.slice(0, cut) : "";
    const query = new URLSearchParams({ filename, type: "input", rand: String(Math.random()) });
    if (subfolder) query.set("subfolder", subfolder);
    const path = "/view?" + query.toString();
    return api?.apiURL ? api.apiURL(path) : path;
}

function drawNode(node) {
    const state = node.__cropPreview;
    if (!state) return;
    // 兜底：文件变了但没有任何回调通知到我们（下拉选择、上传按钮、载入工作流
    // 走的是不同代码路径，不一定经过 widget.setValue）。这里每帧比对一次。
    if ((findWidget(node, "file")?.value ?? "") !== state.value) {
        refreshSource(node, true);
        return;
    }
    const { canvas, video } = state;
    const container = canvas.parentElement;
    if (!container) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;

    const srcW = state.srcW;
    const srcH = state.srcH;
    const hasSource = !!(srcW && srcH);

    // 静帧兜底时没有可播放的视频：不占控件条，画面由画布直接画。
    const showBar = hasSource && !state.frozen;
    const barH = showBar ? BAR_H : 0;
    const areaH = Math.max(1, ch - barH);
    if (state.bar) state.bar.style.display = showBar ? "flex" : "none";

    const dpr = window.devicePixelRatio || 1;
    const pixelW = Math.round(cw * dpr);
    const pixelH = Math.round(areaH * dpr);
    if (canvas.width !== pixelW || canvas.height !== pixelH) {
        canvas.width = pixelW;
        canvas.height = pixelH;
        canvas.style.width = cw + "px";
        canvas.style.height = areaH + "px";
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, areaH);

    if (!hasSource) {
        // 没源就把画面元素收到 0，免得残留上一帧
        video.style.width = "0px";
        video.style.height = "0px";
        ctx.fillStyle = "rgba(160,160,160,0.9)";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(state.message || "选择或上传视频后显示画面", cw / 2, areaH / 2);
        return;
    }

    // 取景框：absolute 模式下框可能比画面大，画多大取决于「画面 ∪ 框」的外接矩形。
    const size = outputSize(srcW, srcH, findWidget(node, "width")?.value,
        findWidget(node, "height")?.value, isOn(findWidget(node, "single_side")));
    const box = size
        ? cropBox(srcW, srcH, size[0], size[1], findWidget(node, "crop")?.value)
        : [0, 0, srcW, srcH];

    const ux0 = Math.min(0, box[0]);
    const uy0 = Math.min(0, box[1]);
    const ux1 = Math.max(srcW, box[0] + box[2]);
    const uy1 = Math.max(srcH, box[1] + box[3]);
    const unionW = Math.max(1, ux1 - ux0);
    const unionH = Math.max(1, uy1 - uy0);
    const scale = Math.min(cw / unionW, areaH / unionH);

    // 画面交给 <video> 自己渲染，画布只画框 —— 把视频帧重画到画布上等于盖了一层
    // 不透明覆盖物，播放/时间轴就会被遮住。
    //
    // 缩放只改 video 的宽高（绝对定位 + 精确 px），不碰 padding、不碰 transform：
    // 元素外框就是画面本身，控件根本不在这条链上 —— 它们在我们自己的 bar 里。
    const dw = srcW * scale;
    const dh = srcH * scale;
    const dx = (cw - dw) / 2;
    const dy = (areaH - dh) / 2;

    video.style.left = dx.toFixed(2) + "px";
    video.style.top = dy.toFixed(2) + "px";
    video.style.width = dw.toFixed(2) + "px";
    video.style.height = dh.toFixed(2) + "px";

    // 编码不受支持时，画面来自后端取的一帧静图，直接画在画布上（video 收到 0）。
    if (state.frozen && state.image) {
        video.style.width = "0px";
        video.style.height = "0px";
        try {
            ctx.drawImage(state.image, dx, dy, dw, dh);
            ctx.font = "11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "bottom";
            ctx.fillStyle = "rgba(255,255,255,0.85)";
            ctx.fillText("静帧预览（浏览器无法解码该编码，采样不受影响）", 6, areaH - 4);
        } catch (error) {
            /* 图片还在解码，下一帧再试 */
        }
    }

    // 右下角：按「目标像素量(MP)」反推的生成宽高（= 目标视频宽/高输出值）。
    // 比例跟着**裁切后的输出尺寸**走（不是源比例），位置贴在取景框内侧右下角。
    const targetMp = Number(findWidget(node, "target_mp")?.value) || 0;
    const refSize = size || [srcW, srcH];
    const gen = generationSize(refSize[0], refSize[1], targetMp);
    if (gen) {
        const glabel = "目标 " + gen[0] + "×" + gen[1];
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        const gpad = 5;
        const gw = ctx.measureText(glabel).width + gpad * 2;
        const gh = 18;
        const boxRight = dx + (box[0] + box[2]) * scale;
        const boxBottom = dy + (box[1] + box[3]) * scale;
        const gx = Math.max(2, Math.min(boxRight - gw - 4, cw - gw - 2));
        const gy = Math.max(2, Math.min(boxBottom - gh - 4, areaH - gh - 2));
        ctx.fillStyle = "rgba(10,40,90,0.78)";
        ctx.fillRect(gx, gy, gw, gh);
        ctx.strokeStyle = "rgba(77,163,255,0.9)";
        ctx.lineWidth = 1;
        ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);
        ctx.fillStyle = "rgba(120,190,255,0.98)";
        ctx.fillText(glabel, gx + gw - gpad, gy + gh / 2);
    }

    // 蓝色虚线框延后到白框之后绘制（见下），保证压在白线之上可见。

    // 跳帧提示（右上角）：非 0 时显示跳过了多少帧、换算成源片里的起点时刻。
    const skipFramesRaw = Math.max(0, Math.round(Number(findWidget(node, "skip_frames")?.value) || 0));
    if (skipFramesRaw > 0) {
        const fps = state.srcFps || TIMELINE_FPS;
        const skipLabel = "跳过前 " + skipFramesRaw + " 帧（源 " + fps + "fps）";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        const spad = 5;
        const sw = ctx.measureText(skipLabel).width + spad * 2;
        const sh = 18;
        const sx = Math.max(2, cw - sw - 2);
        const sy = Math.max(2, 2);
        ctx.fillStyle = "rgba(10,40,90,0.78)";
        ctx.fillRect(sx, sy, sw, sh);
        ctx.strokeStyle = "rgba(77,163,255,0.9)";
        ctx.lineWidth = 1;
        ctx.strokeRect(sx + 0.5, sy + 0.5, sw - 1, sh - 1);
        ctx.fillStyle = "rgba(120,190,255,0.98)";
        ctx.fillText(skipLabel, sx + sw - spad, sy + sh / 2);
    }

    if (!size) return;

    const bx = dx + box[0] * scale;
    const by = dy + box[1] * scale;
    const bw = box[2] * scale;
    const bh = box[3] * scale;

    // 先描一道深色再压一道白线：亮背景和暗背景上都看得清
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.strokeRect(bx, by, bw, bh);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.strokeRect(bx + 0.75, by + 0.75, Math.max(0, bw - 1.5), Math.max(0, bh - 1.5));

    // 四角短标记，方便看清边界
    const tick = Math.min(14, bw / 3, bh / 3);
    if (tick > 2) {
        const corners = [
            [bx, by + tick, bx, by, bx + tick, by],
            [bx + bw - tick, by, bx + bw, by, bx + bw, by + tick],
            [bx + bw, by + bh - tick, bx + bw, by + bh, bx + bw - tick, by + bh],
            [bx + tick, by + bh, bx, by + bh, bx, by + bh - tick],
        ];
        ctx.lineWidth = 4.5;
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.beginPath();
        for (const c of corners) {
            ctx.moveTo(c[0], c[1]); ctx.lineTo(c[2], c[3]); ctx.lineTo(c[4], c[5]);
        }
        ctx.stroke();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.beginPath();
        for (const c of corners) {
            ctx.moveTo(c[0], c[1]); ctx.lineTo(c[2], c[3]); ctx.lineTo(c[4], c[5]);
        }
        ctx.stroke();
    }

    // 蓝色虚线取景框：只表达**比例**，不表达绝对尺寸。
    // 按目标宽高的纵横比，在白色裁切框内部做最大内接矩形并居中 ——
    // 比例完全一致时与白框重合；有偏差立刻能看出宽了还是窄了，也不会溢出画面。
    // 画在白色框/四角标记之后，保证它压在上面（重合时才看得见蓝）。
    if (gen) {
        const genRatio = gen[0] / gen[1];
        const cropRatio = box[2] / box[3];
        let bwSrc;
        let bhSrc;
        if (cropRatio > genRatio) {
            bhSrc = box[3];
            bwSrc = bhSrc * genRatio;
        } else {
            bwSrc = box[2];
            bhSrc = bwSrc / genRatio;
        }
        const boxCx = dx + (box[0] + box[2] / 2) * scale;
        const boxCy = dy + (box[1] + box[3] / 2) * scale;
        const fw = bwSrc * scale;
        const fh = bhSrc * scale;
        const fx = boxCx - fw / 2;
        const fy = boxCy - fh / 2;
        ctx.save();
        ctx.setLineDash([7, 5]);
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.strokeRect(fx, fy, fw, fh);
        ctx.setLineDash([7, 5]);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(77,163,255,0.98)";
        ctx.strokeRect(fx, fy, fw, fh);
        ctx.restore();
    }

    const label = size[0] + "×" + size[1];
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const padding = 5;
    const labelW = ctx.measureText(label).width + padding * 2;
    const labelH = 18;
    const labelX = Math.max(2, Math.min(bx + 2, cw - labelW - 2));
    const labelY = Math.max(2, Math.min(by + 2, areaH - labelH - 2));
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(labelX, labelY, labelW, labelH);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillText(label, labelX + padding, labelY + labelH / 2);
}

// 浏览器内核解不了的编码（典型是 HEVC/H.265）既不会触发 loadedmetadata，
// 也不会触发 error —— <video> 就那么一直挂着，节点看起来卡在「读取…」。
// 所以给一个超时兜底：到点还没元数据，就向 /proyaya/frame 要一帧静图。
const FRAME_FALLBACK_MS = 2500;

function frameUrl(value) {
    const cleaned = String(value || "").replace(/\s*\[(input|output|temp)\]\s*$/, "");
    const norm = cleaned.replace(/\\/g, "/");
    const cut = norm.lastIndexOf("/");
    const filename = cut >= 0 ? norm.slice(cut + 1) : norm;
    const subfolder = cut >= 0 ? norm.slice(0, cut) : "";
    const query = new URLSearchParams({ filename, type: "input" });
    if (subfolder) query.set("subfolder", subfolder);
    const path = "/proyaya/frame?" + query.toString();
    return api?.apiURL ? api.apiURL(path) : path;
}

// 预览视频源：后端按编码判断，能直解就发原文件，HEVC 之类转一份 H.264 代理。
// 这样播放/拖动时间轴在浏览器里都能用，而不是只有一张静帧。
function proxyVideoUrl(value) {
    const cleaned = String(value || "").replace(/\s*\[(input|output|temp)\]\s*$/, "");
    const norm = cleaned.replace(/\\/g, "/");
    const cut = norm.lastIndexOf("/");
    const filename = cut >= 0 ? norm.slice(cut + 1) : norm;
    const subfolder = cut >= 0 ? norm.slice(0, cut) : "";
    const query = new URLSearchParams({ filename, type: "input" });
    if (subfolder) query.set("subfolder", subfolder);
    const path = "/proyaya/preview_video?" + query.toString();
    return api?.apiURL ? api.apiURL(path) : path;
}

function clearFrameFallback(node) {
    const state = node.__cropPreview;
    if (!state) return;
    if (state.frameTimer) {
        clearTimeout(state.frameTimer);
        state.frameTimer = 0;
    }
}

function requestStillFrame(node, value) {
    const state = node.__cropPreview;
    if (!state) return;
    if (state.framePending) return;
    clearFrameFallback(node);
    state.framePending = true;
    const image = new Image();
    image.onload = () => {
        // 期间用户可能又换了文件，换了就丢弃这次结果
        const current = findWidget(node, "file")?.value ?? "";
        if (String(current) !== String(value)) {
            state.framePending = false;
            return;
        }
        state.image = image;
        state.frozen = true;
        state.framePending = false;
        state.frozenReason = "still";
        state.srcW = image.naturalWidth || 0;
        state.srcH = image.naturalHeight || 0;
        state.message = "";
        drawNode(node);
    };
    image.onerror = () => {
        const current = findWidget(node, "file")?.value ?? "";
        state.framePending = false;
        if (String(current) !== String(value)) return;
        state.message = "取帧失败，请查看控制台：/proyaya/frame";
        drawNode(node);
    };
    image.src = frameUrl(value);
}

function startFrameFallback(node, value) {
    const state = node.__cropPreview;
    if (!state) return;
    clearFrameFallback(node);
    state.frameTimer = setTimeout(() => {
        state.frameTimer = 0;
        if (state.srcW && state.srcH) return;   // 元数据已经到了，正常路径
        if (state.hadFrames) return;            // 有画面就别兜底
        requestStillFrame(node, value);
    }, FRAME_FALLBACK_MS);
}

// 预览时间轴：跳帧按**源片帧率**换算（与后端 skip_frames 语义一致）。
// 源 fps 由 /proyaya/video_info 提供，拿到前先用 24 兜底、拿到后自动重新定位。
const TIMELINE_FPS = 24;

function skipSeconds(node) {
    const state = node.__cropPreview;
    const raw = Number(findWidget(node, "skip_frames")?.value);
    if (!Number.isFinite(raw) || raw <= 0) return 0;
    const fps = (state && state.srcFps) || TIMELINE_FPS;
    return raw / fps;
}

/** 拉一次源片帧率；跳帧非 0 时拿到真实 fps 后要重新定位播放头。 */
function fetchSourceFps(node, value) {
    const state = node.__cropPreview;
    if (!state) return;
    const cleaned = String(value || "").replace(/\s*\[(input|output|temp)\]\s*$/, "");
    const norm = cleaned.replace(/\\/g, "/");
    const cut = norm.lastIndexOf("/");
    const filename = cut >= 0 ? norm.slice(cut + 1) : norm;
    const subfolder = cut >= 0 ? norm.slice(0, cut) : "";
    const query = new URLSearchParams({ filename, type: "input" });
    if (subfolder) query.set("subfolder", subfolder);
    const path = "/proyaya/video_info?" + query.toString();
    const url = api?.apiURL ? api.apiURL(path) : path;
    fetch(url).then((r) => (r.ok ? r.json() : null)).then((info) => {
        // 期间换了文件就丢弃
        if ((findWidget(node, "file")?.value ?? "") !== String(value)) return;
        const fps = info && Number(info.fps);
        state.srcFps = Number.isFinite(fps) && fps > 0 ? fps : TIMELINE_FPS;
        if (Number(findWidget(node, "skip_frames")?.value) > 0) applySkip(node);
    }).catch(() => {});
}

/** 把播放器定位到「跳过前 N 帧」的起点；进度条随 seeked 事件自动同步。 */
function applySkip(node) {
    const state = node.__cropPreview;
    if (!state) return;
    const target = skipSeconds(node);
    state.skipSeconds = target;
    // 曾经被静帧兜底冻住时，改跳帧就重新拉一次源：正常能解码的片子要能退回播放态。
    if (state.frozen) {
        refreshSource(node, true);
        return;
    }
    const video = state.video;
    if (!state.srcW) return;   // 元数据没到，等 loadedmetadata 再定位
    if (!Number.isFinite(video.duration) || video.duration <= 0) return;
    const clamped = Math.min(target, Math.max(0, video.duration - 0.05));
    // 加 2ms 余量：浮点落在帧边界上时，有的浏览器会显示前一帧。
    const seekTime = Math.min(clamped + 0.002, Math.max(0, video.duration - 0.05));
    if (Math.abs(video.currentTime - seekTime) > 0.02) {
        // seek 期间 readyState 会短暂掉到 HAVE_METADATA，记一下时间戳，
        // 免得静帧兜底把「正在跳转」误判成「解不出画面」。
        state.seekingAt = Date.now();
        try {
            video.currentTime = seekTime;
        } catch (error) {
            state.seekingAt = 0;
        }
    }
}

function refreshSource(node, force) {
    const state = node.__cropPreview;
    if (!state) return;
    const fileWidget = findWidget(node, "file");
    const value = fileWidget?.value ?? "";
    // 记下这次读到的值。drawNode 每帧跟它比对，所以不管前端用哪条路径改的值，
    // 只要值变了就一定换源 —— 不依赖某个具体回调被触发。
    state.value = value;
    clearFrameFallback(node);
    state.frozen = false;
    state.image = null;
    state.framePending = false;
    state.frozenReason = "";
    state.srcLoadedAt = 0;
    state.hadFrames = false;
    state.seekingAt = 0;
    state.srcFps = null;

    if (!value) {
        state.url = null;
        state.srcW = 0;
        state.srcH = 0;
        state.message = "选择或上传视频后显示画面";
        if (state.progress) state.progress(0);
        if (state.timeEl) state.timeEl.textContent = "0:00 / 0:00";
        try {
            state.video.removeAttribute("src");
            state.video.load();
        } catch (error) {
            /* ignore */
        }
        drawNode(node);
        return;
    }

    const url = proxyVideoUrl(value);
    if (!force && url === state.url) {
        drawNode(node);
        return;
    }
    state.url = url;
    state.srcW = 0;
    state.srcH = 0;
    state.message = "读取 " + value + "…";
    state.video.src = url;
    state.srcLoadedAt = Date.now();
    startFrameFallback(node, value);
    fetchSourceFps(node, value);
    drawNode(node);
}

function watch(node) {
    for (const name of ["width", "height", "crop", "single_side", "target_mp", "skip_frames"]) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        if (!widget.__proyayaPreviewHooked) {
            widget.__proyayaPreviewHooked = true;
            const originalSetValue = widget.setValue;
            widget.setValue = function () {
                const result = originalSetValue?.apply(this, arguments);
                requestAnimationFrame(() => {
                    drawNode(node);
                    if (name === "skip_frames") applySkip(node);
                });
                return result;
            };
            const originalCallback = widget.callback;
            widget.callback = function () {
                const result = originalCallback?.apply(this, arguments);
                requestAnimationFrame(() => {
                    drawNode(node);
                    if (name === "skip_frames") applySkip(node);
                });
                return result;
            };
        }
    }

    const fileWidget = findWidget(node, "file");
    if (fileWidget && !fileWidget.__proyayaPreviewHooked) {
        fileWidget.__proyayaPreviewHooked = true;
        // setValue 和 callback 都要挂：下拉选择走 callback，上传按钮和载入工作流
        // 可能只改 value 不触发任何一个 —— 那种情况靠 drawNode 里的值比对兜底。
        const originalSetValue = fileWidget.setValue;
        fileWidget.setValue = function () {
            const result = originalSetValue?.apply(this, arguments);
            requestAnimationFrame(() => refreshSource(node, true));
            return result;
        };
        const originalCallback = fileWidget.callback;
        fileWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            requestAnimationFrame(() => refreshSource(node, true));
            return result;
        };
    }
}

/* ---------- 与 ComfyUI 原生上传预览合并成一个播放器 ---------- */

/** 找到 ComfyUI 自己渲染的那个视频预览 widget（含 <video> 的 DOM widget）。 */
function findVideoWidget(node) {
    const state = node.__cropPreview;
    for (const widget of node.widgets || []) {
        const el = widget?.element;
        if (!el || el === state?.container || el === state?.outer) continue;
        if (typeof el.querySelector !== "function") continue;
        if (el.querySelector("video")) return widget;
    }
    return null;
}

function findVideoHost(node) {
    return findVideoWidget(node)?.element || null;
}

/**
 * 把某个 widget 的高度钉死成 PREVIEW_FIXED_H。
 *
 * 前端 arrangeWidgets 里是这么算的：
 *     if (widget.computeSize) { widget.computedHeight = widget.computeSize()[1] + 4; }
 * 而且调用时不传参数，所以这里返回的宽度随便给（只取下标 1），宽度回传原值
 * 是为了兼容 litegraph 自己 computeSize(width) 的调用方式。
 * 原生视频预览报的是「按宽度等比后的视频高度」，竖屏视频一来节点就被撑高 ——
 * 钉住之后节点尺寸与分辨率彻底无关。
 */
function pinWidgetHeight(widget) {
    if (!widget) return;
    widget.computeSize = function (width) {
        return [width || 0, PREVIEW_FIXED_H];
    };
}

/** 把所有 widget 高度重算一遍，让节点高度贴合内容（不再残留上次的分辨率高度）。 */
function fitNodeHeight(node) {
    if (!node || typeof node.computeSize !== "function" || typeof node.setSize !== "function") return;
    let height = 0;
    try {
        height = node.computeSize()[1];
    } catch (error) {
        return;
    }
    if (!height) return;
    const width = Math.max(node.size?.[0] ?? 0, 340);
    if (Math.abs((node.size?.[1] ?? 0) - height) > 1 || (node.size?.[0] ?? 0) < 340) {
        node.setSize([width, height]);
    }
}

/**
 * 把自己这层预览（video + canvas）搬到原生预览的位置上，节点里只剩一个播放器。
 *
 * 不缩放原生 <video> 而是搬自己的，原因有两条：
 *   1. 原生 <video> 带浏览器控件，缩放会把控件一起缩掉；
 *   2. 只有自己画才能做「画面 ∪ 取景框」的并集缩放（框比画面大时缩画面）。
 * 原生 <video> 保留布局（visibility:hidden），但宿主的高度由我们钉成固定值，
 * 免得它反过来决定节点大小。
 */
function tryMerge(node) {
    const state = node.__cropPreview;
    if (!state || state.merged) return false;
    const widget = findVideoWidget(node);
    const host = widget?.element;
    if (!host) return false;

    const natives = host.querySelectorAll ? host.querySelectorAll("video") : [];
    for (const native of natives) {
        native.style.visibility = "hidden";   // 留布局、不绘制，也省一次合成
        native.style.pointerEvents = "none";
    }
    if (typeof getComputedStyle === "function" && getComputedStyle(host).position === "static") {
        host.style.position = "relative";
    }
    // 宿主自身也钉死在固定高度：万一原生 <video> 还有内联尺寸把容器顶大，
    // 这里兜一道，容器永远是 PREVIEW_FIXED_H 高。
    host.style.height = PREVIEW_FIXED_H + "px";
    host.style.overflow = "hidden";
    host.style.boxSizing = "border-box";

    // 关键：把原生预览 widget 的布局高度也钉死，否则节点会跟着视频分辨率长高
    state.hostWidget = widget;
    pinWidgetHeight(widget);

    state.container.style.cssText =
        "position:absolute;left:0;top:0;right:0;bottom:0;width:100%;height:100%;" +
        "background:#0b0b0b;border-radius:6px;overflow:hidden;z-index:1;";
    host.appendChild(state.container);

    state.host = host;
    state.merged = true;

    // 自己那个 DOM widget 收起来，否则节点里会空出一块
    if (state.widget) state.widget.computeSize = () => [0, 0];
    if (state.outer) {
        state.outer.style.display = "none";
        // 收起来的槽位仍然是个带 pointer-events:auto 的透明盒子，会吃掉下方控件的点击
        const slot = state.outer.parentElement;
        if (slot && slot.style) slot.style.pointerEvents = "none";
    }

    node.setDirtyCanvas?.(true, true);
    drawNode(node);
    return true;
}

/** 500ms 一次：换源、尝试合并、宿主被前端重建时搬回去。 */
function guardStill(node) {
    const state = node.__cropPreview;
    if (!state || !state.url || state.frozen || state.framePending) return;
    // 已经吐出过真实画面（loadeddata 触发过）就绝不再用静帧盖它：
    // 跳帧 seek、缓冲、拖动时间轴都会让 readyState 短暂低于 2，那是正常现象。
    if (state.hadFrames) return;
    if (state.seekingAt && Date.now() - state.seekingAt < FRAME_FALLBACK_MS) return;
    if (!state.srcLoadedAt) {
        state.srcLoadedAt = Date.now();
        return;
    }
    if (Date.now() - state.srcLoadedAt < FRAME_FALLBACK_MS) return;
    // readyState >= HAVE_CURRENT_DATA 说明真的有画面可画，别用静帧盖住它。
    // 否则两种坏情况都兜住：元数据都读不到（HEVC 常见），或元数据到了但解不出帧。
    if (state.video.readyState >= 2) return;
    requestStillFrame(node, state.value);
}

function tick(node) {
    const state = node.__cropPreview;
    if (!state) return;
    if ((findWidget(node, "file")?.value ?? "") !== state.value) {
        refreshSource(node, true);
    }
    guardStill(node);
    if (!state.merged) {
        if (tryMerge(node)) fitNodeHeight(node);
        return;
    }
    const widget = findVideoWidget(node);
    if (!widget || widget.element !== state.host) {
        // 原生预览 widget 被前端重建了：清掉合并状态，重来一轮
        state.merged = false;
        if (tryMerge(node)) fitNodeHeight(node);
        return;
    }
    // 每轮都重钉一次：前端若把 computeSize / 宿主样式换回去，下一次 tick 就纠正
    pinWidgetHeight(widget);
    if (state.host) {
        state.host.style.height = PREVIEW_FIXED_H + "px";
        state.host.style.overflow = "hidden";
    }
    if (state.container.parentElement !== state.host) {
        state.host.appendChild(state.container);
    }
}

function build(node) {
    if (node.__cropPreview) {
        refreshSource(node);
        return true;
    }
    if (!findWidget(node, "file")) return false;

    // outer 是交给 litegraph 的 DOM widget 外壳；container 才是真正的预览框。
    // 合并时搬走的是 container，outer 留空收起 —— 两者的 DOM 职责分开才不会打架。
    const outer = document.createElement("div");
    outer.style.cssText = "width:100%;height:" + PREVIEW_FIXED_H + "px;";

    const container = document.createElement("div");
    container.style.cssText =
        "position:relative;width:100%;height:100%;background:#0b0b0b;" +
        "border-radius:6px;overflow:hidden;pointer-events:auto;";

    const video = document.createElement("video");
    video.muted = true;
    // 关掉原生控件：它画在 video 的内容盒上，画面一缩小控件就跟着缩。
    // 控件改由下面那条自己画的 bar 承担，尺寸全是固定像素，不参与任何缩放。
    video.controls = false;
    video.playsInline = true;
    video.preload = "metadata";
    video.className = "proyaya-video";
    video.style.cssText =
        "position:absolute;left:0;top:0;width:0;height:0;object-fit:contain;" +
        "display:block;background:#0b0b0b;pointer-events:auto;";

    // 画布只画取景框，铺满「容器 - 控件条」那块区域
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;";

    ensureStyle();
    const bar = document.createElement("div");
    bar.className = "proyaya-bar";
    bar.style.cssText =
        "position:absolute;left:0;right:0;bottom:0;height:" + BAR_H + "px;" +
        "display:none;align-items:center;gap:10px;padding:0 " + BAR_PAD + "px;" +
        "box-sizing:border-box;background:#0b0b0b;" +
        "border-top:1px solid rgba(255,255,255,0.07);z-index:5;" +
        "font:" + BAR_FONT + "px/1 -apple-system,'Segoe UI',Roboto,sans-serif;" +
        "color:#d8d8d8;user-select:none;pointer-events:auto;";

    const playBtn = document.createElement("div");
    playBtn.style.cssText =
        "flex:0 0 " + BAR_BUTTON + "px;width:" + BAR_BUTTON + "px;height:" + BAR_BUTTON + "px;" +
        "display:flex;align-items:center;justify-content:center;border-radius:5px;" +
        "cursor:pointer;pointer-events:auto;";

    // 图标用 CSS 画，不依赖字体字形 —— 字符 ▶/⏸ 在不同机器上大小差别很大
    const playIcon = document.createElement("div");
    playIcon.style.cssText =
        "width:0;height:0;border-style:solid;" +
        "border-width:7px 0 7px 12px;" +
        "border-color:transparent transparent transparent #eaeaea;margin-left:3px;";

    const pauseIcon = document.createElement("div");
    pauseIcon.style.cssText =
        "display:none;width:12px;height:15px;box-sizing:border-box;" +
        "border-left:4px solid #eaeaea;border-right:4px solid #eaeaea;";

    playBtn.appendChild(playIcon);
    playBtn.appendChild(pauseIcon);

    const timeEl = document.createElement("div");
    timeEl.style.cssText =
        "flex:0 0 auto;min-width:92px;text-align:center;font-size:" + BAR_FONT + "px;" +
        "font-variant-numeric:tabular-nums;opacity:0.85;white-space:nowrap;" +
        "pointer-events:auto;";
    timeEl.textContent = "0:00 / 0:00";

    // 自绘时间轴：不用 <input type=range>。
    // 原生滑块靠 UA 的拖拽手势，只要上层有一次 preventDefault 就整个失灵，
    // 而且命中区域是元素本身、没法在 pointer-events 被继承成 none 时救回来。
    // 自己画轨道 + 滑块，用 pointer capture 接管拖拽，行为可控。
    const track = document.createElement("div");
    track.style.cssText =
        "flex:1 1 auto;min-width:0;height:" + BAR_TRACK_H + "px;position:relative;" +
        "cursor:pointer;touch-action:none;pointer-events:auto;";

    // 轨道线、填充、圆球必须共用同一个垂直基准：都以 track 的 50% 为中心线，
    // 各自减去自身高度的一半。之前 rail 只写了 position:absolute 没写 top，
    // 会落到 static position（顶部），圆球按 50% 居中 —— 于是球在线的下面。
    const rail = document.createElement("div");
    rail.style.position = "absolute";
    rail.style.left = "0";
    rail.style.right = "0";
    rail.style.top = "50%";
    rail.style.marginTop = (-BAR_RAIL_H / 2) + "px";
    rail.style.height = BAR_RAIL_H + "px";
    rail.style.borderRadius = (BAR_RAIL_H / 2) + "px";
    rail.style.background = "rgba(255,255,255,0.22)";
    rail.style.pointerEvents = "none";

    const fill = document.createElement("div");
    fill.style.position = "absolute";
    fill.style.left = "0";
    fill.style.top = "50%";
    fill.style.marginTop = (-BAR_RAIL_H / 2) + "px";
    fill.style.width = "0%";
    fill.style.height = BAR_RAIL_H + "px";
    fill.style.borderRadius = (BAR_RAIL_H / 2) + "px";
    fill.style.background = "rgba(255,255,255,0.85)";
    fill.style.pointerEvents = "none";

    const thumb = document.createElement("div");
    thumb.style.position = "absolute";
    thumb.style.left = "0";
    thumb.style.top = "50%";
    thumb.style.marginTop = (-BAR_THUMB / 2) + "px";
    thumb.style.marginLeft = (-BAR_THUMB / 2) + "px";
    thumb.style.width = BAR_THUMB + "px";
    thumb.style.height = BAR_THUMB + "px";
    thumb.style.borderRadius = "50%";
    thumb.style.background = "#fff";
    thumb.style.boxShadow = "0 0 3px rgba(0,0,0,0.7)";
    thumb.style.pointerEvents = "none";

    track.appendChild(rail);
    track.appendChild(fill);
    track.appendChild(thumb);

    bar.appendChild(playBtn);
    bar.appendChild(timeEl);
    bar.appendChild(track);

    container.appendChild(video);
    container.appendChild(canvas);
    container.appendChild(bar);
    outer.appendChild(container);

    /* ---- 自绘播放条的行为 ---- */

    const fmtTime = (t) => {
        const v = Number.isFinite(t) && t > 0 ? t : 0;
        const m = Math.floor(v / 60);
        const s = Math.floor(v % 60);
        return m + ":" + (s < 10 ? "0" + s : String(s));
    };

    let scrubbing = false;

    const setProgress = (ratio) => {
        const r = Math.max(0, Math.min(1, Number.isFinite(ratio) ? ratio : 0));
        const pct = (r * 100).toFixed(2) + "%";
        fill.style.width = pct;
        thumb.style.left = pct;
    };

    const setPlayingIcon = (playing) => {
        playIcon.style.display = playing ? "none" : "block";
        pauseIcon.style.display = playing ? "block" : "none";
    };

    const syncBar = () => {
        const dur = Number.isFinite(video.duration) ? video.duration : 0;
        if (!scrubbing) setProgress(dur > 0 ? video.currentTime / dur : 0);
        timeEl.textContent = fmtTime(video.currentTime) + " / " + fmtTime(dur);
        setPlayingIcon(!video.paused);
    };

    const seekTo = (clientX) => {
        const rect = typeof track.getBoundingClientRect === "function"
            ? track.getBoundingClientRect()
            : { left: 0, width: 1 };
        const width = rect.width || 1;
        const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / width));
        const dur = Number.isFinite(video.duration) ? video.duration : 0;
        if (dur > 0) video.currentTime = ratio * dur;
        setProgress(ratio);
        timeEl.textContent = fmtTime(ratio * dur) + " / " + fmtTime(dur);
    };

    const togglePlay = () => {
        if (!video.getAttribute("src")) return;
        if (video.paused) {
            const p = video.play();
            if (p && typeof p.catch === "function") p.catch(() => {});
        } else {
            video.pause();
        }
        requestAnimationFrame(syncBar);
    };

    // 不让点击穿透到 litegraph，否则拖时间轴会变成拖节点。
    // 同时打一个标记：container 上的兜底逻辑看到标记就不再重复处理。
    const swallow = (event) => {
        event.__proyayaHandled = true;
        event.stopPropagation();
    };
    for (const el of [bar, video]) {
        el.addEventListener("pointerdown", swallow);
        el.addEventListener("mousedown", swallow);
    }
    playBtn.addEventListener("click", togglePlay);
    video.addEventListener("click", togglePlay);

    const beginScrub = (event, captureEl) => {
        scrubbing = true;
        if (captureEl && typeof captureEl.setPointerCapture === "function" &&
            event.pointerId !== undefined) {
            try {
                captureEl.setPointerCapture(event.pointerId);
            } catch (error) {
                /* 不支持就退化成普通拖拽 */
            }
        }
        seekTo(event.clientX);
    };

    track.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.__proyayaHandled = true;
        beginScrub(event, track);
    });

    const onMove = (event) => {
        if (scrubbing) seekTo(event.clientX);
    };
    track.addEventListener("pointermove", onMove);
    container.addEventListener("pointermove", onMove);

    const endScrub = () => {
        if (!scrubbing) return;
        scrubbing = false;
        syncBar();
    };
    for (const el of [track, container]) {
        for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) {
            el.addEventListener(ev, endScrub);
        }
    }

    /**
     * 兜底：ComfyUI 的预览容器普遍是 pointer-events:none，只给 <video> 恢复 auto。
     * 如果哪一级祖先把控件条也变成了 none，bar 自己就收不到事件（现象：画面能点、
     * 控件条点不动）。这时事件会落在 container 自己身上 —— 按坐标补一次处理。
     */
    const inRect = (el, x, y) => {
        if (!el || typeof el.getBoundingClientRect !== "function") return false;
        const r = el.getBoundingClientRect();
        return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    };
    container.addEventListener("pointerdown", (event) => {
        if (event.__proyayaHandled) return;
        if (typeof container.getBoundingClientRect !== "function") return;
        const rect = container.getBoundingClientRect();
        const y = event.clientY - rect.top;
        if (!(y >= rect.height - BAR_H && y <= rect.height)) return;
        event.stopPropagation();
        event.__proyayaHandled = true;
        if (inRect(playBtn, event.clientX, event.clientY)) {
            togglePlay();
            return;
        }
        if (typeof event.preventDefault === "function") event.preventDefault();
        beginScrub(event, container);
    });

    for (const ev of ["timeupdate", "play", "pause", "durationchange",
        "loadedmetadata", "seeked"]) {
        video.addEventListener(ev, syncBar);
    }

    const previewWidget = node.addDOMWidget("scale_preview", "scale_preview", outer, {
        serialize: false,
        hideOnZoom: false,
    });
    if (previewWidget) {
        // 固定高度：不随视频分辨率变化。前端取 computeSize()[1] + 4 作为
        // computedHeight，所以这里给 400 就得到 404 的占位、400 高的元素。
        previewWidget.computeSize = function (width) {
            return [width || 0, PREVIEW_FIXED_H];
        };
    }

    node.__cropPreview = {
        outer, container, video, canvas, bar, track, rail, fill, thumb,
        playBtn, playIcon, pauseIcon, timeEl, widget: previewWidget,
        progress: setProgress,
        srcW: 0, srcH: 0, url: null, message: "", value: undefined,
        frozen: false, image: null, frameTimer: 0,
        framePending: false, frozenReason: "", srcLoadedAt: 0,
        skipSeconds: 0, hadFrames: false, seekingAt: 0, srcFps: null,
        merged: false, host: null,
    };

    // 兜底轮询：上传按钮那条路径可能既不触发 setValue 也不触发 callback，
    // 画布也不一定马上重绘（onDrawForeground 要等下一次渲染）。500ms 比一次字符串，
    // 只在本节点上跑，开销可以忽略。
    node.__cropPreview.timer = setInterval(() => tick(node), 500);

    video.addEventListener("loadedmetadata", () => {
        clearFrameFallback(node);
        node.__cropPreview.frozen = false;
        node.__cropPreview.image = null;
        node.__cropPreview.srcW = video.videoWidth;
        node.__cropPreview.srcH = video.videoHeight;
        // 不自动播放，但要让首帧显示出来：seek 到极小的一个时间点会触发解码。
        if (video.currentTime === 0) {
            try {
                video.currentTime = 0.001;
            } catch (error) {
                /* 有些容器不支持微小 seek，忽略 */
            }
        }
        // 跳帧参数非 0 时把播放头定位到跳过后的起点，进度条随之同步。
        applySkip(node);
        drawNode(node);
    });
    video.addEventListener("loadeddata", () => {
        // 真的有画面了：记下来，同时把可能的静帧兜底撤掉（换源/跳帧后自愈）。
        const st = node.__cropPreview;
        st.hadFrames = true;
        st.seekingAt = 0;
        if (st.frozen) {
            st.frozen = false;
            st.image = null;
            st.frozenReason = "";
            st.srcW = video.videoWidth || st.srcW;
            st.srcH = video.videoHeight || st.srcH;
        }
        drawNode(node);
    });
    video.addEventListener("seeked", () => {
        node.__cropPreview.seekingAt = 0;
        drawNode(node);
    });
    video.addEventListener("error", () => {
        // 编码不受支持时走静帧兜底，而不是把节点卡成一句「无法加载」。
        const value = findWidget(node, "file")?.value || "";
        node.__cropPreview.srcW = 0;
        node.__cropPreview.srcH = 0;
        node.__cropPreview.message = "读取 " + value + "…";
        drawNode(node);
        if (value) requestStillFrame(node, value);
    });

    if (typeof ResizeObserver !== "undefined") {
        node.__cropPreview.observer = new ResizeObserver(() => drawNode(node));
        node.__cropPreview.observer.observe(container);
    }

    watch(node);
    refreshSource(node, true);
    requestAnimationFrame(() => {
        drawNode(node);
        tryMerge(node);
        fitNodeHeight(node);
    });
    return true;
}

app.registerExtension({
    name: "proyaya.videoCropPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            if (!build(node)) {
                let tries = 0;
                const tick = () => {
                    if (build(node) || ++tries > 30) return;
                    requestAnimationFrame(tick);
                };
                requestAnimationFrame(tick);
            }
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            const node = this;
            requestAnimationFrame(() => {
                watch(node);
                refreshSource(node, true);
                const state = node.__cropPreview;
                if (state) {
                    // 载入工作流时节点尺寸是存档里的旧值（可能被上一个分辨率撑大过），
                    // 重新合并一遍把高度钉死，再贴合内容。
                    state.merged = false;
                    tryMerge(node);
                    fitNodeHeight(node);
                }
            });
            return result;
        };

        // 心跳兜底：画布每次重绘时比一次文件名。上传按钮那条路径可能不改
        // widget.value、也不走 setValue/callback，只有靠这个才能补上。
        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const result = originalOnDrawForeground ? originalOnDrawForeground.apply(this, arguments) : undefined;
            if (this.__cropPreview &&
                (findWidget(this, "file")?.value ?? "") !== this.__cropPreview.value) {
                refreshSource(this, true);
            }
            return result;
        };

        const originalOnResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
            requestAnimationFrame(() => drawNode(this));
            return result;
        };

        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const state = this.__cropPreview;
            if (state) {
                state.observer?.disconnect();
                clearInterval(state.timer);
                try {
                    state.video.pause();
                    state.video.removeAttribute("src");
                    state.video.load();
                } catch (error) {
                    /* ignore */
                }
                this.__cropPreview = null;
            }
            const result = originalOnRemoved ? originalOnRemoved.apply(this, arguments) : undefined;
            return result;
        };
    },
});
