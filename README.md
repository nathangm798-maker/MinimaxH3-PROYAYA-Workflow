# PROYAYA · MiniMax-H3 换人 / 二采工作流

**PROYAYA** — 一套基于 MiniMax-H3 视频模型的 ComfyUI 工作流：把任意真人视频里的表演者，逐帧替换成指定的照片级真实人物，再用「二采（latent 2× 放大 + 重采样）」把成片画质提到 768×1344。

整套方案针对 **长视频 + 小显存（16 GB 级）** 做过实测优化：分段窗口采样、流式 VAE 解码（内存常数级，或零缓冲直出 mp4）、参考编码跨运行复用。

---

## 目录

```
custom_nodes/PROYAYA/        PROYAYA 自定义节点包（MIT，全部节点源码）
workflows/
  PRO-基础版.json      一采：换人采样（384×672），输出 joint AV latent
  PRO-二采版.json      二采：latent 2× 放大（768×1344）后接力重采样
examples/
  reference/ref_picture.png     参考人物（换人目标形象）
  reference/source_720x1280.mp4 源视频（动作/构图/机位来源）
  results/pass1_384x672.mp4     一采成品
  results/pass2_768x1344.mp4    二采成品
docs/审查报告.md               两版工作流参数审查 + 节点文案问题清单
```

## 示例素材规格

| 文件 | 规格 |
|---|---|
| `source_720x1280.mp4` | 720×1280 / HEVC / 269 帧 / 8.99 s / 含 AAC 音轨 |
| `ref_picture.png` | 参考人物立绘（换人目标） |
| `pass1_384x672.mp4` | 384×672 / 24 fps / 124 帧 / H.264 crf19 |
| `pass2_768x1344.mp4` | 768×1344 / 24 fps / 124 帧 / H.264 crf19 |

## 两采的分工

**一采（PRO-基础版）**
- 模型：`viggle_remake_swap` + pika 加速 LoRA + `h3_character_swap_v1.5`
- 采样：euler，ManualSigmas `1, 0.9730, 0.9231, 0.8000, 0.571429, 0.333333, 0.0`
- 输出：384×672 的 joint AV latent（视频 + 音频分开 SaveLatent）

**二采（PRO-二采版）**
- latent 走 `Latent Upscale` 2×（768×1344），音频分支原样透传
- 模型：`fl2va` + pika + swap LoRA，euler/lcm，sigmas 从 `0.9231` 接力
- 参考视频 / 参考图 / 提示词与一采**完全相同**（只有 latent 与画布放大）
- 输出：768×1344，再接 `Stream VAE Decode` 出片

## 关键节点（PROYAYA 自定义包）

| 节点 | 作用 |
|---|---|
| `۞▷Resize Video◁۞` | 取帧/缩放/裁切，输出参考视频 + 音频 + 生成尺寸 |
| `۞▷Reference to Window◁۞` | 窗计划 + 每窗参考编码（conditioning），二采时复用一采 plan |
| `۞▷Window Sampler◁۞` | 按窗 masked 采样，支持 rolling / CFG 分段 |
| `۞▷Latent Upscale◁۞` | 3D latent 分块 2× 放大 |
| `۞▷Stream VAE Decode◁۞` | 流式解码：IMAGE 序列（可落盘 memmap）或 **MP4 直写（零缓冲）** |

### Stream VAE Decode 的两种输出模式

| 模式 | 内存 | 磁盘 | 说明 |
|---|---|---|---|
| `IMAGE序列(接VHS)` | 缓冲 ≤ `内存上限(GB)` 时全量驻内存，否则落 temp memmap | 最多 帧数×宽×高×12 B（768p 全长 13129 帧 = 151.5 GiB） | 可继续接 VHS 做音频合成/转码 |
| `MP4直写(零缓冲)` | **恒定 ~2 GB** | **0** | 解码 chunk 直接管道喂 ffmpeg，节点自身即输出节点，产物带视频预览卡 |

实测：768×1344 / 13129 帧，MP4 直写 20.8 分钟出片，与常规解码逐位一致（PSNR = inf）。

## 使用步骤

1. 把 `custom_nodes/PROYAYA` 整个文件夹拷进 `ComfyUI/custom_nodes/`（本仓库已含全部源码，MIT 许可）；
   模型放入 `models/` 对应目录（UNet / LoRA / CLIP / 视频 VAE / 音频 VAE / latent upscaler）。
2. 把 `examples/reference/` 里的两张素材拷进 `ComfyUI/input/`。
3. 载入 `workflows/PRO-基础版.json`，确认参考视频/参考图/提示词，跑一采 → 得到 `latents/PROYAYA_pass1_{video,audio}`。
4. 载入 `workflows/PRO-二采版.json`，指向一采 latent，跑二采。
5. 末端 `Stream VAE Decode` 选 `MP4直写(零缓冲)`，填 `文件名前缀`，直接出片。

## 许可

示例素材（参考图 / 源视频 / 成片）仅供工作流演示，不得用于商业或真人侵权用途。工作流 JSON 可自由修改分发。
