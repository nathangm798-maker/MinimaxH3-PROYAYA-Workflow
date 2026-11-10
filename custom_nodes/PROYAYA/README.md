# PROYAYA 3.3.0

H3 长视频人物替换的 ComfyUI 自定义节点。节点 ID = 官方原始节点名 + 前缀 `PROYY`,类别 `PROYAYA/H3`。
显示名统一成品牌格式 `۞▷X◁۞`(例:`۞▷Reference to Video◁۞`),中英文界面一致,不随语言切换。

| 节点 ID | 对应官方节点 | 显示名 | 作用 |
|---|---|---|---|
| `PROYYMiniMaxH3ReferenceToVideo` | MiniMaxH3ReferenceToVideo | ۞▷Reference to Video◁۞ | 分窗规划 + 逐窗 CLIP/参考编码 + 可选主视频初始 latent |
| `PROYYMaskedSampler` | SamplerCustomAdvanced | ۞▷Sampler PRO◁۞ | 全局 CPU latent 联合分窗采样;接入遮罩即重绘模式,不接则等同原生窗口采样 |
| `PROYYImageScale` | ImageScale | ۞▷Resize Video◁۞ | 节点内上传视频,记录缩放配方(分窗读取时才缩放),并输出源视频音频与目标视频宽高 |
| `PROYYTiledLatentUpscale` | MinimaxH3LatentUpscaler3D | ۞▷Latent Upscale◁۞ | 空间分块放大:在第三方放大器外面套 tile,重叠区线性融合,块数与峰值显存由「分块尺寸(px)」决定 |
| `PROYYPreviewAny` | PreviewAny | ۞▷Preview Any◁۞ | 只读等宽文本预览(plan_report / run_report) |
| `PROYAYA_YoloPersonMask` | — | ۞▷YOLO Mask◁۞ | 逐帧 YOLO 人物遮罩 |
| `PROYAYA_ControlVideoCheck` | — | ۞▷Control Video Check◁۞ | 校验 ControlNet 控制视频的 24fps 与最小帧数 |
| `PROYYRefModInject` | — | ۞▷Refmod Inject◁۞ | 把 RefMod 视觉 latent 注入每个窗口的正面条件 |
| `PROYYRefModFrames` | — | ۞▷Refmod Frames◁۞ | 把 RefMod 视觉 latent 逐帧解码成图片序列 |

## 依赖与版本要求

| 项 | 要求 | 不满足的后果 |
|---|---|---|
| **ComfyUI** | **>= 0.30.0**；四份发布工作流按 **0.39.0** 测试 | 低于 0.30.0 时 H3 节点不可用；低于 0.39.0 不保证四份工作流兼容 |
| **ComfyUI-KJNodes** | 当前工作流需要 | 提供 H3 Sage Attention Patch 节点 |
| **ComfyUI-VideoHelperSuite** | 当前工作流需要 | 提供 VHS 视频输出节点 |
| Python | 跟随 ComfyUI(>= 3.10) | 代码里没有 3.10+ 专属语法,不额外约束 |
| torch / numpy / av / safetensors / opencv | 跟随 ComfyUI | 不在这里声明 —— 便携版上重复声明容易让 pip 去「升级」torch/numpy,把能跑的环境搞坏 |
| `ultralytics` | `>= 8.0.0`,**仅** YOLO Mask 需要 | 只有 `۞▷YOLO Mask◁۞` 不可用,其余节点正常 |

启动时控制台会打一行摘要,排障第一时间看它:

```
[PROYAYA] v3.2.0 已加载 9 个节点（ComfyUI 0.39.0）。
```

环境不对时(`compat.py` 自检 + `loader.py` 按块导入)会分两条说清楚:先 ERROR 说**原因**(缺什么、需要哪个版本),再 WARNING 说**后果**(于是哪几个节点没注册上),而不是只留一句 `IMPORT FAILED`:

```
ERROR [PROYAYA] ComfyUI 0.24.3 版本过低，本包需要 >= 0.30.0（MiniMax-H3 支持是 0.30.0 才加入的）。请升级 ComfyUI 后重启。
ERROR [PROYAYA] 依赖缺失，相关节点将无法注册：
  · 缺少模块 comfy_extras.nodes_minimax_h3（MiniMax-H3 参考生视频节点，需要 ComfyUI >= 0.30.0）：模块不存在
  修法：ComfyUI 版本不够就升级本体；模块本身是 pip 包的就补装。
WARNING [PROYAYA] 以下 1 个节点导入失败，未注册：
  · Reference to Video (PROYYMiniMaxH3ReferenceToVideo)：ModuleNotFoundError: No module named 'comfy_extras.nodes_minimax_h3'
  其余 8 个节点照常加载。
```

可选依赖缺失不影响启动,只降级单个节点:

```
WARNING [PROYAYA] YOLO Mask 不可用：缺 ultralytics
```

便携版安装:`python_embeded\python.exe -m pip install "ultralytics>=8.0.0"`。
装完最好对一下版本,确认 torch / numpy 没被顺手改掉:

```
python_embeded\python.exe -m pip list | findstr /I "torch numpy"
```

## 默认值
文件型输入(`PROYYImageScale` 的视频、`PROYYRefModInject` / `PROYYRefModFrames` 的 RefMod)
下拉框第一项固定是空字符串,代表「不选择」——ComfyUI 的组合框默认取第一项,所以新建节点
不会自动挑 `input` 目录里的第一个文件。空着排队会被校验拦住并给出明确提示,不会跑到一半才报错。

## 分段规则
- `length` 为目标帧数,实际规划使用 `min(length, 主视频实际帧数)`,不会复制尾帧去凑段。
  例:length=1000,主视频 260 帧,124 帧/段,阈值 40 → 2 段,第三段尾帧丢弃。
- `tail_keep_threshold`:剩余尾帧 >= 该值则补成一窗(解码后裁掉补帧),< 该值则丢弃。
- 目标帧数超过实际帧数时,`plan_report` 顶部以 `!!!` / `⚠⚠⚠` 醒目警告,不截断生成;启用 `preview_only` 时只出预览不生成。
- `output_frames` 是真实保留的帧数,下游应以它为准。

## 参考编码缓存

`ref_cache`(默认开) / `ref_cache_mb`(默认 1024，0 关闭) 把参考编码拆成两层复用：

| 层 | 内容 | 复用条件 |
|---|---|---|
| L1 | 参考图 / 辅助视频 / 独立参考音频 / `video_encode` 初始 latent 的 VAE 编码 | 来源对象、尺寸、帧数不变 |
| L2 | 整窗的正面条件 + 空 latent | 提示词、三个增益、窗口 `start`/`frames`、宽高、全部素材身份都不变 |

- 只改 `length`：已有窗口的 L2 直接命中，只编码新增窗口(均匀分段时前面的窗口起点不变)。
- 只改提示词：文本+视觉仍需重跑(Qwen3-VL 32B 是主要开销)，但 L1 命中，参考素材不再进 VAE。
- 空文本负面条件按 CLIP 对象缓存，改时长时不再重跑那一次前向。

key 用**来源身份**而不是内容哈希：124 帧 1344×768 的内容哈希比它省下的编码更贵。视频用
`video_info()` 的 `fingerprint`(路径 + mtime + size)加窗口起点；图片/音频用张量对象身份，
并由缓存持引用，避免 `id` 被回收后串到别的素材上。参考图的编码与输出画布无关，改 `width`/`height`
只让 L2 失效，不会重编参考图。

`plan_report` 末尾会打印本次的命中/未命中与占用。缓存跨执行保留，只受 `ref_cache_mb` 约束；
返回结果与关闭缓存时逐位一致。

## 迁移
节点 ID 已更改,旧画布需重新导入新版工作流 JSON。旧包 `ComfyUI-H3-Continuity-V11`、`ComfyUI-H3-Window-Sampler` 不能与本包同时安装。

## 多语言
参数名、输出槽名随 ComfyUI 语言设置切换(Settings → 语言 / Locale),不需要改工作流。
节点显示名是品牌名 `۞▷X◁۞`,中英一致,刻意不参与翻译。

- 翻译放在 `locales/<lang>/nodeDefs.json`,由 ComfyUI 服务端 `/i18n` 接口合并后下发给前端;
  前端按 `nodeDefs.<节点类型>.inputs.<参数名>.name` 查表,查不到就回落到 Python 里写的显示名。
- 现有 `en` / `zh` 两份,新增语言只需加一个目录(语言码见 ComfyUI 前端 `locales/`:zh-TW、ja、ko、ru 等)。
- 语言包在服务启动时读入并缓存,改完需要重启 ComfyUI。
- 发布工作流不保存自定义 title，节点会使用 ComfyUI 和节点包的默认显示名。

## 安装与工作流使用

将本目录复制到：

ComfyUI/custom_nodes/PROYAYA

当前发布包还依赖以下两个 ComfyUI 自定义节点包：

- ComfyUI-KJNodes
- ComfyUI-VideoHelperSuite

YOLO 版本额外需要：

python_embeded\python.exe -m pip install "ultralytics>=8.0.0"

工作流默认使用的模型文件如下。文件名和子目录应与工作流一致：

models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors

models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

models/vae/minimax_h3_video_vae_int8_convrot (2).safetensors

models/loras/minimax_h3/h3_character_swap_v1.5.safetensors

models/loras/minimax_h3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors

models/model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors

models/latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors

models/ultralytics/segm/person_yolov8m-seg.pt

将源视频、ControlNet 控制视频和参考图片放入 ComfyUI/input/，然后导入 workflows/ 下对应的 JSON。
如果节点中没有找到默认文件，请在节点下拉框中重新选择自己的素材。

### 四份工作流

| 工作流 | 用途 | 关键要求 |
|---|---|---|
| 基础工作流 | 长视频人物替换主流程 | 源视频通过 <Video 1> 参考通道参与生成；video_encode=false 表示不使用源片初始 latent |
| YOLO 遮罩版 | 在人物区域重绘，背景区域保留源片 | video_encode=true；遮罩白色为重绘区域；当前遮罩取景为 440×768，不保证像素级对齐 |
| ControlNet 版 | 用深度/结构视频增强动作和构图控制 | 控制视频必须为 24fps，且帧数不少于目标输出帧数；不足时会被校验节点直接拒绝 |
| 二采版 | latent 放大后进行第二次采样，增强细节 | 二采 video_encode=false；复用一采 windows；音频从一采 latent 锁定保留 |

默认目标时长为 13 秒，对应：

328 帧 = 13.6667 秒 @ 24fps

修改目标秒数时只修改 PrimitiveInt 节点。总帧数会由公式节点同步传给窗口长度、读取帧数、YOLO 帧数和 ControlNet 校验节点。

ControlNet 版本的控制视频必须与主视频使用相同起点和空间取景。当前示例控制视频为 328 帧、24fps；如果改动目标时长，需要同步重新生成或替换控制视频。

### 工作流构建脚本

四份用户工作流仍然独立保存，便于直接导入。公共清理规则和 ControlNet 校验节点由以下脚本统一维护：

tools/build_workflows.py

使用方式：

python tools/build_workflows.py --input-dir path/to/workflows

脚本会统一标题、输出前缀、模型元数据、24fps 控制视频校验连接，并清理旧版 widgets_values_named 与 VHS 预览元数据。

当前发布版包含 9 个节点；如果 H3 依赖缺失，Control Video Check、Resize Video、Sampler PRO、YOLO Mask 等非 H3 节点仍按导入块独立加载。

## 测试
`python -B -m unittest discover -s tests`(CPU,无需模型/视频)。

## 许可
rolling 思路来自 MIT 协议的 user2318/ComfyUI-CustomNodeKit,见 `LICENSE.CustomNodeKit`、`ROLLING_HISTORY.md`、`THIRD_PARTY_NOTICES.md`。
