# H3 V1.1：步内滚动联合采样（初版历史说明）

本文件保留初版说明。主视频初始 latent 和小于 1 的起始 sigma 支持见当前 `README.md`。

本版继续使用 MiniMax H3 / FastH3 模型，把 CustomNodeKit 的 SCAIL2 **步内 x0 rolling** 机制适配到 H3 的 CPU 全局音视频采样器。不是 SCAIL2 模型工作流，也不是完整移植 Wan 专用的锚定系统。

## 使用

1. 保留原 `ComfyUI-H3-Window-Sampler`（V1 的编码器、分段预览等仍由它提供）。
2. 将本目录 `ComfyUI-H3-Continuity-V11` 放入同一个 ComfyUI 的 `custom_nodes`，重启后刷新页面。
3. 导入交付包 `workflows/H3_V1.1_NiceBody_三窗.json`。唯一替换的执行节点是 #40，新类型为 `H3V11NativeWindowSampler`。
4. 默认 `preview_only=true`。先查看三窗输入首尾，再将 #30 的 `preview_only` 关闭以生成视频。
5. #40 `rolling_enabled=false` 或 `rolling_strength=0` 恢复 V1 的逐步融合算法。要完全回退，直接使用 V1 工作流；两个节点包可同时安装，不注册重名节点。

## 信息怎样传递

每个 sigma 内，按窗口时间顺序执行：窗 1 → 把当前 CFG 后 x0 的重叠尾部送给窗 2 → 窗 2 预测 → 把其尾部送给窗 3。传递发生在下一窗前向之前，不用等待下一 sigma。

后窗重叠区的输入采用：

```text
beta_eff = rolling_strength * min(1, 2 * (1 - sigma))
target   = (1 - sigma) * previous_x0 + sigma * noise_scale * global_noise_slice
local_x  = (1 - beta_eff) * local_x + beta_eff * target
```

仅修改当前窗输入的副本；完整全局状态和原始噪声不变。各窗预测仍按 V1 的线性重叠权重融合，然后 Euler 更新一次全局状态。每个 sigma 都清空提交缓存，只保留传给下一窗的重叠尾部。最终输出一份全局 latent，交给原生 VAE 解码，没有三个成片文件的拼接。

沿用 H3 的 124 帧窗、102 帧步长、22 像素帧重叠；视频与音频分别使用原生坐标，对应 7 个视频 latent 时间位置、37 个音频 latent 时间位置。13 秒模板按原公式得到 328 帧，24 FPS 为 13.667 秒。8 步三窗仍为 24 次窗口预测（CFG 内部可能有额外模型前向）。

## 与 donor 的适配差异

来源固定为 `user2318/ComfyUI-CustomNodeKit` commit `5099660055cfe96a26366150f171129fd5578c39` 的 `wan_scail2_rolling.py`，复用重噪混合公式、late 曲线、同 sigma 提交和重叠共写融合思路。

- 默认强度 0.45 来源于 donor；这是 H3 的实验起点，不能引用 donor 的 Wan 画质数字作为 H3 验收。
- H3 使用一次生成的全局 NOISE 的对应切片；donor 按 sigma/全局帧索引生成固定种子噪声。这里避免第二套随机噪声，完整保留 H3 seed 的含义。
- 缓存的是 **CFG 后**预测，兼容 H3 双 CFG；不是 donor `calc_cond_batch` 的单个条件分支。
- H3 音频已经在统一的 packed solver 空间中缩放；不再额外切换 audio sigma 或二次乘除 audio scale。公式包含模型 `noise_scale`。
- 不改 Wan 模型、彩色身份遮罩或 RoPE；不移植 donor 的额外只读前文、锚帧前插、长程记忆组或 `beta_inherited`。这些都依赖 Wan 专用条件/位置机制，本版没有对应输入，因而不展示无效控件。
- 默认启用 rolling 时只接受原生固定网格 Euler；关闭时保留 V1 原有采样器支持。

## 检查与限制

`tests/test_rolling.py` 验证同 sigma 传播、AV 对齐、输入不变、重叠融合与零强度退化；`tests/test_continuity_runtime.py` 使用实际 ComfyUI CPU sampler、替换模型预测，检查集成与生命周期。旧测试作为 V1 路径的回归检查保留。

运行：`python -m unittest discover -s tests -p "test_*.py"`，使用安装 ComfyUI 的 Python。默认 ComfyUI 根目录为 `D:/ComfyUI_windows_portable/ComfyUI`，可用 `COMFYUI_ROOT` 指定。

数值与接口通过不代表生成质量改善。真实 Nice Body 三窗的服装、清晰度、动作与接缝仍需 A/B 成片验证。V1.1 是条件化启发式，会改变采样结果；不是原 ODE 的无损等价变换。原生解码仍返回完整 RGB 批次，不是流式输出。

本包仅注册 V1.1 新采样节点；随包保留的 V1 支持模块用于复现采样路径，不重复注册 V1 节点或加载旧前端扩展。历史说明见 `V1_HISTORY.md`。
