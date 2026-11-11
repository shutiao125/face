# 🎨 face — AIGC 图像生成项目集

从零实现的扩散模型 + SDXL LoRA 微调与推理服务，两条 AIGC 技术线的完整落地。

```
├── README.md                 # 本文档
├── checkpoint.pt             # DDPM 模型权重（134MB, LFS）
├── generated_grid.png        # DDPM 生成采样图
├── real_grid.png             # 真实数据集采样图
└── face/                     # SDXL LoRA 项目
    ├── yabi_dark_4090_v3.safetensors   # LoRA 权重（163MB, LFS）
    ├── app.py                # Gradio 推理服务（支持中文输入）
    ├── requirements.txt
    └── samples/              # 训练过程采样图
```

---

## Part 1：从零实现的 DDPM 动漫头像生成

> **模型架构**: UNet (base_ch=64) + Cosine β Schedule + EMA　|　**参数量**: 8.77M　|　**分辨率**: 64×64

### 📊 核心评估指标对比

| 指标 | 生成数据 (Generated) | 真实数据 (Real) | 比值 (Gen/Real) | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| **多样性 (Diversity)** | 72.19 | 73.89 | **0.98** | ✅ 优秀 (0.7~1.0) |
| **锐度 (Sharpness)** | 0.1294 | 0.1500 | **0.86** | ✅ 优秀 (0.7~1.2) |

*注：Diversity 越接近真实值越好，过高说明充满噪点，过低说明模式塌缩。Sharpness 过高通常意味着生成图像带有大量像素噪点。*

### 🖼️ 视觉采样展示

| 模型生成 (128 张) | 真实数据集 (128 张) |
| :---: | :---: |
| ![Generated](generated_grid.png) | ![Real](real_grid.png) |

### 📝 训练结论
1. **收敛状态良好**：Loss 稳定在 0.042 左右，模型已达到理论收敛极限。
2. **多样性健康**：生成数据的 Diversity 达到了真实数据的 97.7%，表明模型**没有发生模式塌缩**（Mode Collapse）。
3. **画质平滑**：Sharpness 是真实数据的 86.3%，说明生成的二次元图像边缘清晰、色块纯净，没有高频的噪点伪影。

---

## Part 2：SDXL LoRA 风格微调（yabi_dark）

基于 **RealVisXL V4.0** 底模训练的暗调街拍风格 LoRA，在 RTX 4090 上完成训练。

| 训练配置 | 参数 |
| :--- | :--- |
| 底模 | RealVisXL V4.0（写实系 SDXL） |
| 训练数据 | 30 张暗调街拍图片，触发词 `yabi_dark` + `real photo` |
| 网络结构 | LoRA (dim=32 / alpha=16)，共 2166 个张量，85.1M 参数 |
| 训练配置 | 13 epochs / 1000 steps，batch 4，lr 1e-4 (constant)，AdamW8bit，bf16 |
| 分辨率桶 | 512×1024 ~ 832×1024（ar error 0.014） |

### 🚀 本地运行推理服务

```bash
cd face
pip install -r requirements.txt
python app.py   # 默认 7860 端口，可用 PORT 环境变量覆盖
```

**服务特性**：
- 🇨🇳 **中文输入**：自动调用翻译 API 将中文提示词转为英文
- 📐 **多分辨率**：支持 1:1 / 16:9 / 9:16 / 4:3 / 3:4 五种出图尺寸
- 🧠 **低显存优化**：Sequential CPU Offload + VAE Tiling + Attention Slicing，无 GPU 环境也可运行
- 🛡️ **内置负面词**：自动附加质量负面提示词，用户也可通过 `正面词|负面词` 语法自定义

> 底模与 LoRA 均从魔搭 ModelScope 拉取（`shutiao125/RealVisXL-V4`、`shutiao125/sdxl-mixed-style-lora`），首次启动需下载约 7GB。

---

*Part 1 评估报告由自动化脚本生成；Part 2 训练基于 kohya sd-scripts。*
