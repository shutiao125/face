import os
import sys
import torch
import gradio as gr
import re
from diffusers import StableDiffusionXLPipeline
from modelscope import snapshot_download
from deep_translator import GoogleTranslator

# 屏蔽潜在的 torchao 报错
sys.modules['torchao'] = None
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 1. 模型配置 =================
# ★ 请把这里换成你刚才在魔搭上传 RealVisXL 的模型库 ID
REALVIS_REPO_ID = "shutiao125/RealVisXL-V4" 
LORA_REPO_ID = "shutiao125/sdxl-mixed-style-lora"
LORA_FILENAME = "yabi_dark_4090_v3.safetensors"

pipe = None

def load_models():
    global pipe
    if pipe is not None:
        return
    
    # 1. 从魔搭拉取你上传的底模
    print(f"⏳ 正在从魔搭拉取底模: {REALVIS_REPO_ID}...")
    realvis_dir = snapshot_download(REALVIS_REPO_ID)
    
    # 自动寻找文件夹里的 .safetensors 文件
    realvis_path = None
    for f in os.listdir(realvis_dir):
        if f.endswith(".safetensors"):
            realvis_path = os.path.join(realvis_dir, f)
            break
            
    if not realvis_path:
        raise FileNotFoundError("未在底模库中找到 safetensors 文件！")

    # 2. 从魔搭拉取 LoRA
    print(f"⏳ 正在拉取 LoRA: {LORA_REPO_ID}...")
    lora_dir = snapshot_download(LORA_REPO_ID)
    lora_path = os.path.join(lora_dir, LORA_FILENAME)

    # 3. 初始化 Pipeline（极限省内存模式）
    print("⏳ 正在初始化 Pipeline...")
    pipe = StableDiffusionXLPipeline.from_single_file(
        realvis_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    
    # 极限省显存/内存策略
    pipe.enable_sequential_cpu_offload() 
    pipe.vae.enable_tiling()
    pipe.enable_attention_slicing()

    # 4. 注入 LoRA
    print("⏳ 正在注入 LoRA 权重...")
    pipe.load_lora_weights(lora_path)
    print("✅ 所有模型加载成功！")


# ================= 2. 翻译与推理逻辑 =================
def translate_to_english(text):
    if not text.strip():
        return ""
    if re.search(r'[\u4e00-\u9fa5]', text):
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text
    return text

def generate_image(chinese_prompt, size_choice):
    global pipe
    if pipe is None:
        load_models()
        
    if "|" in chinese_prompt:
        parts = chinese_prompt.split("|", 1)
        pos_zh, neg_zh = parts[0].strip(), parts[1].strip()
    else:
        pos_zh, neg_zh = chinese_prompt.strip(), ""
    
    pos_en = translate_to_english(pos_zh)
    neg_en = translate_to_english(neg_zh)
    
    full_prompt = f"yabi_dark, real photo, {pos_en}"
    base_negative = "ugly, blurry, poor quality, bad anatomy, cartoon, anime, cgi, plastic skin, overexposed, watermark"
    full_negative_prompt = f"{base_negative}, {neg_en}" if neg_en else base_negative

    size_map = {
        "正方形 1:1 (1024x1024)": (1024, 1024),
        "横版 16:9 (1344x768)": (1344, 768),
        "竖版 9:16 (768x1344)": (768, 1344),
        "横版 4:3 (1152x896)": (1152, 896),
        "竖版 3:4 (896x1152)": (896, 1152)
    }
    w, h = size_map.get(size_choice, (1024, 1024))

    print(f"🎨 开始生成: {full_prompt} | 尺寸: {w}x{h}")
    
    try:
        # 内存极小的环境下，降低步数防止超时或OOM
        image = pipe(
            prompt=full_prompt,
            negative_prompt=full_negative_prompt,
            num_inference_steps=20, 
            guidance_scale=5.0,
            height=h,
            width=w
        ).images[0]
        return image
    except Exception as e:
        print(f"生成报错: {e}")
        # 生成失败返回黑图
        return torch.zeros((512, 512, 3), dtype=torch.uint8).numpy()

# ================= 3. 网页 UI 搭建 =================
with gr.Blocks(title="Yabi Dark图片生成器", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🖼️ Yabi Dark风格图片生成器\n**支持中文输入！** (如果长时间无响应，可能是内存不足)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="你想画什么？(支持中文)", lines=4)
            size_choice = gr.Radio(
                choices=[
                    "正方形 1:1 (1024x1024)",
                    "横版 16:9 (1344x768)",
                    "竖版 9:16 (768x1344)",
                    "横版 4:3 (1152x896)",
                    "竖版 3:4 (896x1152)"
                ],
                value="正方形 1:1 (1024x1024)",
                label="选择图片尺寸"
            )
            generate_btn = gr.Button("✨ 生成图片", variant="primary", size="lg")
        with gr.Column(scale=2):
            output_image = gr.Image(label="生成结果", type="pil", height=500)
    generate_btn.click(fn=generate_image, inputs=[prompt_input, size_choice], outputs=output_image)

# ================= 4. 启动服务 =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
