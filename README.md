
<div align="center">
<h1> ThinkGen: Generalized Thinking for Visual Generation</h1>
<h3></h3>

Siyu Jiao<sup>1</sup>, Yiheng Lin<sup>1</sup>, Yujie Zhong<sup>2</sup>, Qi She<sup>2</sup>, Wei Zhou<sup>2</sup>, Xiaohan Lan<sup>2</sup>
Zilong Huang<sup>2</sup>, Fei Yu<sup>2</sup>, Yingchen Yu<sup>2</sup>, Yunqing Zhao<sup>2</sup>, Yao Zhao<sup>1</sup>, Yunchao Wei<sup>1</sup>

<sup>1</sup> Beijing Jiaotong University, <sup>2</sup> Bytedance


[![arXiv](https://img.shields.io/badge/arXiv%20paper-2502.20313-b31b1b.svg)](https://github.com/jiaosiyuu/ThinkGen)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-JSYuuu/ThinkGen-yellow)](https://huggingface.co/JSYuuu/ThinkGen)&nbsp;



</div>

<div align="center">
<img src="images/teaser.png" width="958%">
</div>


## 🚀 Quick Start

### 🛠️ Environment Setup

#### ✅ Recommended Setup

```bash
# 1. Clone the repo
git clone https://github.com/jiaosiyuu/ThinkGen.git
cd OmniGen2

# 2. (Optional) Create a clean Python environment
conda create -n thinkgen python=3.11
conda activate thinkgen

# 3. Install dependencies
# 3.1 Install PyTorch (choose correct CUDA version)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3.2 Install other required packages
pip install -r req.txt

# ThinkGen runs even without flash-attn, though we recommend install it for best performance.
pip install  --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation
```

#### 🌏 For users in Mainland China

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu124
pip install -r req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install  --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---


* **Run Locally**:
```bash
from ThinkGen.model import ThinkGen_Chat
import os

model_path = "JSYuuu/ThinkGen"

chat_model = ThinkGen_Chat(
    model_path=model_path,
    dtype='bf16',
    height=1024,
    width=1024
)


# Generation
messages = [
    {"type": "text", "value": '''A close-up image of a red apple with the words 'Tart & Sweet' in white, cursive font on its surface, forming a spiral pattern. The apple is centered in the frame, and the background is a green surface labeled 'Organic Produce' in black, bold letters. The apple has a visible stem and a small bite mark on its side with the word 'Juicy' written in a small, handwritten style near the bite.'''}
]
results = chat_model.generate_image(messages)
output_dir = "vis/chat"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(results.images):
    save_path = os.path.join(output_dir, f"result_{i}.png")
    img.save(save_path)
    print(f"Saved to {save_path}")



# Understanding
messages = [
    {"type": "image", "value": "images/teaser.png"},
    {"type": "text", "value": "Describe this image"}
]

response = chat_model.generate_text(messages)
print(response)


```
