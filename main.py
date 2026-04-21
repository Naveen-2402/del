import sys
import collections
import os
import cv2
import torch
from tqdm import tqdm

# --- COMPATIBILITY PATCHES ---
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

def upscale_image(input_path, output_path, model_name='RealESRGAN_x4plus', scale=4):
    # 1. GPU Detection & Info
    if torch.cuda.is_available():
        device = torch.device('cuda:0') # Uses the first V100
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"🚀 Found {gpu_count} GPU(s). Using: {gpu_name}")
        use_half = True 
    else:
        device = torch.device('cpu')
        print("⚠️  GPU not detected by PyTorch. Check your installation.")
        print("🐌 Status: Using CPU.")
        use_half = False

    # 2. Setup Model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    model_path = os.path.join('weights', f'{model_name}.pth')
    
    if not os.path.exists(model_path):
        os.makedirs('weights', exist_ok=True)
        url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
        load_file_from_url(url=url, model_dir='weights', progress=True)

    # 3. Initialize Upsampler
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400, # V100 16GB can handle 400 easily
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device
    )

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Error: {input_path} not found.")
        return

    # 4. Upscale with Progress Bar
    with tqdm(total=100, desc="💎 Enhancing Image") as pbar:
        pbar.update(10)
        output, _ = upsampler.enhance(img, outscale=scale)
        pbar.update(80)
        cv2.imwrite(output_path, output)
        pbar.update(10)
        print(f"\n✅ Done! Saved as {output_path}")

if __name__ == "__main__":
    upscale_image('post.png', 'post_4x.png')
