import sys
import collections

# --- COMPATIBILITY PATCH START ---
# Fix 1: ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
try:
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except ImportError:
    pass

# Fix 2: AttributeError: module 'collections' has no attribute 'Iterable' (For Python 3.10+)
if not hasattr(collections, 'Iterable'):
    import collections.abc
    collections.Iterable = collections.abc.Iterable
# --- COMPATIBILITY PATCH END ---

import os
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

def upscale_image(input_path, output_path, model_name='RealESRGAN_x4plus', scale=4):
    # Setup model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    model_url = f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth'
    model_path = os.path.join('weights', model_name + '.pth')
    
    if not os.path.exists(model_path):
        os.makedirs('weights', exist_ok=True)
        load_file_from_url(url=model_url, model_dir='weights', progress=True, file_name=None)

    # Initialize upsampler
    # Note: if you don't have an NVIDIA GPU, set half=False
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True 
    )

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not find {input_path}")
        return

    print(f"Processing {input_path}...")
    output, _ = upsampler.enhance(img, outscale=scale)
    cv2.imwrite(output_path, output)
    print(f"Finished! Saved as {output_path}")

if __name__ == "__main__":
    upscale_image('post.png', 'post_4x.png')