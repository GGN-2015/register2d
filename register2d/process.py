from . import rotate
from . import timer

from tqdm import tqdm
from PIL import Image
import cupy as cp
from cupy_fft_match import match_arr
from cupyx.scipy.ndimage import binary_dilation

from typing import Tuple
import functools

def black_to_red_transparent(img_l):
    # 1. Validate input is in L mode (required by design, but validation adds robustness)
    if img_l.mode != 'L':
        raise ValueError(f"Input image must be in L mode, current mode: {img_l.mode}")
    
    # 2. Convert to NumPy array (shape=(height, width), grayscale values 0-255)
    img_arr = cp.array(img_l, dtype=cp.uint8)
    height, width = img_arr.shape
    
    # 3. Construct RGBA array (shape=(height, width, 4), 4 channels: R, G, B, Alpha)
    # Initialize fully transparent (Alpha=0), RGB channels default to 0
    rgba_arr = cp.zeros((height, width, 4), dtype=cp.uint8)
    
    # 4. Core logic: Black pixels (>128) → Red (R=255, G=0, B=0) + Opaque (Alpha=255)
    # Generate mask for black pixels (True indicates black pixel)
    black_mask = img_arr < 128
    
    # Assign values to black pixels: R=255, Alpha=255 (G and B remain 0)
    rgba_arr[black_mask, 0] = 255  # R channel: Red
    rgba_arr[black_mask, 3] = 192  # Alpha channel: Semi-transparent (255=fully opaque, 0=fully transparent)
    
    # Other pixels remain default: RGB=0, Alpha=0 (transparent), no additional processing needed
    
    # 5. Convert to Pillow Image object in RGBA mode and return
    rgba_img = Image.fromarray(cp.asnumpy(rgba_arr), mode='RGBA')
    return rgba_img

def fft_convolve_1d(vec1, vec2) -> cp.ndarray:
    # 1. Validate inputs are 1-dimensional vectors
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional NumPy vectors")
    
    # 2. Calculate minimum required length (avoid circular convolution affecting linear convolution results)
    len1 = len(vec1)
    len2 = len(vec2)
    min_length = max(len1, len2) * 2 - 1  # Theoretical length of linear convolution
    
    # 3. Zero-padding for both vectors (pad to minimum length to ensure complete convolution results)
    vec1_padded = cp.pad(vec1, (0, min_length - len1), mode='constant')
    vec2_padded = cp.pad(vec2, (0, min_length - len2), mode='constant')
    
    # 4. Core steps: FFT → Frequency domain product → IFFT
    fft1 = cp.fft.fft(vec1_padded)  # Frequency domain representation of vector 1
    fft2 = cp.fft.fft(vec2_padded)  # Frequency domain representation of vector 2
    fft_product = fft1 * fft2       # Frequency domain product (corresponds to time domain convolution)
    conv_result = cp.fft.ifft(fft_product)  # IFFT to restore time domain
    
    # 5. Remove imaginary part error (tiny imaginary components from numerical calculation; actual convolution should be real)
    conv_result = cp.real(conv_result)
    
    return conv_result

__global_obj_image_cache = {}

# Get an image
def get_l_image(path_or_img:Image.Image|str):
    if isinstance(path_or_img, str):
        # Avoid repeated reading
        if __global_obj_image_cache.get(path_or_img) is None:
            img = Image.open(path_or_img).convert("L")
            __global_obj_image_cache[path_or_img] = img
        return __global_obj_image_cache[path_or_img]
    
    elif isinstance(path_or_img, Image.Image):
        return path_or_img.convert("L")

    else:
        assert False

@functools.cache
def get_cp_image_str(FULL_IMAGE_INPUT:str):
    return (cp.array(get_l_image(FULL_IMAGE_INPUT)) / 256).astype(cp.float64)

def get_cp_image(FULL_IMAGE_INPUT:str|Image.Image):
    if isinstance(FULL_IMAGE_INPUT, str):
        return get_cp_image_str(FULL_IMAGE_INPUT) 
    
    elif isinstance(FULL_IMAGE_INPUT, Image.Image):
        return (cp.array(FULL_IMAGE_INPUT) / 256).astype(cp.float64)
    
    else:
        assert False

def border_position(arr):
    mask_ge05 = arr < 0.5 # Black pixels
    struct_element = cp.ones((5, 5), dtype=bool)
    mask_neighbor_ge05 = binary_dilation(mask_ge05, structure=struct_element, border_value=False)
    mask_gt05 = arr >= 0.5 # White pixels
    final_mask = mask_gt05 & mask_neighbor_ge05 # Current pixel is white and black pixels exist in surrounding area
    result = final_mask.astype(cp.int32)
    return result

def find_match_pos_raw(FULL_IMAGE_INPUT: str|Image.Image, IMAGE_PART_INPUT: str|Image.Image):
    full_image_cp = get_cp_image(FULL_IMAGE_INPUT)
    part_image = get_l_image(IMAGE_PART_INPUT)

    # Construct numpy object for patch image
    timer.begin_timer("image to numpy: patch image:p3")
    part_image_cp = (cp.array(part_image) / 256).astype(cp.float64)
    border_part_cp = border_position(part_image_cp)
    timer.end_timer("image to numpy: patch image:p3")

    # Preprocess vector X
    timer.begin_timer("preprocessing vector X")
    X = cp.zeros(full_image_cp.shape)
    X[full_image_cp <  0.5] = 1 # Interior: 1
    X[full_image_cp >= 0.5] = 0 # Exterior: 0
    timer.end_timer("preprocessing vector X")

    # Preprocess vectors Y and P
    timer.begin_timer("preprocessing vector Y")
    Y = cp.zeros(part_image_cp.shape)
    Y[part_image_cp  <  0.5] = 1 # Interior: 1
    Y[part_image_cp  >= 0.5] = 0 # Exterior: 0
    P = cp.zeros(part_image_cp.shape)
    P[part_image_cp  <  0.5] = 1.0 # Interior weight: 1
    P[border_part_cp >= 0.5] = 0.5 # Border weight: 0.5
    timer.end_timer("preprocessing vector Y")

    timer.begin_timer("match_nd")
    ANS = match_arr(X, Y, P)
    timer.end_timer("match_nd")

    timer.begin_timer("sorting solution:p1")
    pos = cp.argmin(ANS)
    posX, posY = cp.unravel_index(pos, ANS.shape)
    timer.end_timer("sorting solution:p1")

    return [(posY, posX, ANS[posX, posY] / cp.sum(P))]

def find_match_pos(FULL_IMAGE_INPUT, IMAGE_PART_INPUT) -> cp.ndarray:
    timer.begin_timer("$find_match_pos")
    p1list = find_match_pos_raw(FULL_IMAGE_INPUT, IMAGE_PART_INPUT)
    posX, posY, score = p1list[0]
    timer.end_timer("$find_match_pos")
    return posX, posY, score

# Notes
#   Black pixels are the actual pixels to be matched
#   White pixels are blank background pixels
def find_match_pos_and_rotate(FULL_IMAGE_INPUT, IMAGE_PART_INPUT) -> Tuple[int, int, float, float]:
    timer.begin_timer("$find_match_pos_and_rotate")

    # Record current solution (rotation angle)
    timer.ban_all_timer()
    rotate_now = 0.0
    posX_now, posY_now, score_now = find_match_pos(FULL_IMAGE_INPUT, IMAGE_PART_INPUT)
    timer.allow_all_timer()

    # Record optimal solution
    rotate_best = rotate_now
    posX_best, posY_best, score_best = posX_now, posY_now, score_now

    for i in tqdm(range(4, 360, 4)):
        rotate_now = i
        timer.ban_all_timer()
        posX_now, posY_now, score_now = find_match_pos(FULL_IMAGE_INPUT, 
            rotate.rotate_and_crop_white_borders(IMAGE_PART_INPUT, None, rotate_now))
        timer.allow_all_timer()
        
        if score_now < score_best:
            rotate_best = rotate_now
            posX_best, posY_best, score_best = posX_now, posY_now, score_now

    rotate_base = rotate_best
    for i in tqdm(range(-20, 20, 3)):
        rotate_now = rotate_base + i / 10

        timer.ban_all_timer()
        posX_now, posY_now, score_now = find_match_pos(FULL_IMAGE_INPUT, 
            rotate.rotate_and_crop_white_borders(IMAGE_PART_INPUT, None, rotate_now))
        timer.allow_all_timer()
        
        if score_now < score_best:
            rotate_best = rotate_now
            posX_best, posY_best, score_best = posX_now, posY_now, score_now
    
    timer.end_timer("$find_match_pos_and_rotate")
    return posX_best, posY_best, score_best, rotate_best

# Red mask image
def get_red_mask_image(FULL_IMAGE_INPUT:str|Image.Image, IMAGE_PART_INPUT:str|Image.Image, posY:int, posX:int, rot_deg:float):
    red_mask = black_to_red_transparent(
        rotate.rotate_and_crop_white_borders(get_l_image(IMAGE_PART_INPUT), None, rot_deg))
    ans_image = get_l_image(FULL_IMAGE_INPUT).convert("RGBA").copy()
    ans_image.paste(red_mask, (int(posY), int(posX)), mask=red_mask)
    return ans_image

def get_rotated_and_moved_image(full_image_size: Tuple[int, int]|str, IMAGE_PART_INPUT:str|Image.Image, posY:int, posX:int, rot_deg:float):
    if isinstance(full_image_size, str):
        full_image_size = get_l_image(full_image_size).size
    rotated_image = rotate.rotate_and_crop_white_borders(get_l_image(IMAGE_PART_INPUT), None, rot_deg)
    ans_image = Image.new("L", full_image_size, "white")
    ans_image.paste(rotated_image, (int(posY), int(posX)))
    return ans_image
