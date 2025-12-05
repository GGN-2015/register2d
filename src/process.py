import os
from PIL import Image
import cupy as cp
import numpy as np
import timer
import functools
from match_arr import match_arr
from cupyx.scipy.ndimage import binary_dilation

def black_to_red_transparent(img_l):
    # 1. 验证输入是 L 模式（按需求保证，但添加校验更健壮）
    if img_l.mode != 'L':
        raise ValueError(f"输入图像必须是 L 模式，当前模式为：{img_l.mode}")
    
    # 2. 转为 NumPy 数组（shape=(高度, 宽度)，灰度值 0-255）
    img_arr = cp.array(img_l, dtype=cp.uint8)
    height, width = img_arr.shape
    
    # 3. 构建 RGBA 数组（shape=(高度, 宽度, 4)，4通道：R, G, B, Alpha）
    # 初始化全透明（Alpha=0），RGB 通道默认0
    rgba_arr = cp.zeros((height, width, 4), dtype=cp.uint8)
    
    # 4. 核心逻辑：黑色像素（>128）→ 红色（R=255, G=0, B=0）+ 不透明（Alpha=255）
    # 生成黑色像素的掩码（True 表示是黑色像素）
    black_mask = img_arr < 128
    
    # 给黑色像素赋值：R=255，Alpha=255（G和B保持0）
    rgba_arr[black_mask, 0] = 255  # R通道：红色
    rgba_arr[black_mask, 3] = 128  # Alpha通道：半透明（255=完全不透明，0=完全透明）
    
    # 其他像素保持默认：RGB=0, Alpha=0（透明），无需额外处理
    
    # 5. 转为 RGBA 模式的 Pillow Image 对象并返回
    rgba_img = Image.fromarray(cp.asnumpy(rgba_arr), mode='RGBA')
    return rgba_img

def fft_convolve_1d(vec1, vec2) -> cp.ndarray:
    # 1. 验证输入是一维向量
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("输入必须是一维 NumPy 向量")
    
    # 2. 计算需要的最小长度（避免循环卷积影响线性卷积结果）
    len1 = len(vec1)
    len2 = len(vec2)
    min_length = max(len1, len2) * 2 - 1  # 线性卷积的理论长度
    
    # 3. 对两个向量做零填充（填充到最小长度，确保卷积结果完整）
    vec1_padded = cp.pad(vec1, (0, min_length - len1), mode='constant')
    vec2_padded = cp.pad(vec2, (0, min_length - len2), mode='constant')
    
    # 4. 核心步骤：FFT → 频域乘积 → 逆FFT
    fft1 = cp.fft.fft(vec1_padded)  # 向量1的频域表示
    fft2 = cp.fft.fft(vec2_padded)  # 向量2的频域表示
    fft_product = fft1 * fft2       # 频域乘积（对应时域卷积）
    conv_result = cp.fft.ifft(fft_product)  # 逆FFT还原时域
    
    # 5. 去除虚部误差（数值计算导致的微小虚部，实际卷积结果应为实数）
    conv_result = cp.real(conv_result)
    
    return conv_result

def extract_boundary_pixels(input_img):
    # 1. 处理输入：若为路径则打开图片，若为Image对象则直接使用
    if isinstance(input_img, str):
        with get_l_image(input_img) as img:
            gray_img = img.convert("L")  # 转为单通道灰度图（0-255）
    elif isinstance(input_img, Image.Image):
        gray_img = input_img.convert("L")  # 确保转为L模式，兼容彩色/灰度输入
    else:
        raise TypeError("输入必须是图片路径字符串或 Pillow Image 对象")
    
    # 2. 转为NumPy数组进行处理（shape=(高度, 宽度)）
    img_arr = np.array(gray_img)
    height, width = img_arr.shape
    
    # 3. 初始化输出数组（全黑，同尺寸）
    boundary_arr = np.zeros((height, width), dtype=cp.uint8)  # 0=黑色
    
    # 4. 定义8邻域的偏移量
    neighbors = [
        (dy, dx) 
        for dy in range(-1, 2)  # dy: -2, -1, 0, 1, 2（行方向偏移）
        for dx in range(-1, 2)  # dx: -2, -1, 0, 1, 2（列方向偏移）
        if not (dy == 0 and dx == 0)  # 排除中心像素
    ]
    
    # 5. 遍历每个像素，判断是否为边界像素
    for y in range(height):
        for x in range(width):
            # 自身条件：像素为白色（>128）
            if img_arr[y, x] > 128:
                # 检查8邻域是否存在黑色像素（<128）
                has_black_neighbor = False
                for dy, dx in neighbors:
                    # 计算邻域坐标，避免越界
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if img_arr[ny, nx] < 128:
                            has_black_neighbor = True
                            break  # 找到一个即可，无需继续检查
                # 满足两个条件，标记为边界像素（白色255）
                if has_black_neighbor:
                    boundary_arr[y, x] = 255
    
    # 6. 转为Pillow Image对象并返回（不保存文件）
    boundary_img = Image.fromarray(boundary_arr, mode="L")
    return boundary_img

DIRNOW = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRNOW)

__global_obj_image_cache = {}

# 获取一个图片
def get_l_image(path_or_img:Image.Image|str):
    if isinstance(path_or_img, str):
        # 避免多次重读
        if __global_obj_image_cache.get(path_or_img) is None:
            img = Image.open(path_or_img).convert("L")
            __global_obj_image_cache[path_or_img] = img
        return __global_obj_image_cache[path_or_img]
    
    elif isinstance(path_or_img, Image.Image):
        return path_or_img.convert("L")

    else:
        assert False

@functools.cache
def get_np_image(FULL_IMAGE_INPUT):
    return (cp.array(get_l_image(FULL_IMAGE_INPUT)) / 256).astype(cp.float64)

def border_position(arr):
    mask_ge05 = arr >= 0.5
    struct_element = cp.ones((3, 3), dtype=bool)
    mask_neighbor_ge05 = binary_dilation(mask_ge05, structure=struct_element, border_value=False)
    mask_gt05 = arr > 0.5
    final_mask = mask_gt05 & mask_neighbor_ge05
    result = final_mask.astype(cp.int32)
    return result

def find_match_pos_raw(FULL_IMAGE_INPUT: str|Image.Image, IMAGE_PART_INPUT: str|Image.Image):
    full_image_np = get_np_image(FULL_IMAGE_INPUT)
    part_image = get_l_image(IMAGE_PART_INPUT)

    # 构建子图的 numpy 对象
    timer.begin_timer("image to numpy: patch image:p3")
    part_image_np = (cp.array(part_image) / 256).astype(cp.float64)
    border_part_np = border_position(part_image_np)
    timer.end_timer("image to numpy: patch image:p3")

    # 预处理 X 向量
    timer.begin_timer("preprocessing vector X")
    X = cp.zeros(full_image_np.shape)
    X[full_image_np <  0.5] = 1 # 内部: 1
    X[full_image_np >= 0.5] = 0 # 外部: 0
    timer.end_timer("preprocessing vector X")

    # 预处理 Y 和 P 向量
    timer.begin_timer("preprocessing vector Y")
    Y = cp.zeros(part_image_np.shape)
    Y[part_image_np  <  0.5] = 1 # 内部: 1
    Y[part_image_np  >= 0.5] = 0 # 外部: 0
    P = cp.zeros(part_image_np.shape)
    P[part_image_np  <  0.5] = 1.0 # 内部权重: 1
    P[border_part_np >= 0.5] = 0.5 # 边界权重: 0.5
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

import rotate
from tqdm import tqdm
def find_match_pos_and_rotate(FULL_IMAGE_INPUT, IMAGE_PART_INPUT):
    timer.begin_timer("$find_match_pos_and_rotate")

    # 记录当前解（旋转角度）
    # timer.ban_all_timer()
    rotate_now = 0.0
    posX_now, posY_now, score_now = find_match_pos(FULL_IMAGE_INPUT, IMAGE_PART_INPUT)
    # timer.allow_all_timer()

    # 记录最优解
    rotate_best = rotate_now
    posX_best, posY_best, score_best = posX_now, posY_now, score_now

    for i in tqdm(range(5, 360, 5)):
        rotate_now = i
        timer.ban_all_timer()
        posX_now, posY_now, score_now = find_match_pos(FULL_IMAGE_INPUT, 
            rotate.rotate_and_crop_white_borders(IMAGE_PART_INPUT, None, rotate_now))
        timer.allow_all_timer()
        
        if score_now < score_best:
            rotate_best = rotate_now
            posX_best, posY_best, score_best = posX_now, posY_now, score_now

    rotate_base = rotate_best
    for i in tqdm(range(-25, 25, 6)):
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
# 注意
#   黑色像素是被匹配的实体像素
#   白色像素是空白背景像素
FULL_IMAGE_INPUT = "all_data/data1/full_image.jpg"
IMAGE_PART_INPUT = "all_data/data1/image_part9.jpg"
get_np_image(FULL_IMAGE_INPUT)
get_np_image(IMAGE_PART_INPUT)
posY, posX, score, rot_deg = find_match_pos_and_rotate(FULL_IMAGE_INPUT, IMAGE_PART_INPUT)
print(score)

# 红色的掩码图像
red_mask = black_to_red_transparent(
    rotate.rotate_and_crop_white_borders(get_l_image(IMAGE_PART_INPUT), None, rot_deg))
ans_image = get_l_image(FULL_IMAGE_INPUT).convert("RGBA").copy()
ans_image.paste(red_mask, (int(posY), int(posX)), mask=red_mask)
ans_image.show()
