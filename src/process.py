import os
from PIL import Image
import numpy as np
import timer

# PATTERN_VAL 描述内部点匹配时被视为什么值
# 相比之下边界点在匹配时被视为 1
# 匹配距离为 \sum_i y_i(x_i - y_i)^2
# PATTERN_VAL 越大，内部点匹配的权重相对于边界越大
PATTERN_VAL = 3
assert PATTERN_VAL >= 2 and isinstance(PATTERN_VAL, int)

def black_to_red_transparent(img_l):
    # 1. 验证输入是 L 模式（按需求保证，但添加校验更健壮）
    if img_l.mode != 'L':
        raise ValueError(f"输入图像必须是 L 模式，当前模式为：{img_l.mode}")
    
    # 2. 转为 NumPy 数组（shape=(高度, 宽度)，灰度值 0-255）
    img_arr = np.array(img_l, dtype=np.uint8)
    height, width = img_arr.shape
    
    # 3. 构建 RGBA 数组（shape=(高度, 宽度, 4)，4通道：R, G, B, Alpha）
    # 初始化全透明（Alpha=0），RGB 通道默认0
    rgba_arr = np.zeros((height, width, 4), dtype=np.uint8)
    
    # 4. 核心逻辑：黑色像素（>128）→ 红色（R=255, G=0, B=0）+ 不透明（Alpha=255）
    # 生成黑色像素的掩码（True 表示是黑色像素）
    black_mask = img_arr < 128
    
    # 给黑色像素赋值：R=255，Alpha=255（G和B保持0）
    rgba_arr[black_mask, 0] = 255  # R通道：红色
    rgba_arr[black_mask, 3] = 128  # Alpha通道：半透明（255=完全不透明，0=完全透明）
    
    # 其他像素保持默认：RGB=0, Alpha=0（透明），无需额外处理
    
    # 5. 转为 RGBA 模式的 Pillow Image 对象并返回
    rgba_img = Image.fromarray(rgba_arr, mode='RGBA')
    return rgba_img

def fft_convolve_1d(vec1, vec2) -> np.ndarray:
    # 1. 验证输入是一维向量
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("输入必须是一维 NumPy 向量")
    
    # 2. 计算需要的最小长度（避免循环卷积影响线性卷积结果）
    len1 = len(vec1)
    len2 = len(vec2)
    min_length = max(len1, len2) * 2 - 1  # 线性卷积的理论长度
    
    # 3. 对两个向量做零填充（填充到最小长度，确保卷积结果完整）
    vec1_padded = np.pad(vec1, (0, min_length - len1), mode='constant')
    vec2_padded = np.pad(vec2, (0, min_length - len2), mode='constant')
    
    # 4. 核心步骤：FFT → 频域乘积 → 逆FFT
    fft1 = np.fft.fft(vec1_padded)  # 向量1的频域表示
    fft2 = np.fft.fft(vec2_padded)  # 向量2的频域表示
    fft_product = fft1 * fft2       # 频域乘积（对应时域卷积）
    conv_result = np.fft.ifft(fft_product)  # 逆FFT还原时域
    
    # 5. 去除虚部误差（数值计算导致的微小虚部，实际卷积结果应为实数）
    conv_result = np.real(conv_result)
    
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
    boundary_arr = np.zeros((height, width), dtype=np.uint8)  # 0=黑色
    
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

def sort_array_by_value_with_coords(arr):
    # 1. 获取数组的所有坐标（返回的是每个维度的索引数组，如2D返回(行索引数组, 列索引数组)）
    coords = np.indices(arr.shape)  # 形状：(维度数, 元素总数)，如2D数组shape=(2, h, w)
    
    # 2. 将坐标数组展平为一维（每个维度的索引对应所有元素），再转置为 (元素总数, 维度数) 的坐标矩阵
    # 例如2D数组：coords展平后为(2, h*w)，转置后为(h*w, 2)，每行是一个元素的(行,列)坐标
    coords_flatten = coords.reshape(arr.ndim, -1).T  # arr.ndim 是数组维度数
    
    # 3. 将坐标矩阵的每行转为 tuple（方便后续使用，如(0,1)而非[0,1]）
    coords_list = [tuple(coord) for coord in coords_flatten]
    
    # 4. 将数组本身展平为一维（与坐标列表一一对应）
    values_flatten = arr.flatten()
    
    # 5. 组合 "值-坐标" 有序对（zip 对应索引的 值 和 坐标）
    value_coord_pairs = list(zip(values_flatten, coords_list))
    
    # 6. 按元素值从小到大排序（key=lambda x: x[0] 表示按第一个元素（值）排序）
    sorted_pairs = sorted(value_coord_pairs, key=lambda x: x[0])
    
    return sorted_pairs

def invert_color_pillow(point_img):
    # 核心：对每个像素应用 255 - x（反色公式）
    # RGBA 模式：仅反转 RGB 通道，保留 Alpha 通道（用 lambda x: x 保持不变）
    if point_img.mode == "RGBA":
        # 分别处理 R、G、B、A 通道，A通道不反转
        r, g, b, a = point_img.split()
        r_invert = r.point(lambda x: 255 - x)
        g_invert = g.point(lambda x: 255 - x)
        b_invert = b.point(lambda x: 255 - x)
        invert_img = Image.merge("RGBA", (r_invert, g_invert, b_invert, a))
    else:
        # L/RGB 模式：直接对所有通道应用反色
        invert_img = point_img.point(lambda x: 255 - x)
    return invert_img

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

def find_match_pos_raw(FULL_IMAGE_INPUT, IMAGE_PART_INPUT, INVERT_COLOR, MAX_MATCH_CNT=1):
    assert MAX_MATCH_CNT >= 1

    # 完整图片的 size
    full_size = get_l_image(FULL_IMAGE_INPUT).size

    # 完整图片的 numpy 对象
    timer.begin_timer("image to numpy: full image")
    raw_image = get_l_image(FULL_IMAGE_INPUT)
    if INVERT_COLOR:
        raw_image = invert_color_pillow(raw_image)
    full_image_np = (np.array(raw_image) / 256).astype(np.float64)
    timer.end_timer("image to numpy: full image")

    # 构建和完整图片相同尺寸
    timer.begin_timer("image to numpy: patch image:p1")
    raw_image = get_l_image(IMAGE_PART_INPUT)
    if INVERT_COLOR:
        raw_image = invert_color_pillow(raw_image)
    part_image = raw_image
    white_background = Image.new("L", full_size, "white")
    white_background.paste(part_image, (0, 0))
    timer.end_timer("image to numpy: patch image:p1")

    # 提取边界像素
    timer.begin_timer("image to numpy: patch image:p2")
    border_part = extract_boundary_pixels(white_background)
    border_part_np = np.array(border_part)
    timer.end_timer("image to numpy: patch image:p2")

    # 构建子图的 numpy 对象
    timer.begin_timer("image to numpy: patch image:p3")
    part_image_np = (np.array(white_background) / 256).astype(np.float64)
    timer.end_timer("image to numpy: patch image:p3")

    # 展平
    full_image_np_flat  = full_image_np.flatten()
    part_image_np_flat  = part_image_np.flatten()[::-1] # 翻转了
    border_part_np_flat = border_part_np.flatten()[::-1]
    assert full_image_np_flat.size == part_image_np_flat.size
    flat_size = full_image_np_flat.size

    # 预处理 X 向量
    timer.begin_timer("preprocessing vector X")
    X  = np.zeros(flat_size)
    X[full_image_np_flat <  0.5] = PATTERN_VAL # 内部: PATTERN_VAL
    X[full_image_np_flat >= 0.5] = 1 # 外部: 1
    X2 = X ** 2
    timer.end_timer("preprocessing vector X")

    # 预处理 Y 向量
    timer.begin_timer("preprocessing vector Y")
    Y  = np.zeros(flat_size)
    Y[part_image_np_flat  <  0.5] = PATTERN_VAL # 内部: PATTERN_VAL
    Y[part_image_np_flat  >= 0.5] = 0 # 外部: 0
    Y[border_part_np_flat >= 0.5] = 1 # 边界: 1
    Y2 = Y ** 2
    Y3SUM = (Y ** 3).sum()
    timer.end_timer("preprocessing vector Y")

    timer.begin_timer("fft_convolve_1d: X2Y, XY2")
    X2Y = fft_convolve_1d(X2, Y)
    XY2 = fft_convolve_1d(X, Y2)
    ANS = (X2Y - (2 * XY2) + Y3SUM)[len(X) - 1:]
    assert ANS.size == full_image_np_flat.size
    ANS = ANS.reshape(full_image_np.shape)
    timer.end_timer("fft_convolve_1d: X2Y, XY2")

    timer.begin_timer("sorting solution")
    answer_list = sort_array_by_value_with_coords(ANS)
    arr = []
    for i in range(min(MAX_MATCH_CNT, len(answer_list))):
        posX, posY = answer_list[i][1]

        # 这里最好再限制一下 posY 和 posX 的范围，避免掩码区域中有意义的部分移动到下一行
        arr.append([int(posY), int(posX)])
    timer.end_timer("sorting solution")
    return np.array(arr)

def find_match_pos(FULL_IMAGE_INPUT, IMAGE_PART_INPUT) -> np.ndarray:
    timer.begin_timer("$find_match_pos")
    p1list = find_match_pos_raw(FULL_IMAGE_INPUT, IMAGE_PART_INPUT, False)
    posX, posY = p1list[0]

    # score 描述了当前匹配位置的优越程度，score 越低匹配越优秀
    score = 0
    full_image   = get_l_image(FULL_IMAGE_INPUT)
    part_image   = get_l_image(IMAGE_PART_INPUT)
    border_image = extract_boundary_pixels(part_image).convert("L")
    cnt = 0
    for i in range(part_image.width):
        for j in range(part_image.height):
            if 0 <= posX + i < full_image.width and 0 <= posY + j < full_image.height:
                pixel = full_image.getpixel((posX + i, posY + j))
            else:
                pixel = 255 # 出界

            # 边界惩罚
            if border_image.getpixel((i, j)) > 128:
                if pixel < 128: # 不在外表
                    score += (PATTERN_VAL - 1) ** 2
            
            # 内部惩罚
            if part_image.getpixel((i, j)) < 128:
                if pixel > 128: # 在外面
                    score += PATTERN_VAL * (PATTERN_VAL - 1) ** 2
                cnt += 1 # 统计内部点
    
    timer.end_timer("$find_match_pos")
    return posX, posY, score / cnt

import rotate
from tqdm import tqdm
def find_match_pos_and_rotate(FULL_IMAGE_INPUT, IMAGE_PART_INPUT):

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
    
    return posX_best, posY_best, score_best, rotate_best
# 注意
#   黑色像素是被匹配的实体像素
#   白色像素是空白背景像素
FULL_IMAGE_INPUT = "all_data/data2/full_image.png"
IMAGE_PART_INPUT = "all_data/data2/image_part3.png"
posY, posX, score, rot_deg = find_match_pos_and_rotate(FULL_IMAGE_INPUT, IMAGE_PART_INPUT)
print(score)

# 红色的掩码图像
red_mask = black_to_red_transparent(
    rotate.rotate_and_crop_white_borders(get_l_image(IMAGE_PART_INPUT), None, rot_deg))
ans_image = get_l_image(FULL_IMAGE_INPUT).convert("RGBA").copy()
ans_image.paste(red_mask, (posY, posX), mask=red_mask)
ans_image.show()
