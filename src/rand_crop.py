import random
from PIL import Image
import cupy as cp

def rand_crop(image: Image.Image) -> Image.Image:
    """
    从输入图片中随机截取满足灰度条件的正方形区域
    
    参数:
        image: Pillow Image 对象，输入的原始图片
    
    返回:
        Pillow Image 对象，满足条件的正方形截取区域
    """
    # 将图片转换为灰度模式（方便计算灰度值）
    gray_image = image.convert('L')
    width, height = gray_image.size
    
    # 确定正方形的最大可能边长（取宽高中较小值）
    max_side = round(min(width, height) * 0.35)
    
    # 循环直到找到满足条件的正方形区域
    while True:
        # 1. 随机生成正方形的边长（10像素以上，避免过小无意义）
        side_length = max_side
        
        # 2. 随机计算正方形左上角坐标（确保不越界）
        max_x = width - side_length
        max_y = height - side_length
        if max_x <= 0 or max_y <= 0:
            continue  # 极端情况保护，避免坐标越界
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # 3. 截取正方形区域
        crop_box = (x, y, x + side_length, y + side_length)
        cropped_gray = gray_image.crop(crop_box)
        
        # 4. 转换为numpy数组方便计算（比逐像素遍历更高效）
        crop_array = cp.array(cropped_gray)
        total_pixels = crop_array.size
        
        # 5. 计算满足条件的像素比例
        # 灰度 > 128 的像素占比
        ratio_above = (crop_array > 128).sum() / total_pixels
        # 灰度 < 128 的像素占比
        ratio_below = (crop_array < 128).sum() / total_pixels
        
        # 6. 判断是否满足条件（都超过30%）
        if ratio_above > 0.3 and ratio_below > 0.3:
            # 返回原始图片的对应彩色区域（如果输入是彩色图）
            return image.crop(crop_box)

# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    import os
    DIRNOW = os.path.dirname(os.path.abspath(__file__))
    os.chdir(DIRNOW)

    # 示例：读取图片并截取满足条件的正方形区域
    try:
        # 替换为你的图片路径
        input_image = Image.open("all_data/data1/full_image.jpg")
        # 调用函数获取结果
        result_image = rand_crop(input_image)
        # 保存结果
        result_image.show()
        print("成功截取并保存满足条件的正方形区域！")
        print(f"截取区域尺寸: {result_image.size}")
    except FileNotFoundError:
        print("错误：未找到指定的图片文件，请检查路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")