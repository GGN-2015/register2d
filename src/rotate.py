from PIL import Image
import numpy as np

def rotate_and_crop_white_borders(input_path, output_path, rotate_angle, tolerance=10, anti_alias=True):
    """
    对图片进行逆时针旋转（兼容型抗锯齿）并裁剪白色边框（支持接近白色像素判定）

    参数:
        input_path (str): 输入图片的路径（如"input.jpg"）
        output_path (str): 输出图片的保存路径（如"output.png"）
        rotate_angle (int/float): 逆时针旋转的角度（支持任意角度，如30、45.5）
        tolerance (int): 白色容差值（0-255），默认10。值越大，越多接近白色的像素会被视为白色
        anti_alias (bool): 是否开启抗锯齿，默认True（开启）
    """
    try:
        # 校验容差值范围
        if not (0 <= tolerance <= 255):
            raise ValueError("容差值必须在0-255之间")

        # 1. 读取图片（保持原图通道，支持彩色/灰度图）
        img = Image.open(input_path)
        print(f"成功读取图片：{input_path}，图片尺寸：{img.size}")

        # 2. 配置旋转参数（核心：兼容型抗锯齿配置）
        rotate_kwargs = {
            "angle": rotate_angle,
            "expand": True,       # 自动扩展画布，避免图案被裁
            "fillcolor": "white"  # 空白区域用白色填充
        }

        # 抗锯齿配置：选用兼容旋转的BICUBIC算法（替代LANCZOS）
        if anti_alias:
            if hasattr(Image, "Resampling"):
                # Pillow 9.1.0+ 版本：使用兼容旋转的BICUBIC
                rotate_kwargs["resample"] = Image.Resampling.BICUBIC
            else:
                # Pillow 旧版本：使用Image.BICUBIC
                rotate_kwargs["resample"] = Image.BICUBIC
            print("已开启抗锯齿（使用BICUBIC算法，兼容旋转场景）")
        else:
            # 关闭抗锯齿时使用最基础的NEAREST算法
            if hasattr(Image, "Resampling"):
                rotate_kwargs["resample"] = Image.Resampling.NEAREST
            else:
                rotate_kwargs["resample"] = Image.NEAREST
            print("未开启抗锯齿")

        # 执行逆时针旋转
        rotated_img = img.rotate(**rotate_kwargs)
        print(f"旋转完成（{rotate_angle}度），旋转后尺寸：{rotated_img.size}")

        # 3. 自动裁剪白色边框（保留容差逻辑）
        img_array = np.array(rotated_img)
        white_threshold = 255 - tolerance  # 白色阈值

        if len(img_array.shape) == 3:  # 彩色图：RGB三通道都≥阈值才算白色
            non_white_pixels = np.where(
                (img_array[:, :, 0] < white_threshold) | 
                (img_array[:, :, 1] < white_threshold) | 
                (img_array[:, :, 2] < white_threshold)
            )
        else:  # 灰度图：像素值低于阈值不算白色
            non_white_pixels = np.where(img_array < white_threshold)

        # 检查是否全是白色图片
        if len(non_white_pixels[0]) == 0:
            raise ValueError(f"容差值{tolerance}下，图片旋转后全为白色（含接近白色），无有效图案可保留")

        # 计算非白色区域边界
        top = np.min(non_white_pixels[0])
        bottom = np.max(non_white_pixels[0])
        left = np.min(non_white_pixels[1])
        right = np.max(non_white_pixels[1])

        # 4. 裁剪图片
        cropped_img = rotated_img.crop((left, top, right + 1, bottom + 1))
        print(f"裁剪完成（容差值：{tolerance}），裁剪后尺寸：{cropped_img.size}")

        # 5. 保存结果（PNG格式默认无损，推荐使用）
        cropped_img.save(output_path)
        print(f"图片已保存到：{output_path}")

    except FileNotFoundError:
        print(f"错误：找不到输入图片文件 -> {input_path}")
    except ValueError as ve:
        print(f"参数错误：{str(ve)}")
    except Exception as e:
        print(f"处理失败：{str(e)}，Pillow版本：{Image.__version__}")
        
# ------------------- 测试示例（修改以下参数即可使用）-------------------
if __name__ == "__main__":
    import os
    DIRNOW = os.path.dirname(os.path.abspath(__file__))
    os.chdir(DIRNOW)

    # 请根据你的需求修改这3个参数
    INPUT_IMAGE_PATH = "all_data/data1/image_part1.jpg"    # 输入图片路径（支持jpg、png、bmp等格式）
    OUTPUT_IMAGE_PATH = "all_data/data1/image_part4.jpg"  # 输出图片路径（建议用png格式，避免压缩失真）
    ROTATE_ANGLE = 34.5               # 逆时针旋转角度（例如：30、45、90、120.5等）

    # 执行处理
    rotate_and_crop_white_borders(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH, ROTATE_ANGLE)
