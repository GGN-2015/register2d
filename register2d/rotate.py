from PIL import Image
import numpy as np

def rotate_and_crop_white_borders(input_path, output_path, rotate_angle, tolerance=10, anti_alias=True, disp=False):
    """
    Rotate an image counterclockwise (with compatible anti-aliasing) and crop white borders (supports near-white pixel detection)

    Args:
        input_path (str): Path to the input image (e.g., "input.jpg")
        output_path (str|None): Path to save the output image (e.g., "output.png")
        rotate_angle (int|float): Counterclockwise rotation angle (supports any angle, e.g., 30, 45.5)
        tolerance (int): White tolerance value (0-255), default 10. Larger values treat more near-white pixels as white
        anti_alias (bool): Whether to enable anti-aliasing, default True (enabled)
    """
    try:
        # Validate tolerance value range
        if not (0 <= tolerance <= 255):
            raise ValueError("Tolerance value must be between 0 and 255")

        # 1. Read image (preserve original channels, supports color/grayscale images)
        if isinstance(input_path, str):
            img = Image.open(input_path)

        elif isinstance(input_path, Image.Image):
            img = input_path.copy()

        else:
            assert False

        if disp:
            print(f"Successfully read image: {input_path}, image size: {img.size}")

        # 2. Configure rotation parameters (core: compatible anti-aliasing configuration)
        rotate_kwargs = {
            "angle": rotate_angle,
            "expand": True,       # Automatically expand canvas to avoid cropping the pattern
            "fillcolor": "white"  # Fill blank areas with white
        }

        # Anti-aliasing configuration: use BICUBIC algorithm compatible with rotation (replaces LANCZOS)
        if anti_alias:
            if hasattr(Image, "Resampling"):
                # Pillow 9.1.0+: Use rotation-compatible BICUBIC
                rotate_kwargs["resample"] = Image.Resampling.BICUBIC #type:ignore
            else:
                # Older Pillow versions: Use Image.BICUBIC
                rotate_kwargs["resample"] = Image.BICUBIC #type:ignore

            if disp:
                print("Anti-aliasing enabled (using BICUBIC algorithm, compatible with rotation scenarios)")
        else:
            # Use basic NEAREST algorithm when anti-aliasing is disabled
            if hasattr(Image, "Resampling"):
                rotate_kwargs["resample"] = Image.Resampling.NEAREST #type:ignore
            else:
                rotate_kwargs["resample"] = Image.NEAREST #type:ignore

            if disp:
                print("Anti-aliasing disabled")

        # Perform counterclockwise rotation
        rotated_img = img.rotate(**rotate_kwargs)
        if disp:
            print(f"Rotation completed ({rotate_angle} degrees), size after rotation: {rotated_img.size}")

        # 3. Automatically crop white borders (preserve tolerance logic)
        img_array = np.array(rotated_img)
        white_threshold = 255 - tolerance  # White threshold value

        if len(img_array.shape) == 3:  # Color image: all RGB channels ≥ threshold count as white
            non_white_pixels = np.where(
                (img_array[:, :, 0] < white_threshold) | 
                (img_array[:, :, 1] < white_threshold) | 
                (img_array[:, :, 2] < white_threshold)
            )
        else:  # Grayscale image: pixels below threshold are not white
            non_white_pixels = np.where(img_array < white_threshold)

        # Check if the entire image is white
        if len(non_white_pixels[0]) == 0:
            raise ValueError(f"At tolerance value {tolerance}, the rotated image is entirely white (including near-white), no valid pattern to retain")

        # Calculate boundaries of non-white areas
        top = float(np.min(non_white_pixels[0]))
        bottom = float(np.max(non_white_pixels[0]))
        left = float(np.min(non_white_pixels[1]))
        right = float(np.max(non_white_pixels[1]))

        # 4. Crop the image
        cropped_img = rotated_img.crop((left, top, right + 1, bottom + 1))
        if disp:
            print(f"Cropping completed (tolerance value: {tolerance}), size after cropping: {cropped_img.size}")

        # 5. Save results (PNG format is lossless by default, recommended)
        if output_path is not None:
            cropped_img.save(output_path)
            if disp:
                print(f"Image saved to: {output_path}")
        
        # Return the processed image
        return cropped_img

    except FileNotFoundError:
        print(f"Error: Input image file not found -> {input_path}")
        return None
    
    except ValueError as ve:
        print(f"Parameter error: {str(ve)}")
        return None
    
    except Exception as e:
        print(f"Processing failed: {str(e)}, Pillow version: {Image.__version__}")
        return None
