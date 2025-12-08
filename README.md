# register2d
Given a black-and-white binary image A and another smaller black-and-white binary image B, perform translation and rotation on B, and find the position in A that is closest to the transformed B. A and B should be saved in grey PNG image (mode='L').

## Installation
```bash
pip install register2d
```

## Usage
```python
import register2d
image_a_path = "<Image A Path>"
image_b_path = "<Image B Path>"

# calculate registration
posY, posX, score, rot_deg = register2d.find_match_pos_and_rotate(image_a_path, image_b_path)

# show the position of image B in image A
register2d.get_red_mask_image(image_a_path, image_b_path, posY, posX, rot_deg).show()

# show the position of image B in a white image same size as A
register2d.get_rotated_and_moved_image(image_a_path, IMAGE_PART_INPUT, posY, posX, rot_deg).show()
```
