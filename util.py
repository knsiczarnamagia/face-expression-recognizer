import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_bboxes(img: Image.Image, boxes, color: tuple[int, int, int]) -> Image.Image:
    img_draw = ImageDraw.Draw(img)
    for box in boxes:
        img_draw.rectangle(box.tolist(), outline=color, width=2)
    return img


def draw_label_on_bbox(image: np.ndarray, bbox: list[float], text: str) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    left_pos = bbox[0]
    bottom_pos = bbox[1] - 5
    bottom_left_position = (int(left_pos), int(bottom_pos))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .9
    color = (255, 0, 0)
    thickness = 2

    annotated_image = cv2.putText(image, text, bottom_left_position, font, font_scale, color, thickness)
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
