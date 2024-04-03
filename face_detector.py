from facenet_pytorch import MTCNN
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageDraw
import numpy as np


class FaceDetector:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, device=device)

    def detect_bboxes(self, img: Image.Image) -> np.ndarray:
        boxes, probs = self.model.detect(img)
        return boxes
    
    def extract_faces(self, img: Image.Image, bounding_boxes) -> list[Image.Image]:
        face_images_tensors = self.model.extract(img, bounding_boxes, save_path='extracted/extracted.jpg')
        face_images = []
        for image in face_images_tensors:
            print(image.shape)
            face_images.append(TF.to_pil_image(image))
        return face_images


def draw_bboxes(img: Image.Image, boxes, color: tuple[int, int, int]) -> Image.Image:
    img_draw = ImageDraw.Draw(img.copy())
    for box in boxes:
        img_draw.rectangle(box.tolist(), outline=color, width=2)
    return img
