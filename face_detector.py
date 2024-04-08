from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np


class FaceDetector:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True, post_process=False, device=device)

    def detect_bboxes(self, img: Image.Image) -> np.ndarray:
        boxes, probs = self.model.detect(img)
        return boxes
    
    def extract_faces(self, img: Image.Image, bboxes) -> torch.Tensor:
        face_images_tensors = self.model.extract(img, bboxes, save_path='extracted/extracted.jpg')
        return face_images_tensors
