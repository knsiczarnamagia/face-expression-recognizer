from face_detector import FaceDetector
from model_small import ResNet18
import numpy as np
import torch
from torch import nn
from PIL import Image
from util import draw_bboxes, draw_label_on_bbox
import torchvision.transforms as T


class FaceExpressionRecognizer:

    _DATASET_MEAN = 0.5077385902404785
    _DATASET_STD = 0.255077600479126

    def __init__(self):
        self.face_detector = FaceDetector()
        self.fer_classifier = _make_fer_classifier()
        self.post_process = T.Compose([
            T.Resize((48, 48)),
            T.Grayscale(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(FaceExpressionRecognizer._DATASET_MEAN, FaceExpressionRecognizer._DATASET_STD)
        ])
        self.idx_to_label = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise',
        }

    def handle_frame(self, image: Image.Image) -> Image.Image:
        bboxes = self.face_detector.detect_bboxes(image)
        if bboxes is None:
            return image

        extracted_faces = self.face_detector.extract_faces(image, bboxes)
        extracted_faces = self.post_process(extracted_faces)
        preds = self.fer_classifier(extracted_faces).argmax(dim=1)
        print(f'Preds: {preds}')
        preds = preds.tolist()

        img_w_boxes = draw_bboxes(image.copy(), bboxes, (255, 0, 0))
        image_w_boxes_arr = np.array(img_w_boxes)
        for bbox, pred in zip(bboxes, preds):
            image_w_boxes_arr = draw_label_on_bbox(image_w_boxes_arr, bbox, self.idx_to_label[pred])
        return Image.fromarray(image_w_boxes_arr)


def _make_fer_classifier() -> nn.Module:
    model = ResNet18(1, 7)
    # fer_fc = nn.Linear(256, 7)
    # model = nn.Sequential(*list(model.children())[:-1])
    # model = nn.Sequential(*model, fer_fc)
    model.load_state_dict(torch.load('./saved_models/weighted_sampler200_fer_model.pth', map_location=torch.device('cpu')))
    return model
