import cv2
import gradio as gr
import torch
from torchvision.transforms import v2

from model_old import ResNet18

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet18(1, 7)
model.load_state_dict(torch.load("./model.pth", map_location=device))
model.to(device)
model.eval()

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

class_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

DATASET_MEAN = 0.5077385902404785
DATASET_STD = 0.255077600479126

preprocess = v2.Compose(
    [
        v2.Grayscale(),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(DATASET_MEAN,), std=(DATASET_STD,)),
    ]
)


def get_probs(image):
    inp = preprocess(torch.tensor(image).permute(2, 0, 1).unsqueeze(0))
    inp = inp.to(device)
    pred = model(inp).squeeze()
    probs = torch.softmax(pred, 0).cpu()
    return probs


def draw_labels(image, cords: tuple, label: str):
    x, y, w, h = cords
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    image = cv2.putText(
        image,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return image


def predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for cords in faces:
        x, y, w, h = cords
        resized = cv2.resize(image[y : y + h, x : x + w], (48, 48), cv2.INTER_AREA)
        probs = get_probs(resized)
        label = class_list[probs.argmax(0).item()]
        image = draw_labels(image, cords, label)

    return image


webcam_interface = gr.Interface(
    predict,
    inputs=gr.Image(sources=['webcam'], streaming=True, label='Input webcam'),
    outputs=gr.Image(label='Output video'),
    live=True,
    title='Webcam mode',
    description='Created by Czarna Magia AI Student Club',
    theme=gr.themes.Soft(),
)

img_interface = gr.Interface(
    predict,
    inputs=gr.Image(sources=['webcam', 'upload'], label='Input image'),
    outputs=gr.Image(label='Output image'),
    title='Image upload mode',
    description='Created by Czarna Magia AI Student Club',
    theme=gr.themes.Soft(),
)

iface = gr.TabbedInterface(
    interface_list=[img_interface, webcam_interface],
    tab_names=['Image upload', 'Webcam'],
    title='Face Expression Recognizer',
    theme=gr.themes.Soft(),
)
iface.launch()
