import gradio as gr
from model import ResNet18
idx_to_label = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Suprised',
}
def classify_image(image):
    image_index = ResNet18(image)
    label_text = idx_to_label[image_index]
    return label_text

gradio_interface = gr.Interface(fn=classify_image, inputs=gr.Image("file"), outputs="text")

gradio_interface.launch()