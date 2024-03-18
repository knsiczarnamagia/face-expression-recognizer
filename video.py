import os
from os.path import splitext
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF


class VideoPipeline:

    frame_counter = 1

    class_to_label = {
        0: 'angry',
        1: 'disgust',
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise',
    }

    def batch_predict_on_frames(self, ordered_image_paths: list[str], model: torch.nn.Module) -> None:
        """The function assumes that image paths in the list are in correct order.
        """
        images = [Image.open(path) for path in ordered_image_paths]

        modified_images = list(map(lambda im: self.predict_on_frame(im, model), images))

        save_dir = 'dev/labelled'
        os.makedirs(save_dir, exist_ok=True)
        for img in modified_images:
            save_path = os.path.join(save_dir, f'frame_{VideoPipeline.frame_counter}.jpg')
            VideoPipeline.frame_counter += 1
            img.save(save_path)

    def predict_on_frame(self, image: Image.Image, model: torch.nn.Module, device: str = 'cpu') -> Image.Image:
        model = model.to(device)
        img_tensor = TF.to_tensor(image).to(device)
        with torch.inference_mode:
            predicted_class = model(img_tensor)
        image_label = self.class_to_label[predicted_class]
        draw_text = 'Label: ' + image_label

        w = image.width
        h = image.height
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('./fonts/NotoMusic-Regular.ttf', int(0.03 * w))
        draw.text((int(0.05 * w), int(0.03 * h)), draw_text, (255, 0, 0), font)
        return image

    def _order_image_paths(self, root_dir: str) -> list[str]:
        ext = ['.jpg', '.jpeg', '.png']
        image_filenames = [i for i in os.listdir(root_dir) if splitext(i)[1] in ext]
        image_filenames.sort(key=self._extract_file_counter)
        image_paths = [os.path.join(root_dir, file) for file in image_filenames]
        return image_paths

    def _extract_file_counter(self, filename: str) -> int:
        pattern = r'^.*?_(\d+)$'
        err = ValueError(f'Couldn\'t extract file counter. Filename "{filename}" has a wrong format.')
        filename = splitext(filename)[0]
        match = re.search(pattern, filename)
        if match:
            counter = match.group(1)
            try:
                counter = int(counter)
            except ValueError:
                raise err
            return counter
        else:
            raise err
