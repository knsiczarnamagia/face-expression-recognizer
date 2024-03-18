import cv2
import os

def frames_to_video(image_folder,video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder,image)))

    cv2.destroyAllWindows()
    video.release()