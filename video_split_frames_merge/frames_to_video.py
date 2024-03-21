import cv2
import os

def frames_to_video(image_folder,video_name,character_code,frame_rate):
    images = [img for img in os.listdir(image_folder) if img.endswith(".JPG")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, character_code, frame_rate, (width, height))
    #video_name- The file path name,
    # character_code -4 character code of codec(Could be 0,1,-1 or just ('M','J','P','G') is a motion-jpeg codec for example.
    # frame_rate - It's just framerate of created video.
    # (width,height) - Size of the video frames.

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder,image)))

    cv2.destroyAllWindows()
    video.release()