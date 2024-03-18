import cv2
import os
def video_to_frames(video_path):
    vid_cap = cv2.VideoCapture(video_path)
    try:
        if not os.path.exists('data'):
            os.makedirs('data')

    except OSError:
        print("Error: Creating Directory")
    curr_frame = 0

    while True:
        ret, frame = vid_cap.read()

        if ret:
            name = './data/frame' + str(curr_frame) + ".JPG"
            print("Creating: "+ name)

            cv2.imwrite(name, frame)

            curr_frame += 1
        else:
            break

    vid_cap.release()
    cv2.destroyAllWindows()
