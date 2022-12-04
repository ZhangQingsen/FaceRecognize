import cv2
import numpy as np
from PIL import Image

# img = cv2.imread('.//lfw//Charles_Bell//Charles_Bell_0001.jpg')
# img = cv2.imread('.//lfw//Aaron_Guiel//Aaron_Guiel_0001.jpg')
# img = cv2.imread(".//lfw//Allen_Iverson//Allen_Iverson_0002.jpg")

face_detector = cv2.CascadeClassifier(".//model//haarcascade_frontalface_default.xml")


def align(img):
    eye_detector = cv2.CascadeClassifier(".//model//haarcascade_eye.xml")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(grey, 1.01, 200, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    try:
        dx = eyes[0][0] - eyes[-1][0]
        dx = dx if dx > 0 else -dx
        dy = eyes[0][1] - eyes[-1][1]
        dy = dy if dy > 0 else -dy
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), np.tan(dy / dx), 1)
        res = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    except IndexError:
        res = img.copy()
    return res

def faceDetectordir(img_dir):
    img = cv2.imread(img_dir)
    return faceDetector(img)

def faceDetector(img):
    align_img = align(img)
    faces = face_detector.detectMultiScale(align_img, 1.01, 100, cv2.CASCADE_SCALE_IMAGE, (20, 20))
    if len(faces):
        # print(f'       {len(faces)}， {faces}')
        for x, y, w, h in faces:
            align_img = align_img[x:x + w, y:y + h]
            break
    align_img = Image.fromarray(cv2.cvtColor(align_img, cv2.COLOR_BGR2RGB)) # 转为PIL image
    return align_img

# eyes = eye_detector.detectMultiScale(res, 1.01, 150, cv2.CASCADE_SCALE_IMAGE, (20, 20))
# for (ex, ey, ew, eh) in eyes:
#     cv2.rectangle(res, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# cv2.imshow('face', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
