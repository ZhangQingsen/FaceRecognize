import os
import cv2
import FaceAlign
if __name__ == '__main__':
    g = os.walk(r'.\lfw')
    count = 0
    for path, dir_list, file_list in g:
        for file_name in file_list:
            count += 1
            imgName = os.path.join(path, file_name)
            print(imgName, count)
            img = cv2.imread(imgName)
            img = FaceAlign.faceDetector(img)
            if count == 3000:
                cv2.imshow(imgName, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                count = 0
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113