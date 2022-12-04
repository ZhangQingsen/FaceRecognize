import numpy as np
import torch
import model
import datasetSplite
from PIL import Image
from dataloader import resize_image
from FaceAlign import faceDetectordir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def detect_image(_model, img1_dir, img2_dir, input_shape=114):
        #---------------------------------------------------#
        #   图片预处理，人脸对齐
        #---------------------------------------------------#
        # img_1 = faceDetectordir(img1_dir)
        # img_2 = faceDetectordir(img2_dir)
        img_1 = Image.open(img1_dir).resize((input_shape,input_shape),Image.ANTIALIAS)
        img_2 = Image.open(img2_dir).resize((input_shape,input_shape),Image.ANTIALIAS)
        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        with torch.no_grad():
            img_1 = resize_image(img_1, [input_shape, input_shape], letterbox_image=True)
            img_2 = resize_image(img_2, [input_shape, input_shape], letterbox_image=True)
            
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose((np.array(img_1, np.float32))/ 255.0, (2, 0, 1)), axis=0)).to(device)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose((np.array(img_2, np.float32))/ 255.0, (2, 0, 1)), axis=0)).to(device)

            # photo_1 = torch.from_numpy(np.transpose((np.array(img_1, np.float32))/ 255.0, (2, 0, 1))).to(device)
            # photo_2 = torch.from_numpy(np.transpose((np.array(img_2, np.float32))/ 255.0, (2, 0, 1))).to(device)
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            _model = _model.to(device)
            output1 = _model(photo_1).cpu().numpy()
            output2 = _model(photo_2).cpu().numpy()
            
            #---------------------------------------------------#
            #   计算二者之间的距离
            #---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)
            return l1


def main():
    model_dir = './model_data/ep039-loss1.236-accu0.831.pth'
    model_dir = './model_data/ep125-loss2.163-accu0.732.pth'
    img1_dir = '001.jpg'
    img2_dir = '002.jpg'
    threshold = 1

    df = datasetSplite.main()
    _model = model.ConvNet(num_classes=len(np.unique(df["Name"])))
    if len(model_dir):
        checkpoint = torch.load(model_dir)
        _model.load_state_dict(checkpoint['model'])
    
    probability = detect_image(_model, img1_dir, img2_dir, input_shape=114)
    if probability < threshold:
        print(f'{probability} < {threshold}, two images belongs to the same person')
    else:
        print(f'{probability} > {threshold}, two images belongs to two people')
    
    return probability


if __name__ == "__main__":
    main()