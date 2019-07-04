from model import autoencoder, unet, custom_unet
import keras
import glob
import tensorflow as tf
import cv2
import numpy as np
from skimage.measure import compare_ssim
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
K.set_session(session)

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return err
        
model = autoencoder()
model.load_weights('param/ae-left-weights.0.00663.h5')
resize_size = (128, 128)
for img_path in glob.glob('test/0/*.jpg'):
    print(img_path)
    img = cv2.resize(cv2.imread(img_path), (32, 32))
    
    cv2.imshow('Input', cv2.resize(img, resize_size))
    output_img = model.predict((cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.)[None, :, :, :], batch_size=1)
    # print(output_img)
    output_img = cv2.cvtColor(output_img[0], cv2.COLOR_RGB2BGR)
    cv2.imshow('Output', cv2.resize(output_img, resize_size))

#     print()
    
#     print(compare_ssim(cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY), cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)))
    print(compare_ssim(img / 255., output_img, multichannel=True), K.eval(mse(img / 255., output_img)))

    cv2.waitKey(0)
