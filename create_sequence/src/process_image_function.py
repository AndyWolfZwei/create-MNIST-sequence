import cv2
from PIL import Image
import numpy as np


def rotate_img(param, angle):
    a = Image.fromarray(param)
    img2 = a.rotate(angle)
    return img2


def noise_img(img, noise_ratio):
    img = np.array(img)
    n = img.shape[0] * img.shape[1] * noise_ratio
    for k in range(int(n)):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    return img


def split_margin(img):
    width = img.shape[1]
    v_sum = np.sum(img, axis=0)
    left = 0
    right = width - 1
    for i in range(width):
        if v_sum[i] > 0:
            left = i
            break
    for i in range(width - 1, -1, -1):
        if v_sum[i] > 0:
            right = i
            break
    return img[:, left:right]

def add_erode(img, erode_kernel):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erode_kernel)
    img = cv2.erode(img,kernel)
    return img


def add_dilate(img, dilate_kernel):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    img = cv2.dilate(img,kernel)
    return img

def data_augmentation(process_img, args):
    if args['random_all']:
        args['erode_kernel'] = (np.random.randint(1, 4), np.random.randint(1, 4))
        args["dilate_kernel"] = (np.random.randint(1, 4), np.random.randint(1, 4))
        process_img = add_erode(process_img, args['erode_kernel']) if np.random.random() < 0.5 else process_img
        process_img = add_dilate(process_img, args["dilate_kernel"]) if np.random.random() < 0.5 else process_img
        process_img = rotate_img(process_img, np.random.rand() * 90) if np.random.random() < 0.5 else process_img
        process_img = noise_img(process_img, np.random.rand() / 10) if np.random.random() < 0.5 else process_img
    else:
        process_img = add_erode(process_img, args['erode_kernel'])
        process_img = add_dilate(process_img, args["dilate_kernel"])
        process_img = rotate_img(process_img, args["angle"])
        process_img = noise_img(process_img, args["noise_ratio"])
    return process_img