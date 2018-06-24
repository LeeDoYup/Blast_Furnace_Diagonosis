import glob
import numpy as np
import cv2
import pickle
import os
import fire
import commons

def dir_label(path='./org_img', num_label=5):
    dir_list = glob.glob(path+'/*')
    assert len(dir_list) == num_label
    dir_label_dict = {}
    for idx, dir in enumerate(dir_list):
        label = int(dir.split('/')[-1])
        dir_label_dict[label]=dir

    return dir_label_dict

def img_label(lb_dir_dict):
    img_label = {}
    train_img = {}
    test_img = {}
    for key in lb_dir_dict:
        imgs = []
        img_dir = glob.glob(lb_dir_dict[key]+'/*.bmp')
        for idx, dir in enumerate(img_dir):
            print("Read: ", dir)
            img = commons.imread(dir)
            img = commons.extract_region(img)
            imgs.append(img)
        img_label[key] = imgs
        train_size = int(len(imgs)*0.8)
        train_img[key] = imgs[:train_size]
        test_img[key] = imgs[train_size:]

    return img_label, train_img, test_img

lb_dir_dict = dir_label()
img_dict, train_img, test_img = img_label(lb_dir_dict)

pickle.dump(train_img, open('./train_img.txt','wb'))
pickle.dump(test_img, open('./test_img.txt','wb'))
pickle.dump(img_dict, open('./org_img.txt','wb'))



