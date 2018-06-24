import random

import cv2
import numpy as np
import glob
import os

from configs import VGGConf


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = BGR2RGB(img)
    return img


def BGR2RGB(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def extract_region(img, x_range=[110,290], y_range=[0,180]):
    x_s, x_e = x_range
    y_s, y_e = y_range

    img = img.transpose([1,0,2])
    im_shape = np.shape(img)

    assert x_s >= 0 and x_e <= im_shape[0]
    assert y_s >=0 and y_e <=im_shape[1]

    crop = img[x_s:x_e, y_s:y_e, :]
    resize = cv2.resize(crop, (128,128))
    return resize


def minmax(num, min_num, max_num):
    return max(min(num, max_num), min_num)


def choices(seq, l):
    # for support python2
    return [random.choice(seq) for _ in range(l)]


def random_idxs(max, k):
    if k >= max:
        return [random.randint(0, max - 1) for _ in range(k)]
    else:
        l = list(range(max))
        random.shuffle(l)
        return l[:k]


def choices_by_idx(seq, idxs):
    return [seq[x] for x in idxs]


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def onehot(idxs):
    a = np.zeros(shape=(len(idxs), 5), dtype=np.int8)
    # a[0, np.arange(len(idxs)), idxs] = 1
    for i, _idx in enumerate(idxs):
        idx = int(_idx)
        if idx >= 5 or idx < 0:
            continue
        a[i, idx] = 1
    return a

def onehot_flatten(idxs):
    a = onehot(idxs)
    a = a.reshape((1, a.shape[0]*a.shape[1]))
    return a


def imshow_grid(title, images, cols, rows):
    h, w = images[0].shape[:2]
    canvas = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = (i // cols) * h
        x = (i % cols) * w
        canvas[y:y+h, x:x+w] = img
    cv2.imshow(title, canvas)

def get_image_pathes(data_path='./data', remove_folders=['__MACOSX']):
    names = glob.glob(data_path+'/*')

    #Remove unnecessary folder names
    for _, remove_folder in enumerate(remove_folders):
        remove_path = os.path.join(data_path, remove_folder)
        if remove_path in names: names.remove(remove_path)


    _names = names
    for idx, name in enumerate(_names):
        if '.txt' in name: names.remove(name)
    
    #names: only data folder name
    assert len(names)>0
    names.sort()

    #Make {folder: label.txt} dictionary

    path = {}
    for idx, name in enumerate(names):
        lb_name = name+'_cnn.txt'
        assert os.path.exists(lb_name)
        path[name] = lb_name

    return path #path dictionary = [key: image folder path], [value: label file path]

def read_image_lb_path(path_dict, is_train=True):
    keys = list(path_dict.keys())
    keys.sort()
    num_key = len(keys)
    images, lbs = [], []
    for idx, key in enumerate(keys):
        print("Images and Labels of {} are LOADING....".format(key))
        img, lb = read_image_lb(key, path_dict[key], is_train)
        images.append(img)
        lbs.append(lb)

    return images, lbs # [# of folder, # of image, 128, 128, 3], [# of folder, # of labels, ]

def read_image_lb(i_path, l_path, is_train=True, img_type='.bmp'):
    images, labels = [], []
    image_paths = glob.glob(i_path+'/*'+img_type)
    image_paths.sort()

    label_lines = open(l_path,'r').readlines()
    assert len(image_paths) == len(label_lines)

    #iter_key = {0: 3, 1:1, 2:15, 3:7, 4:1}
    name_list = []
    iter_key = {0: 1, 1:1, 2:1, 3:1, 4:1}
    label_dict = {}

    for idx, path in enumerate(image_paths):
        name = path.split('/')[-1]
        name_list.append(name)

    for idx, path in enumerate(label_lines):
        name, lb = path.split(' ')
        lb = lb.split('\n')[0]
        label_dict[name] = lb

    for path, name in zip(image_paths, name_list):
        img = imread(path)
        img = extract_region(img)
        label = label_dict[name]
        for k in range(4):
            for _ in range(iter_key[int(label)]):
                images.append(np.rot90(img,k))
                labels.append(label)
                if not is_train: break
    return images, labels



def start_index(vid_path):
    dirs = vid_path.split('/')
    tb_check = (dirs[-2] == 'TB-50' or dirs[-2] == 'TB-100')
    print(min(glob.glob(vid_path+'/img/*')))
    start_idx = int(min(glob.glob(vid_path+'/img/*')).split('/')[-1].split('.')[-2])
    print(start_idx)
    if tb_check and dirs[-1] == 'David': return start_idx + (300-1)
    else: return start_idx
