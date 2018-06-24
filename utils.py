import numpy as np
import pickle

def load_result(filename):
    return pickle.load(open(filename, 'rb'))

def confusion_matrix(pred, gt):
    num_data, num_class = np.shape(pred)
    result = np.zeros([num_class, num_class])
    pred_class = np.argmax(pred, axis=1)

    for idx, p in enumerate(pred_class):
        result[gt[idx], p] +=1

    return result

if __name__ == '__main__':
    gt = np.array(load_result('gt.txt'))
    pred = np.array(load_result('pred.txt'))
    print(np.shape(gt), np.shape(pred))
    temp = np.argmax(pred,axis=1)

    print(confusion_matrix(pred, gt))
