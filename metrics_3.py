import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import cv2
def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    sum_n_ii = 0
    sum_t_i = 0
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i
    print("pixel accuracy = %f" %pixel_accuracy_)
    return pixel_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''
    check_size(eval_segm, gt_segm)
    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = list([0]) * n_cl
    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue
        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)
        IU[i] = n_ii / (t_i + n_ij - n_ii)
    mean_IU_ = np.sum(IU) / n_cl_gt
    print("miou = %f" % mean_IU_)
    return mean_IU_

def get_CCA(eval_segm, gt_segm):
    #check_size(eval_segm, gt_segm)
    #cl, n_cl = union_classes(eval_segm, gt_segm)
    #eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    # 建立模型
    cca = CCA(n_components=1)
    #cca_2 = CCA(n_components=2)
    # 训练数据
    #print(eval_mask.shape)
    #print(gt_mask.shape)
    cca.fit(eval_segm, gt_segm)
    #cca_2.fit(eval_mask, gt_mask)
    # 降维操作
    # print(X)
    X_train_r, Y_train_r = cca.transform(eval_segm, gt_segm)
    # print(X_train_r)
    #print(np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1])  # 输出相关系数
    #print(np.corrcoef(X_train_r[:, 1], Y_train_r[:, 1])[0, 1])
    CCA_=np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]
    return CCA_

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    """
    FP = np.float(matrix.sum(axis=0) - np.diag(matrix))
    FN = np.float(matrix.sum(axis=1) - np.diag(matrix))
    TP = np.float(np.diag(matrix))
    TN = np.float(matrix.sum() - (FP + FN + TP))
    """
    return FP, FN, TP, TN

def accuracy_score(prediction, groundtruth):
    """Getting the accuracy of the model"""
    g = groundtruth == 1
    g = np.array(g, bool)
    p = prediction == 1
    p = np.array(p, bool)
    #cm=confusion_matrix(g,p)
    #print(cm)
    FP, FN, TP, TN = numeric_score(p,g)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP, N+FP)
    acc=accuracy * 100.0
    print("cca = %f" % acc)
    return acc

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)
    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm) #cl=[0 1 2]
    n_cl = len(cl) #n_cl=3
    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)
    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)
    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))
    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise
    return height, width

def check_size(eval_segm, gt_segm):

    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)
    #print(h_e, w_e)
    #print(h_g, w_g)
    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")
'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)