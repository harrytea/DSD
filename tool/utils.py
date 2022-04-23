import os
import torch
import random
import numpy as np
import torch.nn as nn
import skimage.io as io
import torch.nn.functional as F

def exist_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


'''  initial seed  '''
def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


class MyBceloss12_n(nn.Module):
    def __init__(self):
        super(MyBceloss12_n, self).__init__()

    def forward(self, pred, gt, dst1, dst2):
        eposion = 1e-10
        sigmoid_dst1 = torch.sigmoid(dst1)
        sigmoid_dst2 = torch.sigmoid(dst2)
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos/(count_pos+count_neg)
        dst_loss = beta*(1+dst2)*gt*F.binary_cross_entropy_with_logits(pred, gt, reduction='none') + \
                   (1+dst1)*(1-gt)*F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        bce2_lss = torch.mean(dst_loss)
        loss = beta_back*bce1(pred, gt) + beta_back*bce2_lss
        return loss


class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()

    def forward(self, pred, gt):
        eposion = 1e-10
        sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss


def save_mask(path_dir, step, image, labels, fuse_pred_shad, labels_dst1, fuse_pred_dst1, labels_dst2, fuse_pred_dst2):
        image, labels= (image.cpu().numpy()*255).astype('uint8'), (labels.cpu().numpy()*255).astype('uint8')
        labels = np.tile(labels, (3,1,1))
        h, w = 320, 320
        gen_num = (2, 1)
        fuse_pred_shad, labels_dst1, fuse_pred_dst1, labels_dst2, fuse_pred_dst2 = \
        (np.tile(fuse_pred_shad.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(labels_dst1.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(fuse_pred_dst1.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(labels_dst2.cpu().data * 255,(3,1,1))).astype('uint8'), \
        (np.tile(fuse_pred_dst2.cpu().data * 255,(3,1,1))).astype('uint8')

        img = np.zeros((gen_num[0]*h, gen_num[1]*7*w, 3)).astype('uint8')
        for i in range(gen_num[0]):
            row = i * h
            for j in range(gen_num[1]):
                idx = i * gen_num[1] + j
                tmp_list = [image[idx], labels[idx], fuse_pred_shad[idx], labels_dst1[idx], fuse_pred_dst1[idx], labels_dst2[idx], fuse_pred_dst2[idx]]
                for k in range(7):
                    col = (j * 7 + k) * w
                    tmp = np.transpose(tmp_list[k], (1, 2, 0))
                    img[row: row+h, col: col+w] = tmp

        img_file = os.path.join(path_dir, '%d.jpg'%(step))
        io.imsave(img_file, img)


def load_checkpoints(model, model_dir, name):
    ckp_path = os.path.join(model_dir, name)
    try:
        obj = torch.load(ckp_path)
    except FileNotFoundError:
        return print("File Not Found")
    model.load_state_dict(obj)