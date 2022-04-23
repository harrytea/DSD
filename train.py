import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import argparse
from tool.utils import *
from tool.dataset import CustomDataset

from torch import optim
from torch.backends import cudnn
from misc import AvgMeter
from model import DSDNet



parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_model',   default='./ckpts/model', help='checkpoint model dir')
parser.add_argument('--ckpt_image',   default='./ckpts/image', help='checkpoint image dir')
parser.add_argument('--batch_size',   default=10, help='training batch size')
parser.add_argument('--weight_decay', default=1e-3, help='')
parser.add_argument('--lr',           default=5e-3, help='')
parser.add_argument('--lr_decay',     default=0.9, help='')
parser.add_argument('--step_num',     default=5000, help='total number of stepations')
parser.add_argument('--data_root',    default='/data4/wangyh/Datasets/shadow/SBU-shadow/SBU-shadow/SBUTrain4KRecoveredSmall')
args = parser.parse_args()



def main():
    '''  1. ensure dir  '''
    exist_dir(args.ckpt_model)
    exist_dir(args.ckpt_image)


    '''  2. datasets  '''
    train_dataset = CustomDataset(args.data_root)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, shuffle=True)


    '''  3. initial  '''
    bce_logit = MyBceloss12_n().cuda()
    bce_logit_dst = MyWcploss().cuda()
    net = DSDNet().cuda().train()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'], 'lr': 2*args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'], 'lr': args.lr, 'weight_decay': args.weight_decay}], momentum=0.9)


    step = 0
    train_loss_record_shad, loss_fuse_record_shad, loss_down1_record_shad = AvgMeter(), AvgMeter(), AvgMeter()
    loss_down2_record_shad, loss_down3_record_shad, loss_down4_record_shad = AvgMeter(), AvgMeter(), AvgMeter()
    train_loss_record_dst1, loss_fuse_record_dst1, loss_down1_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()
    loss_down2_record_dst1, loss_down3_record_dst1, loss_down4_record_dst1 = AvgMeter(), AvgMeter(), AvgMeter()
    train_loss_record_dst2, loss_fuse_record_dst2, loss_down1_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
    loss_down2_record_dst2, loss_down3_record_dst2, loss_down4_record_dst2 = AvgMeter(), AvgMeter(), AvgMeter()
    train_loss_record = AvgMeter()


    train_loader = iter(train_loader)
    while step<=5000:
        optimizer.param_groups[0]['lr'] = 2 * args.lr * (1-float(step)/args.step_num) ** args.lr_decay
        optimizer.param_groups[1]['lr'] = args.lr * (1-float(step)/args.step_num) ** args.lr_decay

        image, labels, labels_dst1, labels_dst2, im = next(train_loader)

        batch_size = image.size(0)
        image, labels = image.cuda(), labels.cuda()
        labels_dst1, labels_dst2 = labels_dst1.cuda(), labels_dst2.cuda()

        optimizer.zero_grad()
        fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
        fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1, \
        fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
        pred_down0_dst1, pred_down0_dst2, pred_down0_shad = net(image)

        loss_fuse_shad = bce_logit(fuse_pred_shad, labels, labels_dst1, labels_dst2)
        loss1_shad = bce_logit(pred_down1_shad, labels, labels_dst1, labels_dst2)
        loss2_shad = bce_logit(pred_down2_shad, labels, labels_dst1, labels_dst2)
        loss3_shad = bce_logit(pred_down3_shad, labels, labels_dst1, labels_dst2)
        loss4_shad = bce_logit(pred_down4_shad, labels, labels_dst1, labels_dst2)
        loss0_shad = bce_logit(pred_down0_shad, labels, labels_dst1, labels_dst2)

        loss_fuse_dst1 = bce_logit_dst(fuse_pred_dst1, labels_dst1)
        loss1_dst1 = bce_logit_dst(pred_down1_dst1, labels_dst1)
        loss2_dst1 = bce_logit_dst(pred_down2_dst1, labels_dst1)
        loss3_dst1 = bce_logit_dst(pred_down3_dst1, labels_dst1)
        loss4_dst1 = bce_logit_dst(pred_down4_dst1, labels_dst1)
        loss0_dst1 = bce_logit_dst(pred_down0_dst1, labels_dst1)

        loss_fuse_dst2 = bce_logit_dst(fuse_pred_dst2, labels_dst2)
        loss1_dst2 = bce_logit_dst(pred_down1_dst2, labels_dst2)
        loss2_dst2 = bce_logit_dst(pred_down2_dst2, labels_dst2)
        loss3_dst2 = bce_logit_dst(pred_down3_dst2, labels_dst2)
        loss4_dst2 = bce_logit_dst(pred_down4_dst2, labels_dst2)
        loss0_dst2 = bce_logit_dst(pred_down0_dst2, labels_dst2)

        loss_shad = loss_fuse_shad + loss1_shad + loss2_shad + loss3_shad + loss4_shad +loss0_shad
        loss_dst1 = loss_fuse_dst1 + loss1_dst1 + loss2_dst1 + loss3_dst1 + loss4_dst1 +loss0_dst1
        loss_dst2 = loss_fuse_dst2 + loss1_dst2 + loss2_dst2 + loss3_dst2 + loss4_dst2 +loss0_dst2
        loss = loss_shad + 2*loss_dst1 + 2*loss_dst2

        loss.backward()
        optimizer.step()

        train_loss_record.update(loss.data, batch_size)
        train_loss_record_shad.update(loss_shad.data, batch_size)
        loss_fuse_record_shad.update(loss_fuse_shad.data, batch_size)
        loss_down1_record_shad.update(loss1_shad.data, batch_size)
        loss_down2_record_shad.update(loss2_shad.data, batch_size)
        loss_down3_record_shad.update(loss3_shad.data, batch_size)
        loss_down4_record_shad.update(loss4_shad.data, batch_size)

        train_loss_record_dst1.update(loss_dst1.data, batch_size)
        loss_fuse_record_dst1.update(loss_fuse_dst1.data, batch_size)
        loss_down1_record_dst1.update(loss1_dst1.data, batch_size)
        loss_down2_record_dst1.update(loss2_dst1.data, batch_size)
        loss_down3_record_dst1.update(loss3_dst1.data, batch_size)
        loss_down4_record_dst1.update(loss4_dst1.data, batch_size)

        train_loss_record_dst2.update(loss_dst2.data, batch_size)
        loss_fuse_record_dst2.update(loss_fuse_dst2.data, batch_size)
        loss_down1_record_dst2.update(loss1_dst2.data, batch_size)
        loss_down2_record_dst2.update(loss2_dst2.data, batch_size)
        loss_down3_record_dst2.update(loss3_dst2.data, batch_size)
        loss_down4_record_dst2.update(loss4_dst2.data, batch_size)

        log = '[step %d], [train loss %.5f], [loss_train_shad %.5f], [loss_train_dst1 %.5f], [loss_train_dst2 %.5f], [lr %.13f]' % \
                (step, train_loss_record.avg, train_loss_record_shad.avg, train_loss_record_dst1.avg,
                train_loss_record_dst2.avg, optimizer.param_groups[1]['lr'])
        print(log)


        fuse_pred_shad, fuse_pred_dst1, fuse_pred_dst2 = [torch.sigmoid(fuse_pred_shad), torch.sigmoid(fuse_pred_dst1), torch.sigmoid(fuse_pred_dst2)]

        if step%500==0:
            torch.save(net.state_dict(), os.path.join(args.ckpt_model, '%d.pth' % step))
        if step%200==0:
            save_mask(args.ckpt_image, step, im, labels, fuse_pred_shad, labels_dst1, fuse_pred_dst1, labels_dst2, fuse_pred_dst2)
        if step==args.step_num:
            torch.save(net.state_dict(), os.path.join(args.ckpt_model, '%d.pth' % step))
            return
        step += 1


if __name__ == '__main__':
    init_seeds(2018)
    main()