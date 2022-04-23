import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import argparse
import numpy as np
from tool.utils import *
from tqdm import tqdm
from PIL import Image
from tool.dataset import TestDataset
from misc import crf_refine
from model import DSDNet


sbu_image = r"/data4/wangyh/Datasets/shadow/SBU-shadow/SBU-shadow/SBU-Test/ShadowImages"
sbu_mask  = r"/data4/wangyh/Datasets/shadow/SBU-shadow/SBU-shadow/SBU-Test/ShadowMasks"


parser = argparse.ArgumentParser(description='')
parser.add_argument('--ckpt_dir',   default='./ckpts/model', help='directory for checkpoints')
parser.add_argument('--save_dir',   default='./ckpts/test',  help='directory for checkpoints')
parser.add_argument('--batch_size', default=1, type=int,     help='number of samples in one batch')
args = parser.parse_args()



def main():
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_dataset    = TestDataset(sbu_image, sbu_mask)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=8, pin_memory=False)

    net = DSDNet().cuda()
    load_checkpoints(net, args.ckpt_dir, "5000.pth")
    net.eval()
    for i, (batch, file_path) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            O, B,= batch['O'], batch['B']
            O, B = O.cuda(), B.cuda()
            predict= net(O)

            image = Image.open(os.path.join(sbu_image, file_path[0])).convert('RGB')
            final = Image.fromarray((predict.cpu().data * 255).numpy().astype('uint8')[0,0,:,:])
            final = np.array(final.resize(image.size))
            final_crf = crf_refine(np.array(image), final)
            io.imsave(os.path.join(args.save_dir, file_path[0]), final_crf)


if __name__ == '__main__':
    main()
