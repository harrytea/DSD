import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, dst1, dst2):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), \
                dst1.transpose(Image.FLIP_LEFT_RIGHT), dst2.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask, dst1, dst2


class CustomDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        # self.imgs = make_dataset(root)
        self.img_path  = [os.path.join(self.root, 'ShadowImages', img) for img in sorted(os.listdir(os.path.join(self.root, 'ShadowImages')))]
        self.gt_path   = [os.path.join(self.root, 'ShadowMasks', img) for img in sorted(os.listdir(os.path.join(self.root, 'ShadowMasks')))]
        self.dst1_path = [os.path.join(self.root, 'fuse_dst1', img) for img in sorted(os.listdir(os.path.join(self.root, 'fuse_dst1')))]
        self.dst2_path = [os.path.join(self.root, 'fuse_dst2', img) for img in sorted(os.listdir(os.path.join(self.root, 'fuse_dst2')))]
        self.num = len(self.img_path)
        self.hflip = RandomHorizontallyFlip()
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.label_trans = transforms.ToTensor()

    def __len__(self):
        return self.num*100

    def __getitem__(self, index):
        img    = Image.open(self.img_path[index%self.num]).convert('RGB').resize((320,320), Image.BILINEAR)
        target = Image.open(self.gt_path[index%self.num]).convert('L').resize((320,320), Image.NEAREST)
        dst1   = Image.open(self.dst1_path[index%self.num]).convert('L').resize((320,320), Image.NEAREST)
        dst2   = Image.open(self.dst2_path[index%self.num]).convert('L').resize((320,320), Image.NEAREST)

        img, target, dst1, dst2 = self.hflip(img, target, dst1, dst2)
        img_nom = self.transform(img)
        target = self.label_trans(target)
        dst1 = self.label_trans(dst1)
        dst2 = self.label_trans(dst2)

        return img_nom, target, dst1, dst2, np.array(img, dtype='float32').transpose(2,0,1)/255



class TestDataset(data.Dataset):
    def __init__(self, imgs_path, labs_path):
        super().__init__()
        self.imgs_path = imgs_path
        self.labs_path = labs_path
        self.imgs = sorted(os.listdir(self.imgs_path))
        self.labs = sorted(os.listdir(self.labs_path))

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        label_path = self.labs[index]
        image = Image.open(os.path.join(self.imgs_path, image_path)).convert('RGB').resize((320, 320))

        label = Image.open(os.path.join(self.labs_path, label_path)).convert('L').resize((320, 320))
        # transform
        label = np.array(label, dtype='float32')/255.0
        if len(label.shape) > 2:
            label = label[:,:,0]
        label = np.expand_dims(label, axis=0)
        image_nom = self.trans(image)
        sample = {'O': image_nom, 'B': label}
        return sample, image_path