import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import os
import torchvision
import pandas as pd



def draw_recon(x, x_recon):
    x_l, x_recon_l = x.tolist(), x_recon.tolist()
    result = [None] * (len(x_l) + len(x_recon_l))
    result[::2] = x_l
    result[1::2] = x_recon_l
    return torch.FloatTensor(result)


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def make_dataloader(args):

    test_loader = None
    train_loader = None
    if args.dataset == 'celeba':

        trans_f = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CelebA(args.data_dir, split='train', download=False, transform=trans_f)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                                   drop_last=True, num_workers=4)

    elif 'pendulum' in args.dataset:
        train_set = dataload_withlabel(args.data_dir, image_size = args.image_size,
                                       mode='train', sup_prop=args.sup_prop)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    elif 'human' in args.dataset:
        train_set = dataload_withlabel(args.data_dir, image_size = args.image_size,
                                       mode='train', sup_prop=args.sup_prop)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    elif 'tree' in args.dataset:
        train_set = dataload_withlabel(args.data_dir, image_size = args.image_size,
                                       mode='train', sup_prop=args.sup_prop)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    elif 'minist' in args.dataset:
        train_set = torchvision.datasets.MNIST(args.data_dir,
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)


    return train_loader, test_loader, train_set


def check_for_CUDA(sagan_obj):
    if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
        print("CUDA is available!")
        sagan_obj.device = torch.device('cuda')
        sagan_obj.config.dataloader_args['pin_memory'] = True
    else:
        print("Cuda is NOT available, running on CPU.")
        sagan_obj.device = torch.device('cpu')

    if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")

def get_scale(train_set):
    '''return max and min of training data'''
    # 假设列表的名字是lst
    # 初始化最小值和最大值为第一个元素的三个值
    lst = train_set.imglabel
    min_val = lst[0].copy()
    max_val = lst[0].copy()
    # 遍历剩余的元素，比较每个位置的值，更新最小值和最大值
    for i in range(1, len(lst)):
        for j in range(len(train_set.imglabel[0])):
            if lst[i][j] < min_val[j]:
                min_val[j] = lst[i][j]
            if lst[i][j] > max_val[j]:
                max_val[j] = lst[i][j]
    # 生成结果列表
    scale = [min_val, max_val]
    scale = torch.tensor(scale)
    # scale = torch.Tensor([[0.0000, 48.0000, 2.0000, 2.0178], [40.5000, 88.5000, 14.8639, 14.4211]])
    return scale

def get_stats():
    '''return mean and std of training data'''
    mm = torch.Tensor([20.2500, 68.2500, 6.9928, 8.7982])
    ss = torch.Tensor([11.8357, 11.8357, 2.8422, 2.1776])
    return mm, ss





class dataload_withlabel(torch.utils.data.Dataset):
    def __init__(self, root, label_file=None, image_size=64, mode="train", sup_prop=1., num_sample=0):
        # label_file: 'pendulum_label_downstream.txt'

        self.label_file = label_file
        if label_file is not None:
            self.attrs_df = pd.read_csv(os.path.join(root, label_file))
            # attr = self.attrs_df[:, [1,2,3,7,5]]
            self.split_df = pd.read_csv(os.path.join(root, label_file))
            splits = self.split_df['partition'].values
            split_map = {
                "train": 0,
                "valid": 1,
                "test": 2,
                "all": None,
            }
            split = split_map[verify_str_arg(mode.lower(), "split",
                                             ("train", "valid", "test", "all"))]
            mask = slice(None) if split is None else (splits == split)
            self.mask = mask
            np.random.seed(2)
            if num_sample > 0:
                idxs = [i for i, x in enumerate(mask) if x]
                not_sample = np.random.permutation(idxs)[num_sample:]
                mask[not_sample] = False
            self.attrs_df = self.attrs_df.values
            self.attrs_df[self.attrs_df == -1] = 0
            self.attrs_df = self.attrs_df[mask][:, [0,1,2,3,6]]
            self.imglabel = torch.as_tensor(self.attrs_df.astype(np.float))
            self.imgs = []
            for i in range(3):
                mode1 = list(split_map.keys())[i]
                root1 = root + mode1
                imgs = os.listdir(root1)
                self.imgs += [os.path.join(root, mode1, k) for k in imgs]
            self.imgs = np.array(self.imgs)[mask]
        else:
            root = root + mode
            imgs = os.listdir(root)
            self.imgs = [os.path.join(root, k) for k in imgs]
            self.imglabel = [list(map(float, k[:-4].split("_")[1:])) for k in imgs]
        self.transforms = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        np.random.seed(2)
        self.n = len(self.imgs)
        self.available_label_index = np.random.choice(self.n, int(self.n * sup_prop), replace=0)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        if not (idx in self.available_label_index):
            label = torch.zeros(4).long() - 1
        else:
            if self.label_file is None:
                label = torch.from_numpy(np.asarray(self.imglabel[idx]))
            else:
                label = self.imglabel[idx]
        pil_img = Image.open(img_path).convert('RGB')
        array = np.array(pil_img)
        array1 = np.array(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96,96,3)
            data = torch.from_numpy(pil_img)
        return data, label.float()

    def __len__(self):
        return len(self.imgs)
