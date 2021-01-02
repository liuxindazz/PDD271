import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
INPUT_SIZE=(224, 224)

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.transform = transform
        self.data_root = data_root
       # print(self.data_root)

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.labels = []

        for data in data_list:
            image_path = data[:-1]
            label = image_path.split('/')[1]
            self.img_paths.append(image_path)
            self.labels.append(label)

    def __getitem__(self, item):
        img_path, label= self.img_paths[item], self.labels[item]
        # print(self.data_root)
        img_path = (self.data_root + img_path)
       # print(img_path)
        img = Image.open(img_path).convert('RGB')
        # label = np.array(label,dtype='float32')
        label = torch.tensor(int(label)-100)#.view(1,1)
        # label_one_hot = torch.zeros(1, 271).scatter_(1, label, 1)
        if self.transform is not None:
            img = self.transform(img)

        return img, label#label_one_hot[0].type(torch.LongTensor)

    def __len__(self):
        return self.n_data

if __name__ == "__main__":
    train_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])
    # train_dataset = GetLoader(data_root='/media/liuxinda/本地磁盘/BaiduYunDownload/image_resize_2/', 
    #                     data_list='/media/liuxinda/LXDUP/train_list.txt',  
    #                     transform=train_transform)
    # print(len(train_dataset))
    # print(len(train_dataset.labels))
    # for data in train_dataset:
    #     print(data[0].size(), data[1])
    test_dataset = GetLoader(data_root='/home/LAB/wangzz/lxd/data/image_resize_224/', 
                    data_list='validate_list.txt',  
                    transform=test_transform)
    print(len(test_dataset))
    print(len(test_dataset.labels))
    for data in test_dataset:
        print(data[0].size(), data[1])
