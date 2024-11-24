#开发者：王逸轩
#开发时间：2024/10/28 12:40
#-*- coding = utf-8 -*-
from dominate.tags import label
from torch.utils.data import Dataset
# 读取图片
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        # 读取图片的位置
        self.root_dir = root_dir        # 数据集位置
        self.label_dir = label_dir      # 标签数据位置
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)
        # print("image path", self.image_path)
        '''
        测试结果表明，该模块只运行一次，获取的是所有label_img的名字；
        整体是一个list
        结果为：image path ['0013035.jpg', '1030023514_aad5c608f9.jpg', '1095476100_3906d8afde.jpg',....]
        '''
    # 读取每一个图片
    def __getitem__(self, item):
        # 读取文件名字
        img_name = self.image_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        print("item path", img_item_path)
        img = Image.open(img_item_path)
        img.show()
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


if __name__ == '__main__':
    # 数据集根目录
    root_dir = "hymenoptera_data/train"
    # 标签为ants的数据
    ants_label_dir = "ants"
    # 标签为bees的数据
    bees_label_dir = "bees"
    # 通过类整合成list数据集
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)

    train_dataset = ants_dataset + bees_dataset