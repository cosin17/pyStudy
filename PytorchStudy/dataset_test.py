from logging import root
from torch.utils.data import Dataset
from PIL import Image
import os



class MyData(Dataset):
    # 初始化
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)  # 拼接图片路径
        self.img_path = os.listdir(self.path)   # 读取图片路径

    # 获取数据
    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 读取图片名称
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir # 直接把文件夹名称作为标签返回
        return img,label

    # 获取数据集大小
    def __len__(self):
        return len(self.img_path)

root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"

ants_dataset = MyData(root_dir,ants_label_dir) # 加载蚂蚁数据集
