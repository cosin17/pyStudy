import os
import torch
import xmltodict

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VOCDataset(Dataset):
    # 数据初始化
    """
    image_folder: 图片文件夹路径
    label_folder: 标签文件夹路径
    transform: 图片预处理方法
    label_transform: 标签预处理方法
    """
    def __init__(self,image_folder,label_folder,transform,label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder) # 获取图片文件夹下的所有图片文件名，如new1.png,new2.png,new3.png
        self.classes_list = ["no helmet","motor","number","with helmet"] # 定义类别列表
    def __len__(self):
         # 返回数据集的大小
        return len(self.image_names)

    def __getitem__(self, index):
        # 按索引返回数据和标签
        img_name = self.image_names[index]
        img_path = os.path.join(image_folder,img_name)
        image = Image.open(img_path).convert("RGB")
        # new1.png -> new1.xml
        # new1.png -> [new1, png] -> new1 + ".xml"
        label_name = img_name.split('.')[0] + '.xml' # [0]表示取列表的第一个元素new
        label_path = os.path.join(label_folder,label_name)

        # 读取文件内容
        with open(label_path,'r',encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)  # 将xml文件内容转换为字典
        objects = label_dict["annotation"]["object"]  # 获取目标对象信息
        target = []  # 初始化目标列表

        for object in objects:
            object_name = object["name"]  # 获取目标对象的类别名称
            object_class_id = self.classes_list.index(object_name)  # 获取目标对象的类别索引
            object_ymax = float(object["bndbox"]["ymax"])  # 获取目标对象
            object_xmax = float(object["bndbox"]["xmax"])  # 获取目标对象的边界框右下角x坐标
            object_xmin = float(object["bndbox"]["xmin"])  # 获取目标对象的边界框左上角x坐标
            object_ymin = float(object["bndbox"]["ymin"])  # 获取目标对象
            target.extend([object_class_id, object_xmin, object_ymin, object_xmax, object_ymax])  # 将目标对象信息添加到target列表中

        target = torch.tensor(target, dtype=torch.float32)  # 将目标列表转换为张量

        if self.transform is not None:
            image = self.transform(image)  # 对图片进行预处理
        return image, target


if __name__ == "__main__":
    # 测试代码
    image_folder = r"D:\pyStudy\yolo_Project\HelmetDataset-VOC\train\images"
    label_folder = r"D:\pyStudy\yolo_Project\HelmetDataset-VOC\train\labels"
    transform = transforms.ToTensor()  # 替换为实际的图片预处理方法
    label_transform = None  # 替换为实际的标签预处理方法

    train_dataset = VOCDataset(image_folder, label_folder, transform, label_transform)
    print(f"Dataset size: {len(train_dataset)}")
    print(f"First image: {train_dataset[0]}")