from torch.utils.data import Dataset

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