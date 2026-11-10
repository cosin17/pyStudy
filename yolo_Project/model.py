from torch import nn
import torch
from torchvision import transforms
import torch.nn.functional as F


from yolo_dataset import YOLODataset

class keWeiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

if __name__ == "__main__":
    model = keWeiModel()
    image_folder = r"D:\pyStudy\yolo_Project\HelmetDataset-YOLO-Train\images"
    label_folder = r"D:\pyStudy\yolo_Project\HelmetDataset-YOLO-Train\labels"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512,512))])  # 替换为实际的图片预处理方法
    label_transform = None  # 替换为实际的标签预处理方法
    dataset = YOLODataset(image_folder, label_folder, transform, label_transform)
    image,target = dataset[0]
    output = model(image)
    # print(output)
    torch.onnx.export(model, image, "keWeiModel.onnx") # 导出为ONNX模型