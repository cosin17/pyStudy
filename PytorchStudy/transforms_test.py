from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import image_path

# python的用法 -》 tensor数据类型
# 通过transforms.ToTensor去看两个问题
# 2. 为什么我们需要Tensor数据类型

# 绝对路径 C:\Users\Cosin17\Desktop\pyStudy\hymenoptera_data\train\ants\0013035.jpg
# 相对路径 hymenoptera_data/train/ants/0013035.jpg
img_path = "../hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

# 1. transforms该如何使用
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()

