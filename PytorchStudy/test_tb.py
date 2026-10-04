from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")   # 创建SummaryWriter对象，日志保存到当前目录下的logs文件夹
image_path = "../hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)    # PIL图片转numpy数组，shape是 [H,W,C] 高、宽、通道

writer.add_image("test", img_array, 1, dataformats="HWC")

# y = x
for i in range(100):  # i从0-99
    # add_scalar() 方法添加标量值
    # 第一个参数是标签名，第二个参数是值，第三个参数是步长
    writer.add_scalar("y=2x", 2*i, i)

writer.close()