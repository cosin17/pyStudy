from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("../logs")   # 创建SummaryWriter对象，日志保存到【logs文件夹】
image_path = "../hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)    # PIL图片转numpy数组，shape是 [H,W,C] 高、宽、通道


writer.add_image("test", img_array, 2, dataformats="HWC")
writer.close()
writer = SummaryWriter("../logs")
# y = x
for i in range(100):# i从0-99
    writer.add_scalar("y=2x", 2*i, i)

writer.close()