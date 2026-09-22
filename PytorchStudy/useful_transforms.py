from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from test_tb import writer

writer = SummaryWriter("../logs")
img = Image.open("../images/Jay.jpg")

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("Tensor_img", tensor_img)

# Normalize
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(tensor_img)
writer.add_image("Normalize", img_norm,1)


# Resize
trans_resize = transforms.Resize([512, 512])
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> ToTensor -> img_resize Tensor
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize,0)

# Compose - resize -> 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans]) # 先resize再ToTensor
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2,1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Crop", img_crop,1)



writer.close()
