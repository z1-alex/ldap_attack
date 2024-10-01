import PIL.Image as Image
from torchvision import transforms


def open_img(img_path):
    a = Image.open(img_path).convert('RGB')
    b = transforms.ToTensor()(a)
    return b
