import torch
from torchvision import transforms

from PIL import Image
from sppnet import SPPNet

model_path = './data/model_single.pth'
# model_path = './data/model_multi.pth'
image_path = './data/test.jpg'

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(Image.open(image_path))

    # the first way to load model
    # model = torch.load(model_path)
    # model = model.cpu()

    # the second way to load model
    model = SPPNet()
    model.load_state_dict(torch.load(model_path))
    model = model.cpu()

    output = model(image.unsqueeze(0))
    pred = torch.max(output, 1)[1]  # get the index of the max log-probability

    print(int(pred))
