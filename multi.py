import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import numpy as np
import scipy.io
from PIL import Image
from sppnet import SPPNet

image_path = './data/jpg'
label_path = './data/imagelabels.mat'
setid_path = './data/setid.mat'
save_path = './data/model_multi.pth'
BATCH = 32
EPOCH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./log_multi')


class MyDataset(Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, image_path, label_path, setid_path, train=True, transform=None):
        """
        image_00001.jpg
        image_00002.jpg
        image_00003.jpg
        ......
        """
        setid = scipy.io.loadmat(setid_path)
        labels = scipy.io.loadmat(label_path)['labels'][0]
        if train:
            trnid = setid['tstid'][0]
            self.labels = [labels[i - 1] - 1 for i in trnid]
            self.images = ['%s/image_%05d.jpg' % (image_path, i) for i in trnid]
        else:
            tstid = np.append(setid['valid'][0], setid['trnid'][0])
            self.labels = [labels[i - 1] - 1 for i in tstid]
            self.images = ['%s/image_%05d.jpg' % (image_path, i) for i in tstid]
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(Image.open(image))
        return image, label

    def __len__(self):
        return len(self.labels)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if (batch_idx + 1) % 20 == 0:
            train_loss /= 20

            print('Train Epoch: %d [%d/%d (%.4f%%)]\tLoss: %.4f' % (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), train_loss))
            train_loss = 0


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    total_true = 0
    total_loss = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)

            output = model(image)
            loss = criterion(output, label)

            pred = torch.max(output, 1)[1]  # get the index of the max log-probability
            total_true += (pred.view(label.size()).data == label.data).sum().item()
            total_loss += loss.item()

    accuracy = total_true / len(test_loader.dataset)
    loss = total_loss / len(test_loader.dataset)
    print('\nTest Epoch: %d ====> Accuracy: [%d/%d (%.4f%%)]\tAverage loss: %.4f\n' % (
        epoch, total_true, len(test_loader.dataset), 100. * accuracy, loss))
    writer.add_scalar('accuracy', accuracy, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_image('image', image.cpu(), epoch)


def load(size):
    train_dataset = MyDataset(image_path, label_path, setid_path,
                              train=True, transform=
                              transforms.Compose([
                                  transforms.Resize((size, size)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))

    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    print('Train size:', len(train_loader))

    test_dataset = MyDataset(image_path, label_path, setid_path,
                             train=False, transform=
                             transforms.Compose([
                                 transforms.Resize((size, size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader_350, test_loader_350 = load(350)
    train_loader_400, test_loader_400 = load(400)
    train_loader_450, test_loader_450 = load(450)
    train_loader_500, test_loader_500 = load(500)
    train_loaders = [train_loader_350, train_loader_400, train_loader_450, train_loader_500]
    test_loaders = [test_loader_350, test_loader_400, test_loader_450, test_loader_500]

    model = SPPNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH + 1):
        for train_loader, test_loader in zip(train_loaders, test_loaders):
            train(model, device, train_loader, criterion, optimizer, epoch)
            test(model, device, test_loader, criterion, epoch)

    torch.save(model, save_path)
