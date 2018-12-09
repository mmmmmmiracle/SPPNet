import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import scipy.io
from PIL import Image
from sppnet import SPPNet

image_path = './data/jpg'
label_path = './data/imagelabels.mat'
setid_path = './data/setid.mat'
BATCH = 1
EPOCH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            self.labels = torch.tensor([labels[i - 1] for i in trnid], dtype=torch.int64)
            self.images = ['%s/image_%05d.jpg' % (image_path, i) for i in trnid]
        else:
            tstid = np.append(setid['valid'][0], setid['trnid'][0])
            self.labels = torch.tensor([labels[i - 1] for i in tstid], dtype=torch.int64)
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
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                       (batch_idx + 1) * len(image) / len(train_loader.dataset) * 100., loss.item()))


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)

            output = model(image)
            test_loss += criterion(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Accuracy: [{}/{} ({:.0f}%)]\tAverage loss: {:.4f},'.format(
        correct, len(test_loader.dataset), correct / len(test_loader.dataset) * 100., test_loss))


if __name__ == '__main__':
    train_dataset = MyDataset(image_path, label_path, setid_path,
                              train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    print('Train size:', len(train_loader))

    test_dataset = MyDataset(image_path, label_path, setid_path,
                             train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=True)
    print('Test size:', len(test_loader))

    model = SPPNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader, criterion)
