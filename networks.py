import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, BasicBlock

NUM_TARGETS = 2
NUM_CLASSES = 10


class MLP(nn.Module):
    """
    Multilayer Perceptron
    """

    def __init__(self):
        super(MLP, self).__init__()
        self.__name__ = "MLP"
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14 * 14 * 2, 64),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, NUM_TARGETS)
        )

    def forward(self, x):
        out = self.layers(x)

        predicted = torch.argmax(out.data, 1)

        return predicted, out


class ConvNet(nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.__name__ = "ConvNet"
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn1 = nn.BatchNorm2d(16)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2 * 2 * 64, 100)
        self.fc2 = nn.Linear(100, NUM_TARGETS)

    def forward(self, x):
        out = self.bn1(self.layer1(x))
        out = self.bn2(self.layer2(out))
        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        predicted = torch.argmax(out.data, 1)

        return predicted, out


class ConvTransposeNet(nn.Module):
    """
    Convolutional Transpose Neural Network
    """

    def __init__(self):
        super(ConvTransposeNet, self).__init__()
        self.__name__ = "ConvTransposeNet"
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(2, 16, kernel_size=4, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(288, NUM_TARGETS)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)

        predicted = torch.argmax(out.data, 1)

        return predicted, out


class ConvSiameseNetClasses(nn.Module):
    """
    Convolutional Siamese Neural Network that takes the digit classes into account
    """

    def __init__(self):
        super(ConvSiameseNetClasses, self).__init__()
        self.__name__ = "ConvSiameseNetClasses"
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn1 = nn.BatchNorm2d(8)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn2 = nn.BatchNorm2d(16)
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 4 * 16, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)

    def forward_once(self, x):
        out = self.bn1(self.cnn1(x))
        out = self.bn2(self.cnn2(out))
        out = out.reshape(out.size()[0], -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def forward(self, x):
        input1 = x[:, :1]
        input2 = x[:, 1:]
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        class_pred1 = torch.argmax(out1.data, dim=1)
        class_pred2 = torch.argmax(out2.data, dim=1)

        return torch.le(class_pred1, class_pred2).long(), out1, out2


class ConvSiameseNetTargets(nn.Module):
    """
    Convolutional Siamese Neural Network that only learns on the targets 0/1
    """

    def __init__(self):
        super(ConvSiameseNetTargets, self).__init__()
        self.__name__ = "ConvSiameseNetTargets"
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn1 = nn.BatchNorm2d(8)
        self.cnn2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.SELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.bn2 = nn.BatchNorm2d(16)
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 4 * 16, 100)
        self.fc2 = nn.Linear(100, NUM_CLASSES)
        self.fc3 = nn.Linear(2 * NUM_CLASSES, NUM_TARGETS)

    def forward_once(self, x):
        out = self.bn1(self.cnn1(x))
        out = self.bn2(self.cnn2(out))
        out = out.reshape(out.size()[0], -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def forward(self, x):
        input1 = x[:, :1]
        input2 = x[:, 1:]
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        out = self.fc3(torch.hstack([out1, out2]))

        predicted = torch.argmax(out.data, 1)

        return predicted, out


class ResConvNetBlock(nn.Module):
    """
    Blocks for Convolutional Residual Network
    """

    def __init__(self, nb_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
                               padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        y = self.bn1(F.selu(self.conv1(x)))
        y = self.conv2(y)
        y += x
        y = F.selu(y)
        y = self.bn2(y)

        return y


class ConvResNet(nn.Module):
    """
    Convolutional Residual Neural Network
    """

    def __init__(self, nb_channels, kernel_size, nb_blocks):
        super().__init__()
        self.__name__ = "ConvResNet"
        self.conv0 = nn.Conv2d(2, nb_channels, kernel_size=(1, 1))
        self.resblocks = nn.Sequential(*(ResConvNetBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(nb_channels, NUM_TARGETS)

    def forward(self, x):
        out = F.selu(self.conv0(x))
        out = self.resblocks(out)
        out = F.selu(self.avg(out))
        out = out.reshape(out.size()[0], -1)
        out = self.fc(out)

        predicted = torch.argmax(out.data, 1)

        return predicted, out


class ConvSiameseResNet(nn.Module):
    """
    Convolutional Siamese Residual Neural Network
    """

    def __init__(self, nb_channels, kernel_size, nb_blocks):
        super().__init__()
        self.__name__ = "ConvSiameseResNet"
        self.conv0 = nn.Conv2d(1, nb_channels, kernel_size=(1, 1))
        self.resblocks = nn.Sequential(*(ResConvNetBlock(nb_channels, kernel_size) for _ in range(nb_blocks)))
        self.avg = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(nb_channels, NUM_CLASSES)

    def forward_once(self, x):
        out = F.selu(self.conv0(x))
        out = self.resblocks(out)
        out = F.selu(self.avg(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out

    def forward(self, x):
        input1 = x[:, :1]
        input2 = x[:, 1:]
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        class_pred1 = torch.argmax(out1.data, dim=1)
        class_pred2 = torch.argmax(out2.data, dim=1)

        return torch.le(class_pred1, class_pred2).long(), out1, out2


class MNISTResNet(ResNet):
    """
    Pretrained Residual Neural Network ResNet18
    """

    def __init__(self):
        super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=2)  # Based on ResNet18
        self.__name__ = "MNISTResNet"
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.fc = nn.Linear(512, NUM_TARGETS, bias=True)

    def forward(self, x):
        out = super().forward(x)
        predicted = torch.argmax(out.data, 1)

        return predicted, out


class MNISTResSiameseNet(ResNet):
    """
    Pretrained Residual Siamese Neural Network ResNet18
    """

    def __init__(self):
        super(MNISTResSiameseNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES)  # Based on ResNet18
        self.__name__ = "MNISTResSiameseNet"
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(4096, NUM_CLASSES)

    def forward_once(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size()[0], -1)
        out = self.fc(out)

        return out

    def forward(self, x):
        input1 = x[:, :1]
        input2 = x[:, 1:]
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)

        class_pred1 = torch.argmax(out1.data, dim=1)
        class_pred2 = torch.argmax(out2.data, dim=1)

        return torch.le(class_pred1, class_pred2).long(), out1, out2
