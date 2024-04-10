import torch
from torch import nn
from torchvision import models
from utils import setup_seed


ngf = 64


def Resnet18(pretrained=False, class_num=1000):
    model = models.resnet18(pretrained=pretrained)
    model.avgpool = nn.Sequential()
    model.fc = nn.Linear(512 * 7 * 7, class_num)
    return model


def Resnet50(pretrained=False, class_num=1000):
    model = models.resnet50(pretrained=pretrained)
    model.avgpool = nn.Sequential()
    model.fc = nn.Linear(2048 * 7 * 7, class_num)
    return model


def vgg16(pretrained=False, class_num=1000):
    model = models.vgg16(pretrained=pretrained)
    model.fc = nn.Linear(2048 * 7 * 7, class_num)
    return model


def chooseAttackedModel(modelName="resnet18", pretrained=False, class_num=1000):
    setup_seed(1234)
    if modelName=="resnet18":
        return Resnet18(pretrained=pretrained, class_num=class_num)
    elif modelName=="resnet50":
        return Resnet50(pretrained=pretrained, class_num=class_num)
    elif modelName=="vgg16":
        return vgg16(pretrained=pretrained, class_num=class_num)
    else:
        exit("wrong attacked model")


class Generator336(nn.Module):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    def __init__(self):
        super(Generator336, self).__init__()

        # self.initial = nn.Sequential(
        #     nn.Linear(11 * 11, 4 * 4),  # Resize to match 6x6 feature maps with 1024 channels
        #     nn.LeakyReLU()
        # )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),  # 14 * 14
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),

            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 28 * 28
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=2, output_padding=1),  # 56 * 56
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=2, output_padding=1),  # 112 * 112
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=3, output_padding=1),  # 224 * 224
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # out = x.view(-1, 11 * 11)
        # out = self.initial(out)
        out = x.view(-1, 2048, 11, 11)
        out = self.conv(out)
        return out


class Generator(nn.Module):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    def __init__(self):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),  # 14 * 14
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),

            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 28 * 28
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 56 * 56
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 112 * 112
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 224 * 224
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 2048, 7, 7)
        out = self.conv(out)
        return out


class Generator2(nn.Module):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    def __init__(self, hidden):
        super(Generator2, self).__init__()
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden//2, 3, stride=2, padding=1, output_padding=1),  # 2 * 2
            nn.LeakyReLU(),
            self.ResConv(hidden//2),
            self.ResConv(hidden//2),
            self.ResConv(hidden//2),

            nn.ConvTranspose2d(hidden//2, hidden//4, 3, stride=2, padding=1, output_padding=1),  # 4 * 4
            nn.LeakyReLU(),
            self.ResConv(hidden//4),
            self.ResConv(hidden//4),
            self.ResConv(hidden//4),

            nn.ConvTranspose2d(hidden//4, hidden//8, 3, stride=2, padding=1, output_padding=1),  # 8 * 8
            nn.LeakyReLU(),
            self.ResConv(hidden//8),
            self.ResConv(hidden//8),
            self.ResConv(hidden//8),

            # nn.ConvTranspose2d(hidden//8, hidden//16, 3, stride=2, padding=1, output_padding=1),  # 112 * 112
            # nn.LeakyReLU(),
            # self.ResConv(hidden//16),
            # self.ResConv(hidden//16),
            # self.ResConv(hidden//16),
            #
            # nn.ConvTranspose2d(hidden//16, hidden//32, 3, stride=2, padding=1, output_padding=1),  # 224 * 224
            # nn.LeakyReLU(),
            # self.ResConv(hidden//32),
            # self.ResConv(hidden//32),
            # self.ResConv(hidden//32),

            nn.Conv2d(hidden//8, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, self.hidden * 7 * 7, 1, 1)
        out = self.conv(out)
        return out


if __name__ == "__main__":
    chooseAttackedModel("resnet18")

