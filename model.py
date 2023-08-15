import torch
import torch.nn as nn
from torchvision.models import resnet34

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x);
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, type) -> None:
        super().__init__();

        if type == 'convtrans':
            self.net = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=1);
        else:
            self.net = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=1)
            )
        
        self.conv_after = nn.Sequential(
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1),
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1))
    
    def forward(self, x):
        x = self.net(x);
        x = self.conv_after(x);
        return x;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upblock(nn.Module):
    def __init__(self, in_features, out_features, concat_features = None) -> None:
        super().__init__();
        if concat_features == None:
            concat_features = out_features*2;

        self.upsample = Upsample(in_features, out_features, 4, 'convtrans');
        self.conv1 = ConvBlock(in_channels=concat_features, out_channels=out_features, kernel_size=3, stride=1);
        self.conv2 = ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1);

    def forward(self, x1, x2):
        x1 = self.upsample(x1);
        ct = torch.cat([x1,x2], dim=1);
        ct = self.conv1(ct);
        out = self.conv2(ct);
        return out;
#---------------------------------------------------------------

class SAM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs);
        resnet = resnet34(pretrained= True);
        ckpt = torch.load('resnet34.pth');
        resnet.load_state_dict(ckpt);
        self.input_blocks = nn.Sequential(*list(resnet.children()))[:3];
        resnet_down_blocks = [];
        for btlnck in list(resnet.children()):
            if isinstance(btlnck, nn.Sequential):
                resnet_down_blocks.append(btlnck);
    
        self.down_blocks = nn.Sequential(*resnet_down_blocks);
    
        self.bottle_neck = ConvBlock(256,128,1,1);
        self.global_embedding = nn.Sequential(
                ConvBlock(512, 128, 1, 1),
                ConvBlock(128,128,3,1)
            );
    
        self.up_1 = Upblock(128,128);
        self.up_2 = Upblock(128, 128, 128+64);
    
        self.local_embedding = ConvBlock(128, 128, 3, 1);
    
    def forward(self, x):
        d_1 = self.input_blocks(x);
        d_2 = self.down_blocks[0](d_1);
        d_3 = self.down_blocks[1](d_2);
        d_4 = self.down_blocks[2](d_3);
        d_5 = self.down_blocks[3](d_4);

        global_embedding = self.global_embedding(d_5);
        global_embedding = nn.functional.normalize(global_embedding, 2, dim = 1);

        d_4 = self.bottle_neck(d_4);
        u_1 = self.up_1(d_4, d_3);
        u_2 = self.up_2(u_1, d_2);
        local_embedding = self.local_embedding(u_2);
        local_embedding = nn.functional.normalize(local_embedding, 2, dim = 1);

        return global_embedding, local_embedding;


def test():
    inp = torch.randn((1, 3, 400, 400));
    le, ge = SAM()(inp);
    print(ge.shape);


if __name__ == "__main__":
    test();