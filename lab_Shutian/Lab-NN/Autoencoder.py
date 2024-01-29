import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
# H = 24; W = 24
# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the encoder
        self.conv1 = nn.Sequential(
            # 输入[3,24,24]
            nn.Conv2d(
                in_channels=3,    # 输入图片的高度
                out_channels=32,  # 输出图片的高度
                kernel_size=3,    # 3*3的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=1,        # (kernel_size-1)/2
            ),
            # 经过卷积层 输出[32,H,W] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[32,H/2,W/2] 传入下一个卷积
        )
        
        self.conv2 = nn.Sequential(
            # 输入[32,H/2,W/2] 
            nn.Conv2d(
                in_channels=32,   # 输入图片的高度
                out_channels=64,  # 输出图片的高度
                kernel_size=3,    # 3*3的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=1,        # (kernel_size-1)/2
            ),
            # 经过卷积层 输出[64,H/2,W/2]传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # 经过池化 输出[64,H/4,W/4] 传入下一个卷积
        )
        
        self.conv3 = nn.Sequential(
            # 输入[64,H/4,W/4] 
            nn.Conv2d(
                in_channels=64,    # 输入图片的高度
                out_channels=128,  # 输出图片的高度
                kernel_size=3,    # 3*3的卷积核，相当于过滤器
                stride=1,         # 卷积核在图上滑动，每隔一个扫一次
                padding=1,        # (kernel_size-1)/2
            ),
            nn.ReLU(),
        )
        
        self.out = nn.Linear(128*6*6,encoding_dim)
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x3.view(x3.size(0),-1)
        v = self.out(x4)
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the decoder
        self.fc = nn.Linear(encoding_dim,128*6*6)
        self.convtrans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.convtrans2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.convtrans3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=2,stride=2),
            nn.ReLU()
        )
        
        
    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        
        # TODO: implement the forward pass
        v = self.fc(v)
        v = v.view(v.size(0), 128, 6, 6)  # reshape to (batch_size, 128, 6, 6)

        v1 = self.convtrans1(v)
        v2 = self.convtrans2(v1)
        x = self.convtrans3(v2)
        
        return x


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"

