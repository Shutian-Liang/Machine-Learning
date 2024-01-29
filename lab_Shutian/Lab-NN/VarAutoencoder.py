import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
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
        
        self.mu = nn.Linear(128*6*6,encoding_dim)
        self.var = nn.Linear(128*6*6,encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x3.view(x3.size(0),-1)
        mu = self.mu(x4)
        log_var = self.var(x4)
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder
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
        

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the reparameterization trick to sample v
        epsilon = torch.randn_like(mu)
        sigma = torch.exp(log_var * 0.5)
        v = mu+epsilon*sigma
        return v
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu,log_var = self.encoder(x)
        v = self.reparameterize(mu,log_var)
        x = self.decoder(v)
        return x, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    
    x, mu, log_var = outputs[0],outputs[1],outputs[2]
    #beta = 0.01
    #samples = images.size(0)
    CE = torch.sum((x-images)**2)
    KL = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
    return CE+KL
