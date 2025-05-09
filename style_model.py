import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self._gram_matrix(target_feature).detach()
        
    def forward(self, x):
        G = self._gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x
        
    def _gram_matrix(self, x):
        batch_size, n_channels, height, width = x.size()
        features = x.view(batch_size * n_channels, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * n_channels * height * width)

class StyleTransferModel(nn.Module):
    def __init__(self):
        super(StyleTransferModel, self).__init__()
        # Use a pre-trained VGG model
        vgg = models.vgg16(pretrained=True).features.eval()
        
        # Extract key layers for style and content features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for i in range(4):  # First conv block
            self.slice1.add_module(str(i), vgg[i])
        for i in range(4, 9):  # Second conv block
            self.slice2.add_module(str(i), vgg[i])
        for i in range(9, 16):  # Third conv block
            self.slice3.add_module(str(i), vgg[i])
        for i in range(16, 23):  # Fourth conv block
            self.slice4.add_module(str(i), vgg[i])
            
        # Freeze model parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Transformation network
        self.transform = nn.Sequential(
            # Initial convolution layers
            nn.Conv2d(3, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            
            # Upsampling
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Transform input and normalize output to [0, 255]
        transformed = self.transform(x)
        return (transformed + 1) * 127.5
        
    def extract_features(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class InferenceModel(nn.Module):
    """Lightweight model for inference only"""
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x):
        return (self.model(x) + 1) * 127.5

def create_style_model(pretrained=False, path=None):
    """Create a style transfer model"""
    model = StyleTransferModel()
    
    if pretrained and path is not None:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    
    return model

def create_inference_model(path=None):
    """Create a lightweight inference model"""
    model = InferenceModel()
    
    if path is not None:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    
    return model