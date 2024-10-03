import torch
import torch.nn as nn

class STGC_attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout, residual, A_size):    
        super().__init__()
        self.num_attention_matrices = A_size[0]
        self.spatial_kernel_size = A_size[0] * 2

        self.sgc_conv = nn.Conv2d(in_channels, out_channels * self.spatial_kernel_size, kernel_size=1)
        self.M = nn.Parameter(torch.ones(A_size))
        
        self.tgc = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)), #padding=(4, 0) is equivalent to ((9-1)//2, 0)
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, A, A_attention):
 
        res=x
        x = self.sgc_conv(x)       
        N, KC, T, V = x.size()
        x = x.view(N, self.spatial_kernel_size, KC // self.spatial_kernel_size, T, V)
        x1 = x[:, :self.spatial_kernel_size - self.num_attention_matrices, :, :, :]
        x2 = x[:, -self.num_attention_matrices:, :, :, :]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A * self.M))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, A_attention))
        x_sum = x1 + x2
        x = self.tgc(x_sum) + self.residual(res)
        return x
 