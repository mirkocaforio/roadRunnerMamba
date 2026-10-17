import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class EinFFT(nn.Module):
    """
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim  # 768
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        
        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale
        )
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale
        )
    
    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"N ({N}) deve essere uguale a H*W ({H*W})"
        
        # Reshape da (B, N, C) a (B, H, W, C) poi a (B, H, W, num_blocks, block_size)
        x = x.view(B, H, W, self.num_blocks, self.block_size)
        
        # FFT 2D sulle dimensioni spaziali H e W
        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')
        
        # Primo layer complesso con ReLU
        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - 
            self.multiply(x.imag, self.complex_weight_1[1]) + 
            self.complex_bias_1[0]
        )
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + 
            self.multiply(x.imag, self.complex_weight_1[0]) + 
            self.complex_bias_1[1]
        )
        
        # Secondo layer complesso senza ReLU
        x_real_2 = (
            self.multiply(x_real_1, self.complex_weight_2[0]) - 
            self.multiply(x_imag_1, self.complex_weight_2[1]) + 
            self.complex_bias_2[0]
        )
        x_imag_2 = (
            self.multiply(x_real_1, self.complex_weight_2[1]) + 
            self.multiply(x_imag_1, self.complex_weight_2[0]) + 
            self.complex_bias_2[1]
        )
        
        # Stack e applica sparsity
        x = torch.stack([x_real_2, x_imag_2], dim=-1)
        if self.sparsity_threshold:
            x = F.softshrink(x, lambd=self.sparsity_threshold)
        
        # Converti a complesso
        x = torch.view_as_complex(x)
        
        # IFFT 2D sulle dimensioni spaziali H e W
        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")
        
        # Converti da complesso a reale (prende solo la parte reale)
        x = x.real
        
        # Reshape finale da (B, H, W, num_blocks, block_size) a (B, N, C)
        x = x.reshape(B, N, C)
        
        return x
    """
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        
        # Usa direttamente parametri complessi
        self.complex_weight_1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, self.block_size, 2) * self.scale
            )
        )
        self.complex_weight_2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, self.block_size, 2) * self.scale
            )
        )
        self.complex_bias_1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, 2) * self.scale
            )
        )
        self.complex_bias_2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, 2) * self.scale
            )
        )
    
    def complex_matmul_bias(self, x, weight, bias):
        # x: [..., num_blocks, block_size]
        # weight: [num_blocks, block_size, block_size]
        # bias: [num_blocks, block_size]
        return torch.einsum('...bk,bkd->...bd', x, weight) + bias
    
    def complex_relu(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Reshape e converti a complesso una sola volta
        x = x.view(B, N, self.num_blocks, self.block_size)
        
        # FFT - lavora direttamente con float, poi converti
        x = torch.fft.fft2(x.to(torch.float32), dim=(1, 2), norm='ortho')
        
        # Layer 1 con ReLU
        x = self.complex_matmul_bias(x, self.complex_weight_1, self.complex_bias_1)
        x = self.complex_relu(x)
        
        # Layer 2
        x = self.complex_matmul_bias(x, self.complex_weight_2, self.complex_bias_2)
        
        # Sparsity (evita il branch con una moltiplicazione)
        if self.training and self.sparsity_threshold > 0:
            # Soft thresholding su parte reale e immaginaria
            x_real = F.softshrink(x.real, lambd=self.sparsity_threshold)
            x_imag = F.softshrink(x.imag, lambd=self.sparsity_threshold)
            x = torch.complex(x_real, x_imag)
        
        # IFFT
        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")
        
        # Converti a float e reshape
        x = x.real.reshape(B, N, C)
        
        return x
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EinFFTOptimized(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        
        # Usa direttamente parametri complessi
        self.complex_weight_1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, self.block_size, 2) * self.scale
            )
        )
        self.complex_weight_2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, self.block_size, 2) * self.scale
            )
        )
        self.complex_bias_1 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, 2) * self.scale
            )
        )
        self.complex_bias_2 = nn.Parameter(
            torch.view_as_complex(
                torch.randn(self.num_blocks, self.block_size, 2) * self.scale
            )
        )
    
    def complex_matmul_bias(self, x, weight, bias):
        # x: [..., num_blocks, block_size]
        # weight: [num_blocks, block_size, block_size]
        # bias: [num_blocks, block_size]
        return torch.einsum('...bk,bkd->...bd', x, weight) + bias
    
    def complex_relu(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Reshape e converti a complesso una sola volta
        x = x.view(B, N, self.num_blocks, self.block_size)
        
        # FFT - lavora direttamente con float, poi converti
        x = torch.fft.fft2(x.to(torch.float32), dim=(1, 2), norm='ortho')
        
        # Layer 1 con ReLU
        x = self.complex_matmul_bias(x, self.complex_weight_1, self.complex_bias_1)
        x = self.complex_relu(x)
        
        # Layer 2
        x = self.complex_matmul_bias(x, self.complex_weight_2, self.complex_bias_2)
        
        # Sparsity (evita il branch con una moltiplicazione)
        if self.training and self.sparsity_threshold > 0:
            # Soft thresholding su parte reale e immaginaria
            x_real = F.softshrink(x.real, lambd=self.sparsity_threshold)
            x_imag = F.softshrink(x.imag, lambd=self.sparsity_threshold)
            x = torch.complex(x_real, x_imag)
        
        # IFFT
        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")
        
        # Converti a float e reshape
        x = x.real.reshape(B, N, C)
        
        return x


# Versione ancora più ottimizzata usando operazioni fuse
class EinFFTFused(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        
        # Combina i pesi in un'unica matrice per ridurre le operazioni
        # Shape: [2, num_blocks, block_size, block_size]
        self.weight_real = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size) * 0.02)
        self.bias_real = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size) * 0.02)
        self.bias_imag = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size) * 0.02)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)
        
        # FFT
        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')
        
        # Layer 1 - operazioni fuse
        x_r, x_i = x.real, x.imag
        
        # Pattern einsum corretto usando lettere diverse:
        # B=batch, N=sequence, M=num_blocks, K=block_size_in, D=block_size_out
        # x_r/x_i shape: [B, N, M, K]
        # weights shape: [M, K, D]
        # output shape: [B, N, M, D]
        out_r1 = torch.einsum('bnmk,mkd->bnmd', x_r, self.weight_real[0]) - \
                 torch.einsum('bnmk,mkd->bnmd', x_i, self.weight_imag[0]) + self.bias_real[0]
        out_i1 = torch.einsum('bnmk,mkd->bnmd', x_r, self.weight_imag[0]) + \
                 torch.einsum('bnmk,mkd->bnmd', x_i, self.weight_real[0]) + self.bias_imag[0]
        
        # ReLU
        out_r1 = F.relu(out_r1)
        out_i1 = F.relu(out_i1)
        
        # Layer 2
        out_r2 = torch.einsum('bnmk,mkd->bnmd', out_r1, self.weight_real[1]) - \
                 torch.einsum('bnmk,mkd->bnmd', out_i1, self.weight_imag[1]) + self.bias_real[1]
        out_i2 = torch.einsum('bnmk,mkd->bnmd', out_r1, self.weight_imag[1]) + \
                 torch.einsum('bnmk,mkd->bnmd', out_i1, self.weight_real[1]) + self.bias_imag[1]
        
        # Sparsity
        if self.training and self.sparsity_threshold > 0:
            out_r2 = F.softshrink(out_r2, lambd=self.sparsity_threshold)
            out_i2 = F.softshrink(out_i2, lambd=self.sparsity_threshold)
        
        # Ricombina e IFFT
        x = torch.complex(out_r2, out_i2)
        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")
        x = x.real.reshape(B, N, C)
        
        return x
"""