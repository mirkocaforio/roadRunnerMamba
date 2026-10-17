import torch
import torch.nn as nn


class PreScanFGMask(nn.Module):
    def __init__(self, d_inner, k_ratio=0.5):
        super().__init__()
        # proiettore leggero: niente bias, output scalare per token
        self.proj = nn.Linear(d_inner, 1, bias=False)
        self.k_ratio = k_ratio  # frazione di token foreground

    def forward(self, x):
        # x: [B, C, L] — come arriva prima dello scan
        # trasponiamo per il linear: [B, L, C]
        B, C, L = x.shape
        
        scores = self.proj(x.transpose(1, 2)).squeeze(-1)  # [B, L]
        scores = torch.sigmoid(scores)
        
        K = max(1, int(L * self.k_ratio))

        # Trova il valore soglia (K-esimo score più alto).
        # topk è più veloce di kthvalue su GPU: la soglia è il più piccolo dei top-K.
        threshold = scores.topk(K, dim=-1).values[:, -1:].contiguous()  # [B, 1]
        
        # Maschera continua: foreground (score >= threshold) → peso 1.0
        # background (score < threshold) → peso 0.1
        # Tutto differenziabile, niente in-place
        fg_weight = (scores >= threshold).to(scores.dtype)  # [B, L] binaria (dtype = autocast)
        mask = fg_weight + (1 - fg_weight) * 0.1           # fg=1.0, bg=0.1
        
        # top-K come foreground, resto background
        #K = max(1, int(x.shape[-1] * self.k_ratio))
        #_, fg_idx = scores.topk(K, dim=-1)
        
        # mask continua: foreground=1, background in [0,1]
        #mask = scores.clone()
        # i top-K rimangono al loro valore sigmoid
        # il resto viene scalato verso 0
        #bg_mask = torch.ones_like(scores, dtype=torch.bool)
        #bg_mask.scatter_(1, fg_idx, False)
        #mask[bg_mask] = mask[bg_mask] * 0.1
        
        # [B, L]
        return mask.unsqueeze(1) # [B, 1, L]

"""
class ForegroundAttention(nn.Module):
    #Spatial attention per foreground/background separation
    def __init__(self, dim, top_k_ratio=0.25):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.dim = dim
        
        # Attention score predictor
        self.score_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        
        print("ForegroundAttention input shape:", x.shape)
        print("ForegroundAttention dim:", self.dim)
        print("ForegroundAttention score_net:", self.score_net)
        
        # Compute attention scores
        scores = self.score_net(x).squeeze(-1)  # [B, H, W]
        
        # Select top-K foreground regions
        K = max(1, int(H * W * self.top_k_ratio))  # Ensure K >= 1
        scores_flat = scores.view(B, -1)
        top_k_values, top_k_indices = torch.topk(scores_flat, K, dim=1)
        
        # Create soft attention mask
        attention_mask = torch.zeros_like(scores_flat)
        attention_mask.scatter_(1, top_k_indices, top_k_values)
        attention_mask = attention_mask.view(B, H, W, 1)
        
        # Apply attention
        return x * (1 + attention_mask)  # Amplify foreground
"""