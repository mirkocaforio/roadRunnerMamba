import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

from core.model.bloks.spatial_mamba import SpatialMambaBlock

class RscdLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        use_fg_mask=True,
        stage_idx=0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                stage_idx=stage_idx,
                use_fg_mask=use_fg_mask,
            )
            for i in range(depth)])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
