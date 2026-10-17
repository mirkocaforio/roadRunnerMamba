import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
from functools import partial
import copy
import math

from core.model.layers.rscdLayer import RscdLayer

from .bloks.stem import Stem
from .bloks.downsampling import DownSampling
from .bloks.condition_branch import ConditionBranch

try:
    from ..utility.selective_scan import selective_scan_state_flop_jit, selective_scan_fn
except:
    from utility.selective_scan import selective_scan_state_flop_jit, selective_scan_fn

class RSCDMamba(nn.Module):
    def __init__(
        self,
        img_size=224, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        d_state=1, 
        dt_init="random",
        mlp_ratio=4.0,
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, 
        patch_norm=True,
        use_checkpoint=False,
        use_fg_mask=True,
        use_condition_branch=True,
        num_conditions=6,          # 6 per RSCD completo
        condition_hidden_dim=None, # default: num_features // 4
        condition_dropout=0.1,

        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.use_fg_mask = use_fg_mask
        self.use_condition_branch = use_condition_branch

        self.patch_embed = Stem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim
        )

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RscdLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=DownSampling if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_fg_mask=use_fg_mask,
                stage_idx=i_layer,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        if self.use_condition_branch:
            self.condition_branch = ConditionBranch(
                d_model=self.num_features,
                num_conditions=num_conditions,
                hidden_dim=condition_hidden_dim,
                dropout=condition_dropout,
            )
        else:
            self.condition_branch = None
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanStateFn": selective_scan_state_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda()#.eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1, 2)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        if self.use_condition_branch:
            x = self.condition_branch(x)
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

"""
class Backbone_SpatialMamba(RSCDMamba):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):

        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None) 
        kwargs.update(norm_layer=norm_layer)       

        # add norm ========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        # modify layer ========================
        def layer_forward(self: RscdLayer, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            y = None
            if self.downsample is not None:
                y = self.downsample(x)

            return x, y

        for l in self.layers:
            l.forward = partial(layer_forward, l)

        # del self.classifier ===================
        del self.head
        del self.avgpool
        del self.norm

        self.load_pretrained(pretrained, key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x

        return outs
"""