"""MAE, ConvNeXt, and the alternative JEPA/FiLM architectures.
Sections: masking; MAE; ConvNeXt head; JEPA; FiLM/axial head.
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_Y, GRID_T = 31, 20 # how many patches in each dim
DEFAULT_STRIDES = ((2, 2), (2, 5), (2, 1), (2, 1)) #used in jepa

# Masking
def random_spatial_mask(batch_size, n_patches, mask_ratio, generator=None):
    """(B, L) mask: False is visible; keep at least one patch per sample."""
    len_keep = int(round(n_patches * (1.0 - mask_ratio))) # patchs kept
    len_keep = max(1, min(n_patches, len_keep))
    noise = torch.rand(batch_size, n_patches, generator=generator) #(batch_size, n_patches)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones(batch_size, n_patches, dtype=torch.bool)
    mask.scatter_(1, ids_keep, False) # False is non-masking
    return mask


# positional encoding
def _sincos_1d(dim, positions):
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
    out = np.einsum("p,d->pd", positions.astype(np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (P, dim)


def sincos_2d(dim, grid_y, grid_t):
    assert dim % 4 == 0
    ys, ts = np.meshgrid(np.arange(grid_y), np.arange(grid_t), indexing="ij")
    emb_y = _sincos_1d(dim // 2, ys.reshape(-1))
    emb_t = _sincos_1d(dim // 2, ts.reshape(-1))
    return torch.from_numpy(np.concatenate([emb_y, emb_t], axis=1)).float()  # (L, dim)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  #cfg=Config()
        V, L = cfg.n_vars, cfg.n_patches
        p = cfg.patch_y * cfg.patch_t
        self.V, self.L, self.p = V, L, p
        # V: num of var; L: num of patchs in var; p: num of points in a patch
        self.mask_ratio = cfg.mask_ratio

        # encoder
        self.patch_embed = nn.Conv2d(
            1,
            cfg.enc_dim,
            kernel_size=(cfg.patch_y, cfg.patch_t),
            stride=(cfg.patch_y, cfg.patch_t),
        )
        self.var_embed_enc = nn.Parameter(torch.zeros(V, cfg.enc_dim))
        self.register_buffer("pos_embed_enc", sincos_2d(cfg.enc_dim, cfg.grid_y, cfg.grid_t))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.enc_dim))
        self.enc_blocks = nn.ModuleList(
            [Block(cfg.enc_dim, cfg.enc_heads, cfg.mlp_ratio) for _ in range(cfg.enc_depth)]
        ) #transformer blocks
        self.enc_norm = nn.LayerNorm(cfg.enc_dim)

        # decoder
        self.dec_embed = nn.Linear(cfg.enc_dim, cfg.dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_dim))
        self.var_embed_dec = nn.Parameter(torch.zeros(V, cfg.dec_dim))
        self.register_buffer("pos_embed_dec", sincos_2d(cfg.dec_dim, cfg.grid_y, cfg.grid_t))
        self.dec_blocks = nn.ModuleList(
            [Block(cfg.dec_dim, cfg.dec_heads, cfg.mlp_ratio) for _ in range(cfg.dec_depth)]
        )
        self.dec_norm = nn.LayerNorm(cfg.dec_dim)
        self.head = nn.Linear(cfg.dec_dim, p)

        # Initialize the learned tokens and variable embeddings.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.var_embed_enc, std=0.02)
        nn.init.normal_(self.var_embed_dec, std=0.02)

    # ----- patch helpers -----

    def patchify(self, x):
        """(B, V, H, W) -> (B, V, L, p)"""
        # L: num of patches; p: number of points in a patch
        B, V, H, W = x.shape
        c = self.cfg
        x = x.reshape(B, V, c.grid_y, c.patch_y, c.grid_t, c.patch_t)
        return x.permute(0, 1, 2, 4, 3, 5).reshape(B, V, self.L, self.p)

    def unpatchify(self, tokens):
        """(B, V, L, p) -> (B, V, H, W)"""
        B = tokens.shape[0]
        c = self.cfg
        x = tokens.reshape(B, self.V, c.grid_y, c.grid_t, c.patch_y, c.patch_t)
        return x.permute(0, 1, 2, 4, 3, 5).reshape(B, self.V, c.y_pad_to, c.nt)

    def _embed_tokens(self, x):
        """(B, V, H, W) -> (B, V*L, D) with pos + variable embeddings added."""
        # D: token embedding dim
        B, V, H, W = x.shape
        t = self.patch_embed(x.reshape(B * V, 1, H, W))  # (B*V, D, gy, gt)
        t = t.flatten(2).transpose(1, 2).reshape(B, V, self.L, -1)  # (B, V, L, D)
        t = t + self.pos_embed_enc.unsqueeze(0).unsqueeze(0)
        t = t + self.var_embed_enc.unsqueeze(0).unsqueeze(2)
        return t.reshape(B, V * self.L, -1)

    @staticmethod
    def _visible_token_ids(spatial_mask, V, L):
        """(B, L) bool -> (B, V*K) long ids of visible tokens (v-major order)."""
        keep_counts = (~spatial_mask).sum(dim=1) # visible token num
        K = int(keep_counts[0].item())
        assert bool((keep_counts == K).all()), "mask must keep same count per row"
        ids_keep = torch.argsort(spatial_mask.long(), dim=1, stable=True)[:, :K]  # visible ID first
        ids_keep, _ = torch.sort(ids_keep, dim=1) #sort the visible ID
        return torch.cat([ids_keep + v * L for v in range(V)], dim=1)  # (B, V*K)

    # ----- MAE forward -----

    def forward_encoder(self, x, spatial_mask):
        tokens = self._embed_tokens(x)  # (B, V*L, D)
        token_ids = self._visible_token_ids(spatial_mask, self.V, self.L).to(x.device) #(B, V*L*ratio)
        vis = torch.gather(tokens, 1, token_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])) # the corresponding ID
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, vis], dim=1)
        for blk in self.enc_blocks:
            h = blk(h)
        return self.enc_norm(h), token_ids

    def forward_decoder(self, latent, token_ids):
        B = latent.shape[0]
        h = self.dec_embed(latent)  # linear layer (B, 1+V*K, dec)
        cls, vis = h[:, :1], h[:, 1:]
        full = self.mask_token.to(vis.dtype).expand(B, self.V * self.L, -1).clone()
        # expand to the original input size
        full.scatter_(1, token_ids.unsqueeze(-1).expand(-1, -1, vis.shape[-1]), vis) # put visible tokens back
        pv = (self.pos_embed_dec.unsqueeze(0) + self.var_embed_dec.unsqueeze(1)).reshape(
            self.V * self.L, -1
        )  # (V*L, dec-dim)
        full = full + pv.unsqueeze(0).to(full.dtype)
        h = torch.cat([cls, full], dim=1)
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        return self.head(h[:, 1:])  # (B, V*L, p)

    def forward(self, x, spatial_mask=None, generator=None):
        """Returns (loss, per_var_loss dict, pred (B,V,L,p), spatial_mask (B,L))."""
        B = x.shape[0]
        if spatial_mask is None:
            spatial_mask = random_spatial_mask(B, self.L, self.mask_ratio, generator)
        spatial_mask = spatial_mask.to(x.device)
        if not spatial_mask.any():
            raise ValueError("Reconstruction loss needs masked patches; use encode_features for unmasked input.")

        latent, token_ids = self.forward_encoder(x, spatial_mask)
        pred = self.forward_decoder(latent, token_ids).reshape(B, self.V, self.L, self.p)

        target = self.patchify(x)

        per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, V, L)
        # the MSE for all the patches
        m = spatial_mask.unsqueeze(1).float()  # (B, 1, L)
        denom = m.sum() * self.V # num of total masked patches
        loss = (per_patch * m).sum() / denom
        # per_patch[:, v] : [B, L]
        per_var = {
            self.cfg.variables[v]: float(
                (per_patch[:, v] * spatial_mask.float()).sum() / spatial_mask.float().sum()
            )
            for v in range(self.V)
        }
        return loss, per_var, pred, spatial_mask

    # ----- feature extraction (no masking) -----

    @torch.no_grad()
    def encode_features(self, x):
        tokens = self._embed_tokens(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_norm(h)
        cls_f = h[:, 0]
        patches = h[:, 1:].reshape(x.shape[0], self.V, self.L, -1)
        return {
            "cls": cls_f,  # (B, D)
            "mean": patches.mean(dim=(1, 2)),  # (B, D)
            "var_mean": patches.mean(dim=2),  # (B, V, D)
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ConvNeXt regression head
def norm_case_id(cid):
    fam, name = cid.split(":")
    return f"{fam}:{re.sub(r'^Rib', 'Ri', name)}"


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):  # (B, C, H, W)
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):  # (B, C, H, W)
        r = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        return r + x.permute(0, 3, 1, 2)


class ConvNeXtHead(nn.Module):
    def __init__(self, enc_dim=384, n_vars=3, red=64, dim=128, drop=0.2, out_act=None):
        super().__init__()
        self.n_vars = n_vars
        self.out_act = out_act  
        self.reduce = nn.Conv2d(enc_dim, red, 1)  # shared across variables
        self.proj = nn.Conv2d(n_vars * red, dim, 1)
        self.block0 = ConvNeXtBlock(dim)
        self.block1 = ConvNeXtBlock(dim)
        self.down = nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dim, 2, stride=2))
        self.block2 = ConvNeXtBlock(dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, 1)

    def forward(self, tok):  # (B, 3, 620, 384)
        B, V, L, D = tok.shape
        x = tok.reshape(B, V, GRID_Y, GRID_T, D) 
        x = x.permute(0, 1, 4, 2, 3)  # (B, 3, 384, 31, 20)
        x = self.reduce(x.reshape(B * V, D, GRID_Y, GRID_T)) # (384,) -> (64)
        x = x.reshape(B, V * x.shape[1], GRID_Y, GRID_T)  # concat -> (B, 192, 31, 20)
        x = self.proj(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = x.mean(dim=(2, 3))  #(B, 128, 15, 10) -> (B, 128) global average pooling
        out = self.norm(x)
        out = self.drop(out)
        out = self.head(out)
        out = out.squeeze(-1)
        if self.out_act == "sigmoid":
            out = torch.sigmoid(out)
        return out


# Alternative JEPA encoder
class JEPALayerNorm2d(nn.Module):
    """LayerNorm over channels for (B, C, H, W) tensors."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ResidualBlock(nn.Module):
    """ConvNeXt block: depthwise 7x7 -> LN -> 1x1 (x4) -> GELU -> 1x1 -> layer scale."""

    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        r = x
        x = self.conv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return r + x


class ConvEncoder2D(nn.Module):
    def __init__(
        self,
        in_chans=3,
        dims=(32, 64, 128, 256, 384),
        num_res_blocks=(2, 2, 2, 6, 2),
        strides=DEFAULT_STRIDES,
    ):
        super().__init__()
        assert len(dims) == len(num_res_blocks) == len(strides) + 1
        self.dims = list(dims)
        self.strides = [tuple(s) for s in strides]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(  # stem: no downsampling
                nn.Conv2d(in_chans, dims[0], kernel_size=4, padding="same"),
                JEPALayerNorm2d(dims[0]),
            )
        )
        for i, s in enumerate(self.strides):
            self.downsample_layers.append(
                nn.Sequential(
                    JEPALayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=s, stride=s),
                )
            )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(*[ResidualBlock(dims[i]) for _ in range(num_res_blocks[i])])
            for i in range(len(dims))
        )

    @property
    def out_dim(self):
        return self.dims[-1]

    def forward(self, x):  # (B, C, Y, T)
        for ds, blocks in zip(self.downsample_layers, self.res_blocks):
            x = blocks(ds(x))
        return x  # (B, D, Y/16, T/10)


class ConvPredictor2D(nn.Module):
    """Context latent -> predicted target latent, same shape."""

    def __init__(self, dim, scale_factor=1):
        super().__init__()
        hid = dim * scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hid, kernel_size=2, padding=1),
            ResidualBlock(hid),
            nn.Conv2d(hid, dim, kernel_size=2),
        )

    def forward(self, x):
        return self.conv(x)


def _off_diagonal(m):
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg(x, y, sim_coeff, std_coeff, cov_coeff):
    """x, y: (N, C) float32."""
    N, C = x.shape
    sim = F.mse_loss(x, y)
    xc, yc = x - x.mean(0), y - y.mean(0)
    std_x = torch.sqrt(xc.var(dim=0, unbiased=False) + 1e-4)
    std_y = torch.sqrt(yc.var(dim=0, unbiased=False) + 1e-4)
    std = F.relu(1.0 - std_x).mean() / 2 + F.relu(1.0 - std_y).mean() / 2
    cov_x = (xc.T @ xc) / max(1, N - 1)
    cov_y = (yc.T @ yc) / max(1, N - 1)
    cov = _off_diagonal(cov_x).pow(2).sum() / C + _off_diagonal(cov_y).pow(2).sum() / C
    return sim_coeff * sim + std_coeff * std + cov_coeff * cov, sim, std, cov


def vicreg_loss_dense(
    pred, target, sim_coeff=2.0, std_coeff=40.0, cov_coeff=2.0, n_chunks=5, generator=None
):
    """VICReg over every latent position. pred/target: (B, C, H, W).

    Positions are shuffled and split into n_chunks so the variance/covariance
    statistics are not dominated by spatially adjacent (correlated) positions.
    Computed in float32 regardless of autocast.
    """
    C = pred.shape[1]
    x = pred.float().permute(0, 2, 3, 1).reshape(-1, C)
    y = target.float().permute(0, 2, 3, 1).reshape(-1, C)
    perm = torch.randperm(x.shape[0], device=x.device, generator=generator)
    x, y = x[perm], y[perm]
    parts = [
        _vicreg(xc, yc, sim_coeff, std_coeff, cov_coeff)
        for xc, yc in zip(x.chunk(n_chunks), y.chunk(n_chunks))
    ]
    loss, sim, std, cov = (torch.stack(v).mean() for v in zip(*parts))
    return {"loss": loss, "sim": sim, "std": std, "cov": cov}


def split_context_target(x, n_ctx=100):
    """(B, C, Y, T) -> first n_ctx frames, remaining frames."""
    return x[..., :n_ctx], x[..., n_ctx:]


# Alternative FiLM/axial head
class RelPosAttention(nn.Module):
    """Multi-head self-attention along one axis with a learned relative-position bias."""

    def __init__(self, dim, heads, length):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * length - 1))
        idx = torch.arange(length)[:, None] - torch.arange(length)[None, :] + length - 1
        self.register_buffer("rel_idx", idx, persistent=False)

    def forward(self, x):  # (N, L, D)
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        bias = self.rel_bias[:, self.rel_idx].to(q.dtype)  # (heads, L, L), added to logits
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(x.transpose(1, 2).reshape(N, L, D))


class AxialBlock(nn.Module):
    def __init__(self, dim, heads, grid_y=GRID_Y, grid_t=GRID_T, mlp_ratio=4):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = RelPosAttention(dim, heads, grid_t)
        self.norm_y = nn.LayerNorm(dim)
        self.attn_y = RelPosAttention(dim, heads, grid_y)
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):  # (B, Y, T, D)
        B, Y, T, D = x.shape
        h = self.attn_t(self.norm_t(x).reshape(B * Y, T, D)).reshape(B, Y, T, D)
        x = x + h  # fixed height, along time
        h = self.norm_y(x).permute(0, 2, 1, 3).reshape(B * T, Y, D)
        h = self.attn_y(h).reshape(B, T, Y, D).permute(0, 2, 1, 3)
        x = x + h  # fixed time, along height
        return x + self.ffn(self.norm_f(x))


class QueryReadout(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):  # (B, L, D)
        q = self.query.expand(tokens.shape[0], -1, -1)
        a, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.norm(q + a).squeeze(1)


class FiLMAxialHead(nn.Module):
    def __init__(self, enc_dim=384, n_vars=3, dim=128, heads=4, use_axial=True, drop=0.2):
        super().__init__()
        self.n_vars, self.dim, self.use_axial = n_vars, dim, use_axial
        self.in_norm = nn.ModuleList(nn.LayerNorm(enc_dim) for _ in range(n_vars))
        self.in_proj = nn.ModuleList(nn.Linear(enc_dim, dim) for _ in range(n_vars))

        def film_mlp():
            m = nn.Sequential(nn.Linear((n_vars - 1) * dim, dim), nn.GELU(), nn.Linear(dim, dim))
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)  # identity at init
            return m

        self.film_gamma = nn.ModuleList(film_mlp() for _ in range(n_vars))
        self.film_beta = nn.ModuleList(film_mlp() for _ in range(n_vars))

        self.fusion = nn.Sequential(nn.Linear(n_vars * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.axial = AxialBlock(dim, heads) if use_axial else None
        self.readout = QueryReadout(dim, heads)
        self.reg = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim, 1)
        )

    def forward(self, tok):  # (B, V, 620, enc_dim)
        B, V, L, _ = tok.shape
        h = [self.in_proj[v](self.in_norm[v](tok[:, v])) for v in range(V)]
        out = []
        for v in range(V):  # all from pre-update features
            cond = torch.cat([h[u] for u in range(V) if u != v], dim=-1)
            out.append((1 + self.film_gamma[v](cond)) * h[v] + self.film_beta[v](cond))
        z = self.fusion(torch.cat(out, dim=-1))  # (B, 620, dim)
        if self.axial is not None:
            z = self.axial(z.reshape(B, GRID_Y, GRID_T, self.dim)).reshape(B, L, self.dim)
        return self.reg(self.readout(z)).squeeze(-1)

    def param_groups(self):
        return {
            "readout": list(self.readout.parameters()) + list(self.reg.parameters()),
            "film": list(self.film_gamma.parameters()) + list(self.film_beta.parameters()),
        }


def set_trainable(model, mode):
    """mode: 'all' | 'readout' | 'readout_film'"""
    for p in model.parameters():
        p.requires_grad_(mode == "all")
    if mode != "all":
        g = model.param_groups()
        for p in g["readout"] + (g["film"] if mode == "readout_film" else []):
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]
