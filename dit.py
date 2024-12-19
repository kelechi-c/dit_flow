import jax, math
from jax import Array, numpy as jnp
from flax import nnx
from einops import rearrange

rngs = nnx.Rngs(333)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0,
    )


def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)  # (1, H*W, D)


class PatchEmbed():
    def __init__(self, in_channels=4, img_size: int=32, dim=1024, patch_size: int = 2):
        self.dim = dim
        self.patch_size = patch_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.conv_project = nnx.Conv(
            in_channels,
            dim,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
            padding="VALID",
            kernel_init=xavier_init,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches = (H // self.patch_size[0])
        x = self.conv_project(x) # (B, P, P, hidden_size)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x


class FeedForward(nnx.Module):
    def __init__(self, dim: int, rngs=rngs):
        super().__init__()
        self.dim = dim
        self.mlp_dim = 2 * dim
        self.linear_1 = nnx.Linear(dim, self.mlp_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(self.mlp_dim, dim, rngs=rngs)
        
    def __call__(self, x: Array):
        x = self.linear_1(x)
        x = self.linear_2(nnx.gelu(x))
        
        return x


class DiTBlock(nnx.Module):
    def __init__(self, dim: int, attn_heads: int, drop: float = 0.0, rngs=rngs):
        super().__init__()
        self.dim = dim
        self.norm_1 = nnx.LayerNorm(
            dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init
        )
        self.attention = nnx.MultiHeadAttention(num_heads=attn_heads, in_features=dim, decode=False, dropout_rate=drop, rngs=rngs)
        self.adaln = nnx.Linear(dim, 6*dim, kernel_init=zero_init, bias_init=zero_init, rngs=rngs)

    def __call__(self, x_img: Array, cond: Array):
        cond = self.adaln(nnx.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.array_split(cond, 6, axis=-1)

        attn_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))
        x = x_img + (jnp.expand_dims(gate_msa, 1) * attn_x)

        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_x = self.mlp_block(x)
        x = x + (jnp.expand_dims(gate_mlp, 1) * mlp_x)

        return x


class FinalMLP(nnx.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()

        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * out_channels,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=zero_init,
        )

        self.adaln_linear = nnx.Linear(
            hidden_size,
            2 * hidden_size,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=zero_init
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=-1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


class DiT(nnx.Module):
    def __init__(self, dim: int = 1024, patch_size: tuple = (2, 2), depth: int = 12, attn_heads=16):
        super().__init__()
        self.dim = dim
        self.patch_embed = PatchEmbed(dim=dim, patch_size=patch_size[0])
        self.label_embedder = None
        
    
    def __call__(self, x: Array):
        
        return x
