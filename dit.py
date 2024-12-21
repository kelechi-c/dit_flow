import jax, math
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from einops import rearrange
from tqdm import tqdm

class config:
    seed = 333
    batch_size = 128
    img_size = 32
    patch_size = (2, 2)
    lr = 2e-4
    cfg_scale = 2.0
    vaescale_factor = 0.13025
    data_id = "cloneofsimo/imagenet.int8"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    imagenet_id = "ILSVRC/imagenet-1k"


rngs = nnx.Rngs(config.seed)
randkey = jrand.key(config.seed)

xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)
normal_init = nnx.initializers.normal(0.02)

## helpers/util functions
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


def get_2d_sincos_pos_embed(embed_dim, length):
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
        patch_tuple = (patch_size, patch_size)
        self.num_patches = (img_size // self.patch_size) ** 2
        self.conv_project = nnx.Conv(
            in_channels,
            dim,
            kernel_size=patch_tuple,
            strides=patch_tuple,
            use_bias=False,
            padding="VALID",
            kernel_init=xavier_init
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side = (H // self.patch_size)
        x = self.conv_project(x) # (B, P, P, hidden_size)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches_side, w=num_patches_side)
        return x


class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size, freq_embed_size=256):
        super().__init__()
        self.lin_1 = nnx.Linear(freq_embed_size, hidden_size, rngs=rngs)
        self.lin_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def timestep_embedding(time_array: Array, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half) / half)

        args = jnp.float_(time_array[:, None]) * freqs[None]

        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concat(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )

        return embedding

    def __call__(self, x: Array) -> Array:
        t_freq = self.timestep_embedding(x, self.freq_embed_size)
        t_embed = nnx.silu(self.lin_1(t_freq))

        return self.lin_2(t_embed)


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes, hidden_size, drop):
        super().__init__()
        use_cfg_embeddings = drop > 0
        self.embedding_table = nnx.Embed(
            num_classes + int(use_cfg_embeddings),
            hidden_size,
            rngs=rngs,
            embedding_init=nnx.initializers.normal(0.02),
        )
        self.num_classes = num_classes
        self.dropout = drop

    def token_drop(self, labels, force_drop_ids=None) -> Array:
        if force_drop_ids is None:
            drop_ids = jrand.normal(key=randkey, shape=labels.shape[0])
        else:
            drop_ids = force_drop_ids == 1

        labels = jnp.where(drop_ids, self.num_classes, labels)

        return labels

    def __call__(self, labels, train: bool, force_drop_ids=None) -> Array:
        use_drop = self.dropout > 0
        if (train and use_drop) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        label_embeds = self.embedding_table(labels)

        return label_embeds


class CaptionEmbedder(nnx.Module):
    def __init__(self, cap_embed_dim, embed_dim):
        super().__init__()
        self.linear_1 = nnx.Linear(cap_embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

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
    def __init__(self, in_channels: int=4, dim: int = 1024, patch_size: tuple = (2, 2), depth: int = 12, attn_heads=16, drop=0.0):
        super().__init__()
        self.dim = dim
        self.out_channels = in_channels
        self.patch_embed = PatchEmbed(dim=dim, patch_size=patch_size[0])
        self.num_patches = self.patch_embed.num_patches 
        self.label_embedder = LabelEmbedder(1000, dim, drop=drop)
        self.time_embed = TimestepEmbedder(dim)
        self.dit_layers = [DiTBlock(dim, attn_heads, drop=drop) for _ in range(depth)]
        self.final_layer = FinalMLP(dim, patch_size, self.out_channels)

    def _unpatchify(self, x, patch_size=(2, 2), height=32, width=32):

        bs, num_patches, patch_dim = x.shape
        H, W = patch_size
        in_channels = patch_dim // (H * W)
        # Calculate the number of patches along each dimension
        num_patches_h, num_patches_w = height // H, width // W

        # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
        x = x.view(bs, num_patches_h, num_patches_w, H, W, in_channels)

        # Permute x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Reshape x to (bs, height, width, in_channels)
        reconstructed = x.view(bs, height, width, in_channels)

        return reconstructed

    def __call__(self, x_t: Array, label: Array, t: Array):

        pos_embed = get_2d_sincos_pos_embed(rng=rngs, embed_dim=self.dim, length=self.num_patches**2)

        x = self.patch_embed(x_t)
        x = x + pos_embed
        t = self.time_embed(t)
        y = self.label_embedder(label)

        cond = t + y
        for layer in self.dit_layers:
            x = layer(x, cond)

        x = self.final_layer(x)
        x = self._unpatchify(x)

        return x

    def flow_step(self, x_t: Array, cond: Array, t_start: Array, t_end: Array) -> Array:
        t_start = jnp.broadcast_to(t_start.reshape(1, 1), (x_t.shape[0], 1))

        return x_t + (t_end - t_start) * self(
            t=t_start + (t_end - t_start) / 2,
            x_t=x_t + self(x_t=x_t, cond=cond, t=t_start) * (t_end - t_start) / 2,
            cond=cond,
        )

    def sample(self, x_t: Array, label: Array, num_steps: int = 50):
        # t = jnp.zeros(randkey, (bs,))
        time_steps = jnp.linspace(0, 1.0, num_steps + 1)

        for k in tqdm(range(num_steps), desc='smapling images(hopefully...)'):
            x_t = self.flow_step(
                x_t=x_t, cond=label, t_start=time_steps[k], t_end=time_steps[k + 1]
            )

        return x_t
