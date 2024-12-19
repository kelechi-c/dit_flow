import jax, math
from jax import Array, numpy as jnp
from flax import nnx

rngs = nnx.Rngs(333)
xavier_init = nnx.initializers.xavier_uniform()
zero_init = nnx.initializers.constant(0.0)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.array_split(cond, 6, axis=1)

        attn_mod_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))

        x = x_img + jnp.expand_dims(gate_msa, 1) * attn_mod_x
        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_mod_x = self.mlp_block(x)
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp_mod_x

        return x
