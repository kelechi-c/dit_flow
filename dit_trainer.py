import jax, optax, wandb, torch, os, click, math, gc, time
import numpy as np
from flax import nnx
from jax import Array, numpy as jnp, random as jrand
from jax.sharding import NamedSharding, Mesh, PartitionSpec as PS
from jax.experimental import mesh_utils
from jax import random as jrand
jax.config.update("jax_default_matmul_precision", "bfloat16")

import streaming
from tqdm import tqdm
from einops import rearrange
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset
from typing import Any
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"


class config:
    seed = 333
    batch_size = 128
    data_split = 10000
    img_size = 32
    patch_size = (2, 2)
    lr = 2e-4
    cfg_scale = 2.0
    vaescale_factor = 0.13025
    data_id = "cloneofsimo/imagenet.int8"
    vae_id = "madebyollin/sdxl-vae-fp16-fix"
    imagenet_id = "ILSVRC/imagenet-1k"


num_devices = jax.device_count()
devices = jax.devices()

print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")

mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, axis_names="axis")
sharding = NamedSharding(mesh, PS("axis"))

# sd VAE for decoding latents
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
print("loaded vae")


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


def jax_collate(batch):
    latents = jnp.stack([jnp.array(item["vae_output"]) for item in batch], axis=0)
    labels = jnp.stack([int(item["label"]) for item in batch], axis=0)

    return {
        "vae_output": latents,
        "label": labels,
    }


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet"  # just a local mirror path


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
        ), axis=0
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


class PatchEmbed:
    def __init__(
        self, in_channels=4, img_size: int = 32, dim=1024, patch_size: int = 2
    ):
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
            kernel_init=xavier_init,
            rngs=rngs,
        )

    def __call__(self, x):
        B, H, W, C = x.shape
        num_patches_side = H // self.patch_size
        x = self.conv_project(x)  # (B, P, P, hidden_size)
        x = rearrange(x, "b h w c -> b (h w) c", h=num_patches_side, w=num_patches_side)
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

    def __call__(self, labels, train: bool = True, force_drop_ids=None) -> Array:
        use_drop = self.dropout > 0
        if (train and use_drop) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        label_embeds = self.embedding_table(labels)

        return label_embeds


class CaptionEmbedder(nnx.Module):
    def __init__(self, cap_embed_dim, embed_dim, rngs=nnx.Rngs(config.seed)):
        super().__init__()
        self.linear_1 = nnx.Linear(cap_embed_dim, embed_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = nnx.silu(self.linear_1(x))
        x = self.linear_2(x)

        return x


class FeedForward(nnx.Module):
    def __init__(self, dim: int, rngs=nnx.Rngs(config.seed)):
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
    def __init__(
        self, dim: int, attn_heads: int, drop: float = 0.0, rngs=nnx.Rngs(config.seed)
    ):
        super().__init__()
        self.dim = dim
        self.norm_1 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init)
        self.norm_2 = nnx.LayerNorm(dim, epsilon=1e-6, rngs=rngs, bias_init=zero_init)
        self.attention = nnx.MultiHeadAttention(
            num_heads=attn_heads,
            in_features=dim,
            decode=False,
            dropout_rate=drop,
            rngs=rngs,
        )
        self.adaln = nnx.Linear(
            dim, 6 * dim, kernel_init=zero_init, bias_init=zero_init, rngs=rngs
        )
        self.mlp_block = FeedForward(dim)

    def __call__(self, x_img: Array, cond: Array):
        cond = self.adaln(nnx.silu(cond))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.array_split(cond, 6, axis=-1)
        )

        attn_x = self.attention(modulate(self.norm_1(x_img), shift_msa, scale_msa))
        x = x_img + (jnp.expand_dims(gate_msa, 1) * attn_x)

        x = modulate(self.norm_2(x), shift_mlp, scale_mlp)
        mlp_x = self.mlp_block(x)
        x = x + (jnp.expand_dims(gate_mlp, 1) * mlp_x)

        return x


class FinalMLP(nnx.Module):
    def __init__(
        self, hidden_size, patch_size, out_channels, rngs=nnx.Rngs(config.seed)
    ):
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
            bias_init=zero_init,
        )

    def __call__(self, x_input: Array, cond: Array):
        linear_cond = nnx.silu(self.adaln_linear(cond))
        shift, scale = jnp.array_split(linear_cond, 2, axis=-1)

        x = modulate(self.norm_final(x_input), shift, scale)
        x = self.linear(x)

        return x


class DiT(nnx.Module):
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 1024,
        patch_size: tuple = (2, 2),
        depth: int = 12,
        attn_heads=16,
        drop=0.0,
        rngs=nnx.Rngs(config.seed),
    ):
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
        x = x.reshape((bs, num_patches_h, num_patches_w, H, W, in_channels))

        # transpose x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
        x = x.transpose(0, 1, 3, 2, 4, 5)

        # Reshape x to (bs, height, width, in_channels)
        reconstructed = x.reshape((bs, height, width, in_channels))

        return reconstructed

    def __call__(self, x_t: Array, cond: Array, t: Array):
        pos_embed = get_2d_sincos_pos_embed(embed_dim=self.dim, length=self.num_patches)

        x = self.patch_embed(x_t)
        x = x + pos_embed
        t = self.time_embed(t)
        y = self.label_embedder(cond)

        cond = t + y

        for layer in self.dit_layers:
            x = layer(x, cond)

        x = self.final_layer(x, cond)
        x = self._unpatchify(x)
        return x

    def flow_step(self, x_t: Array, cond: Array, t_start: float, t_end: float) -> Array:
        """Performs a single flow step using Euler's method."""
        t_mid = (t_start + t_end) / 2.0
        # Broadcast t_mid to match x_t's batch dimension
        t_mid = jnp.full((x_t.shape[0],), t_mid)
        # Evaluate the vector field at the midpoint
        v_mid = self(x_t=x_t, cond=cond, t=t_mid)
        # Update x_t using Euler's method
        x_t_next = x_t + (t_end - t_start) * v_mid
        return x_t_next

    def sample(self, label: Array, num_steps: int = 50):
        """Generates samples using flow matching."""
        time_steps = jnp.linspace(0.0, 1.0, num_steps + 1)
        x_t = jax.random.normal(randkey, (len(label), 32, 32, 4))  # important change

        for k in tqdm(range(num_steps), desc="Sampling images"):
            x_t = self.flow_step(
                x_t=x_t, cond=label, t_start=time_steps[k], t_end=time_steps[k + 1]
            )

        return x_t / config.vaescale_factor


def wandb_logger(key: str, project_name, run_name=None):
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def device_get_model(model):
    state = nnx.state(model)
    state = jax.device_get(state)
    nnx.update(model, state)

    return model


def sample_image_batch(step, model, labels):
    pred_model = device_get_model(model)
    pred_model.eval()
    image_batch = pred_model.sample(labels)
    file = f"fmsamples/{step}_flowdit.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")
    del pred_model

    return gridfile


def vae_decode(latent, vae=vae):
    tensor_img = rearrange(latent, "b h w c -> b c h w")
    tensor_img = torch.from_numpy(np.array(tensor_img))
    x = vae.decode(tensor_img).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(10, 10)):
    rows, cols = grid_size
    assert len(pil_images) <= rows * cols, "Grid size must accommodate all images."

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            # Convert PIL image to NumPy array and plot
            ax.imshow(np.array(pil_images[i]))
            ax.axis("off")  # Turn off axis labels
        else:
            ax.axis("off")  # Hide empty subplots for unused grid spaces

    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")
    plt.show()

    return file


@nnx.jit
def train_step(model, optimizer, batch):
    def flow_lossfn(model, batch):  # loss function for flow matching
        img_latents, labels = batch["vae_output"], batch["label"]

        img_latents = img_latents.reshape(-1, 4, 32, 32) * config.vaescale_factor
        img_latents = rearrange(img_latents, "b c h w -> b h w c") # jax uses channels-last format

        img_latents, labels = jax.device_put((img_latents, labels), jax.devices()[0])

        x_1, c = img_latents, labels  # reassign to more concise variables
        bs = x_1.shape[0]

        x_0 = jrand.normal(randkey, x_1.shape)  # noise
        t = jrand.uniform(randkey, (bs,))
        t = nnx.sigmoid(t)

        inshape = [1] * len(x_1.shape[1:])
        t_exp = t.reshape([bs, *(inshape)])
        # print(f'{x_1.shape = } / {c.shape = } / {t.shape = }')

        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        dx_t = x_1 - x_0  # actual vector/velocity difference

        vtheta = model(x_t, c, t)  # model vector prediction

        mean_dim = list(range(1, len(x_1.shape)))  # across all dimensions except the batch dim
        mean_square = (dx_t - vtheta) ** 2  # squared difference/error
        batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss
        loss = jnp.mean(batchwise_mse_loss)

        return loss

    loss, grads = nnx.value_and_grad(flow_lossfn)(model, batch)
    optimizer.update(grads)

    return loss


def batch_trainer(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(key="", project_name="dit_jax")

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        wandb.log({"loss": train_loss, "log_loss": math.log10(train_loss)})

        if epoch % 50 == 0:
            gridfile = sample_image_batch(epoch, model, batch["label"])
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log})

        jax.clear_caches()
        jax.clear_backends()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    epoch_file = sample_image_batch("overfit", model, batch["label"])
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"overfit_sample": epoch_image_log})

    return model, train_loss


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=config.epochs)
@click.option("-bs", "--batch_size", default=config.batch_size)
def main(run, epochs, batch_size):
    # DiT-B config
    dit_model = DiT(dim=1024, depth=24, attn_heads=16)

    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(dit_model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    optimizer = nnx.Optimizer(dit_model, optax.adamw(learning_rate=config.lr))

    state = nnx.state((dit_model, optimizer))
    state = jax.device_put(state, jax.devices()[0])
    nnx.update((dit_model, optimizer), state)

    streaming.base.util.clean_stale_shared_memory()
    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        batch_size=config.batch_size,
    )

    train_loader = DataLoader(
        dataset[: config.data_split],
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=jax_collate,
    )

    sp = next(iter(train_loader))
    print(f"loaded data \n data sample: {sp['vae_output'].shape}")

    if run == "single_batch":
        model, loss = batch_trainer(epochs, model=dit_model, optimizer=optimizer, train_loader=train_loader)
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")
        
    elif run == "train":
        print(f"you missed your train looop impl boy")

main()
