import jax, optax, wandb, torch, os, click, math, gc
import numpy as np
from flax import nnx
from jax import Array, numpy as jnp, random as jrand
jax.config.update("jax_default_matmul_precision", "bfloat16")

from tqdm import tqdm
from time import time
from einops import rearrange
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset
from typing import Any
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dit import config, randkey, DiT

import warnings
warnings.filterwarnings("ignore")

JAX_TRACEBACK_FILTERING = "off"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
XLA_PYTHON_CLIENT_MEM_FRACTION = 0.20
JAX_DEFAULT_MATMUL_PRECISION = "bfloat16"

# mesh / sharding configs
num_devices = jax.device_count()
devices = jax.devices()

print(f"found {num_devices} JAX device(s)")
for device in devices:
    print(f"{device} / ")

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


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
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
    # labels = jnp.array(
    #     [76, 292, 293, 979, 968, 967, 33, 88, 404]
    # )  # 76, 292, 293, 979, 968 imagenet
    randnoise = jrand.uniform(randkey, (len(labels), 32, 32, 4))
    pred_model = device_get_model(model)
    pred_model.eval()
    image_batch = pred_model.sample(randnoise, labels)
    file = f"samples/{step}_dit_output.png"
    batch = [process_img(x) for x in image_batch]

    gridfile = image_grid(batch, file)
    print(f"sample saved @ {gridfile}")
    del pred_model

    return gridfile


def vae_decode(latent, vae=vae):
    # print(f'decoding... (latent shape = {latent.shape}) ')
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
    def flow_lossfn(model, batch):# loss function for flow matching
        img_latents, labels = batch["vae_output"], batch["label"]

        img_latents = img_latents.reshape(-1, 4, 32, 32) * config.vaescale_factor
        img_latents = rearrange(img_latents, "b c h w -> b h w c")
        print(f"latents => {img_latents.shape}")

        img_latents, labels = jax.device_put((img_latents, labels))

        x_1, c = img_latents, labels # reassign to more concise variables

        x_0 = jrand.normal(randkey, x_1.shape) # noise
        t = jrand.normal(randkey, len(x_1)) # t/time cond

        x_t = (1-t) * x_0 + t * x_1
        dx_t = x_1 - x_0 # actual vector/velocity difference

        vtheta = model(x_t, t, c) # model vector prediction

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

    wandb_logger(
        key="", project_name="dit_jax"
    )

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        wandb.log(
            {"loss": train_loss.item(), "log_loss": math.log10(train_loss.item())}
        )

        jax.clear_caches()
        jax.clear_backends()
        gc.collect()

    etime = time.time() - stime
    print(
        f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs"
    )

    epoch_file = sample_image_batch("overfit", model)
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"epoch_sample": epoch_image_log})

    return model, train_loss


@click.command()
@click.option("-r", "--run", default="overfit")
@click.option("-e", "--epochs", default=10)
@click.option("-bs", "--batch_size", default=32)
def main(run, epochs, batch_size):
    
    # DiT-B config
    dit_model = DiT(
        dim=1024, 
        depth=16,
        attn_heads=16
    )
    
    n_params = sum([p.size for p in jax.tree.leaves(nnx.state(dit_model))])
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    optimizer = nnx.Optimizer(dit_model, optax.adamw(learning_rate=config.lr))

    
    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        batch_size=config.batch_size,
    )

    train_loader = DataLoader(
        dataset[: config.data_split],
        batch_size=batch_size,  # config.batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=jax_collate,
        # sampler=dataset_sampler, # thi sis for multiprocess/v4-32
    )

    if run == "overfit":
        model, loss = batch_trainer(epochs, model=dit_model, optimizer=optimizer, train_loader=train_loader)
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f'you missed your train looop impl boy')
        # trainer(epochs)
        # wandb.finish()
        # print("microdit (test) training (on imagenet-1k) in JAX..done")


main()
