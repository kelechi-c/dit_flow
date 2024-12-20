import jax, optax
from jax import Array, numpy as jnp, random as jrand
from flax import nnx
from tqdm import tqdm

randkey = jrand.key(333)
optimizer = nnx.Optimizer(tx=optax.adamw(learning_rate=1e-4))

@nnx.jit
def flow_lossfn(model, batch):
    x_1, c = batch['vae_output'], batch['labels']
    x_0 = jrand.normal(randkey, x_1.shape)
    t = jrand.normal(randkey, len(x_1))
    
    x_t = (1-t) * x_0 + t * x_1
    dx_t = x_1 - x_0 # actual vector/velocity difference 
    
    vtheta = model(x_t, t, c) # model prediction
    
    mean_dim = list(range(1, len(x_1.shape)))  # across all dimensions except the batch dim
    mean_square = (dx_t - vtheta) ** 2  # squared difference/error
    batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss
    loss = jnp.mean(batchwise_mse_loss)

    return loss


def flow_step(model, x_t: Array, t_start: Array, t_end: Array) -> Array:
    t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

    return (
        x_t + (t_end - t_start)
        * model(
            t=t_start + (t_end - t_start) / 2,
            x_t=x_t + model(x_t=x_t, t=t_start) * (t_end - t_start) / 2,
        )
    )


def sample(model, z_noise, cond, num_steps=50):
    dt = 1.0 / num_steps
    bs = z_noise.shape[0]
    
    t = jnp.zeros(randkey, (bs,))
    time_steps = jnp.linspace(0, 1.0, num_steps + 1)

    for k in tqdm(range(num_steps)):
        pass
    # def step_fn(x, i):
    #     t = jnp.full((bs,), i * dt)
    #     x = x + dt * model(x, cond, t)
    #     return x, None

    # x, _ = jax.lax.scan(step_fn, x, jnp.arange(num_steps))
    return x