"""
The following implementations are for image/2D data, ported to JAX (and for diffusion purposes)
from Meta's 1D-data pytorch implementation @ https://github.com/facebookresearch/flow_matching/blob/main/examples/standalone_flow_matching.ipynb 
"""

import jax
from jax import Array, numpy as jnp, random as jrand
from tqdm import tqdm
from flax import nnx

randkey = jrand.key(333) # set seed/key for array creation

@jax.jit
def flow_lossfn(model, batch): # flow matching loss for diffusion/images(DiTs specifically)
    x_1, c = batch # data, image latent/label pairs
    bs = x_1.shape[0]

    x_0 = jrand.normal(randkey, x_1.shape)  # noise
    t = jrand.uniform(randkey, (bs,))
    t = nnx.sigmoid(t)

    inshape = [1] * len(x_1.shape[1:]) # shape of array excluding batch axis
    t_exp = t.reshape([bs, *(inshape)]) 

    x_t = (1 - t_exp) * x_0 + t_exp * x_1 # add noise
    dx_t = x_1 - x_0  # actual vector/velocity difference

    vtheta = model(x_t, c, t)  # model/vector field prediction

    mean_dim = list(range(1, len(x_1.shape)))  # across all dimensions except the batch dim
    mean_square = (dx_t - vtheta) ** 2  # squared difference/error
    batchwise_mse_loss = jnp.mean(mean_square, axis=mean_dim)  # mean loss
    loss = jnp.mean(batchwise_mse_loss) # average again
    
    return loss


# Performs a single flow step using Euler's method, with a midpoint step.
def flow_step(self, x_t: Array, cond: Array, t_start: float, t_end: float) -> Array:
    t_mid = (t_start + t_end) / 2.0
    # Broadcast t_mid to match x_t's batch dimension
    t_mid = jnp.full((x_t.shape[0],), t_mid)
    # Evaluate the vector field at the midpoint
    v_mid = self(x_t=x_t, cond=cond, t=t_mid)
    # Update x_t using Euler's method
    x_t_next = x_t + (t_end - t_start) * v_mid
    return x_t_next

# Generates samples using flow matching
def sample(self, label: Array, num_steps: int = 50):
    time_steps = jnp.linspace(0.0, 1.0, num_steps + 1)
    x_t = jax.random.normal(randkey, (len(label), 32, 32, 4))  # important change

    for k in tqdm(range(num_steps), desc="Sampling images(hopefully...)"):
        x_t = self.flow_step(
            x_t=x_t, cond=label, t_start=time_steps[k], t_end=time_steps[k + 1]
        )

    return x_t / 0.13025
