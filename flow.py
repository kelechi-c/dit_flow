import optax
from jax import Array, numpy as jnp, random as jrand
from flax import nnx

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
