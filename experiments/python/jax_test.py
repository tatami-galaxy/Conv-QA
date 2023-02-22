from flax import linen as nn
import jax
import jax.numpy as jnp

class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.relu(nn.Dense(8)(x))
    x = self.perturb('hidden', x)
    x = nn.Dense(2)(x)
    x = self.perturb('logits', x)
    return x

x = jnp.empty((1, 4)) # random data
y = jnp.empty((1, 2)) # random data

print(x)
quit()

model = Model()
variables = model.init(jax.random.PRNGKey(1), x)
params, perturbations = variables['params'], variables['perturbations']

def loss_fn(params, perturbations, x, y):
  y_pred = model.apply({'params': params, 'perturbations': perturbations}, x)
  return jnp.mean((y_pred - y) ** 2)

intermediate_grads = jax.grad(loss_fn, argnums=[0,1])(params, perturbations, x, y)

print(intermediate_grads)
