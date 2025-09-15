# Learning Jax

[Tutorial link](https://flax.readthedocs.io/en/latest/mnist_tutorial.html)

Aim of this task: train Flax model on MNIST to understand basic Flax structure.

## Task 1: MNIST
1. flax.nnx: similar to torch.nn
2. nnx.Module acts in same way as torch.nn.Module
3. Utilise __call__ for forward pass rather than forward
4. In the tutorial, they use non-flax functions for functools. These have to be jax compatible and not contain parameters which require gradients.
There are however, jax versions provided, which I will likely use in future to avoid confusion.
5. Define loss function, training etc. in function and then **jax compile them**.
6.  Very fucntional...  example:

```
def train_step(model: CNN, optimiser: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimiser.update(grads)
    ```