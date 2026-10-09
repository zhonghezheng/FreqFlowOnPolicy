"""TPU env: tpu/model.py with weights converted from dump_torch_reference.py must reproduce PyTorch's float32
outputs, per-row FM loss and gradients."""
import importlib.util
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, '.')
from tpu.convert import flax_to_torch, torch_to_flax  # noqa: E402
from tpu.losses import fm_loss_rows  # noqa: E402
from tpu.model import FreqFlow  # noqa: E402

jax.config.update('jax_default_matmul_precision', 'highest')
cfg_file, ref_file = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location('c', cfg_file)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
margs = c.get_config().nnet.model_args
ref = np.load(ref_file)
sd = {k[3:]: ref[k] for k in ref.files if k.startswith('sd/')}
params = jax.tree.map(jnp.asarray, torch_to_flax(sd, margs))
x, t, v, y = (jnp.asarray(ref[k]) for k in ('x', 't', 'v', 'y'))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


for remat in (False, True):
    model = FreqFlow(margs, dtype=jnp.float32, remat=remat)
    if not remat:
        shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), x, t * 8 - 4, y))['params']
        ok = jax.tree.structure(shapes) == jax.tree.structure(params) and all(
            a.shape == b.shape for a, b in zip(jax.tree.leaves(shapes), jax.tree.leaves(params)))
        print('converted tree matches model.init:', ok)
    apply = lambda p, xx, ls, yy, g=1: model.apply({'params': p}, xx, ls, yy, g)
    out0, out1 = jax.jit(apply)(params, x, t * 8 - 4, y)
    loss_fn = lambda p: fm_loss_rows(apply, p, x, t, v, y).mean()
    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(params)
    rows = jax.jit(lambda p: fm_loss_rows(apply, p, x, t, v, y))(params)
    g_sd = flax_to_torch(jax.device_get(grads), margs)
    print(f'remat={remat}: out0 rel {rel(out0, ref["out0"]):.2e}, out1 rel {rel(out1, ref["out1"]):.2e}, '
          f'loss rows rel {rel(rows, ref["loss"]):.2e} ({np.asarray(rows).round(5)} vs {ref["loss"].round(5)})')
    for k in ref.files:
        if k.startswith('grad/'):
            print(f'    grad {k[5:]:55s} rel {rel(g_sd[k[5:]], ref[k]):.2e}')
