"""GPU env: FreqFlow forward, per-row FM loss and selected gradients in float32, for check_equivalence.py."""
import importlib.util
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from libs.ours.blocks_RoPERMSNorm import FreqFlow  # noqa: E402
import onpolicy  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg_file, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location('c', cfg_file)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
torch.manual_seed(0)
model = FreqFlow(c.get_config().nnet.model_args).cuda()
with torch.no_grad():  # move the time-dependent and zero-initialized parameters off their init, to exercise them
    for name, p in model.named_parameters():
        if name.endswith(('param0', 'param1', 'nodecay_weight', 'nodecay_bias')) or name.endswith('.bias'):
            p.add_(0.05 * torch.randn_like(p))
B = 4
g = torch.Generator(device='cuda').manual_seed(1)
x = torch.randn(B, 4, 32, 32, device='cuda', generator=g)
t = torch.tensor([0.05, 0.4, 0.8, 0.97], device='cuda')
v = torch.randn(B, 4, 32, 32, device='cuda', generator=g)
y = torch.tensor([3, 999, 1000, 17], device='cuda')
out0, out1 = model(x, log_snr=t * 8 - 4, y=y)
loss = onpolicy.fm_loss(model, x, t, v, y)
loss.mean().backward()
grads = ['stages.0.blocks.3.block2.norm.scale.param0', 'stages.0.blocks.3.block2.norm.scale.nodecay_weight',
         'stages.0.blocks.20.block1.query_key_value.weight', 'stages.1.upsample.1.kernel.weight',
         'stages.1.upsample.0.weight.nodecay_bias', 'stages.1.blocks.5.block1.1.weight', 'stages.0.class_embeds.weight',
         'stages.0.skip_denses.4.weight', 'stages.1.output_layer.2.weight', 'stages.0.input_layer.1.weight']
named = dict(model.named_parameters())
np.savez(out, **{f'sd/{k}': v.detach().cpu().numpy() for k, v in model.state_dict().items()},
         x=x.cpu().numpy(), t=t.cpu().numpy(), v=v.cpu().numpy(), y=y.cpu().numpy(),
         out0=out0.detach().cpu().numpy(), out1=out1.detach().cpu().numpy(), loss=loss.detach().cpu().numpy(),
         **{f'grad/{k}': named[k].grad.cpu().numpy() for k in grads})
print('wrote', out, 'params', sum(p.numel() for p in model.parameters()))
