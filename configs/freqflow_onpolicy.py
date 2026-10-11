"""On-policy finetuning of FreqFlow (train_onpolicy.py): the FM loss plus a
CFG-direction term and an MMD term on the model's own sampler rollouts.
Both terms default to weight 0, which is plain finetuning."""
import ml_collections

from configs.freqflow import get_config as get_base_config


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = get_base_config()

    config.init_path = ''  # pretrained weights (e.g. nnet_ema.pth), loaded into nnet and nnet_ema
    config.ema_rate = 0.9999
    config.train.n_steps = 20000
    config.train.grad_accum = 1      # micro-batches per optimizer step; batch_size is the global batch
    # recompute the blocks and the skip concat in backward (configs/freqflow_l.py): FreqFlow-L's eager activations
    # do not fit a 128-row micro-batch in 80 GB otherwise
    config.train.checkpoint_blocks = True
    config.train.checkpoint_lowres = True
    config.train.save_interval = 2000
    config.train.eval_interval = 1000  # sample grid every N steps (needs the autoencoder); 0 = off
    config.optimizer.lr = 1e-5
    config.lr_scheduler.warmup_steps = 500

    config.onpolicy = d(
        pool_path='',            # folder written by dump_latent_pool.py
        start_step=500,          # before this, plain FM (no rollouts): the LR warmup

        # rollout = eval's euler_sampler, cut at the band
        num_steps=50,
        shift=0.3,
        cfg_scale=1.0,           # sampler convention: v_u + s (v_c - v_u), applied only for
        guidance_low=0.0,        # guidance_low <= t <= guidance_high; the default band sits
        guidance_high=0.7,       # above 0.7, so it is unguided whatever cfg_scale is
        band_t_min=0.8245,       # band = sampler grid states with t > band_t_min (t=1 is noise): 1, 0.9363, 0.878,
                                 # 0.82456 (just above the split). At 0.82456 the exact empirical posterior is a
                                 # nearest-neighbour lookup (ESS ~1-2)
        n_roll=32,               # trajectories per GPU per micro-batch; CFG needs n_roll >= n_band (22 band rows of
                                 # a 128-row micro-batch) and MMD needs mmd_n_target % (n_gpus * n_roll) == 0

        # expert posterior for the band's FM labels: exact (full) or a K-latent bank, no duplicates
        bank_full=True,          # conditional rows: whole class pool
        bank_full_uncond=True,   # dropped rows: whole pool
        bank_k=1024,
        bank_k_uncond=1024,
        expert_chunk=16384,      # shard rows scored per GEMM

        # CFG-direction term: mean || (v_c - v_u) - (v*_c - v*_u) ||^2 on the band rows.
        # When on, the band's FM rows (batch share 1 - band_t_min, one per distinct
        # trajectory) are expert-labeled rollout states.
        cfg_gamma=0.0,
        cfg_target_full=True,    # exact v*_c and v*_u whatever bank_full / bank_full_uncond say

        # MMD term: mean over band states (t < 1) of one MMD^2 over all GPUs' rollout states
        # against all targets, the targets class-matched to the batch's rollouts
        mmd_gamma=0.0,           # MMD^2 at these bandwidths is ~1e-8 near noise: read gamma off the logged noise floor
        mmd_n_target=2048,       # global target rows per band state, distinct; each rollout of class C brings
                                 # mmd_n_target / (n_gpus * n_roll) class-C latents
        mmd_sigma_scales=(0.1, 0.25),   # x d2_base; distances concentrate within a few % in 4096-D, so
                                        # scales >= 1 only see the mean and total variance
        mmd_grad_ckpt=True,
    )
    return config
