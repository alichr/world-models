"""Hyper-parameters for every stage of the World Models pipeline.

Two presets are provided:

* ``paper``  - the values used by Ha & Schmidhuber (2018) and their reference
               implementation (github.com/hardmaru/WorldModelsExperiments).
* ``laptop`` - a scaled-down budget that trains on a single Apple-silicon laptop
               in a few hours. Only the *amount* of data / compute changes; the
               architectures and losses are identical to the paper.

Every value can also be overridden from the command line of each script.
"""

# Shared architecture constants (identical in both presets).
Z_SIZE = 32          # VAE latent size  (paper: 32)
ACTION_SIZE = 3      # steer, gas, brake
RNN_HIDDEN = 256     # LSTM hidden units (paper: 256)
N_MIXTURES = 5       # Gaussians per latent dim in the MDN (paper: 5)
FRAME_SIZE = 64      # observations are resized to 64x64x3

PRESETS = {
    "paper": {
        "collect": dict(num_rollouts=10_000, max_steps=1000),
        "vae": dict(epochs=10, batch_size=100, lr=1e-4, kl_tolerance=0.5),
        "mdrnn": dict(steps=4000, batch_size=100, seq_len=999, lr=1e-3, grad_clip=1.0),
        "controller": dict(
            popsize=64,             # CMA-ES population
            episodes=16,            # rollouts averaged per candidate
            sigma=0.1,              # initial CMA-ES step size
            weight_decay=0.01,      # L2 penalty on controller params (estool default)
            generations=2000,       # no fixed budget in the paper; run until it plateaus
            eval_every=25,          # paper: evaluate the mean every 25 generations ...
            eval_episodes=1024,     # ... on 1024 random tracks
            early_stop=0,           # 0 = always play the full 1000-step episode
        ),
    },
    "laptop": {
        "collect": dict(num_rollouts=1000, max_steps=1000),
        "vae": dict(epochs=10, batch_size=100, lr=1e-4, kl_tolerance=0.5),
        "mdrnn": dict(steps=4000, batch_size=100, seq_len=999, lr=1e-3, grad_clip=1.0),
        "controller": dict(
            popsize=32,
            episodes=4,
            sigma=0.1,
            weight_decay=0.01,
            generations=200,
            eval_every=10,
            eval_episodes=32,
            early_stop=100,         # end an episode after 100 steps without a new tile
        ),
    },
}


def stage_defaults(preset: str, stage: str) -> dict:
    return dict(PRESETS[preset][stage])


def resolve(args, stage: str) -> dict:
    """Preset values for ``stage``, overridden by any CLI flag the user passed."""
    cfg = stage_defaults(args.preset, stage)
    for key, value in vars(args).items():
        if key in cfg and value is not None:
            cfg[key] = value
    return cfg
