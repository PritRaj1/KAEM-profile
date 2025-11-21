# KAEM 

KAEM is a generative model presented [here](https://www.arxiv.org/abs/2506.14167).

---

## Brief

KAEM is an *extremely* fast and interpretable generative model for any data modality/top-down generator network.

---

### ⚡ How is it so fast?

#### Inference
- KAEM uses inverse transform sampling from its latent prior. This produces exact samples within a single forward pass, i.e., almost instantaneously.

#### Training
- 3 training strategies are provided depending on statistical requirements. All of them avoid the use of encoders, and learning is solely conducted in a low-dimensional latent space.
- Annealing/population-based MCMC is also presented as an embarassingly parallel and scalable alternative to diffusion/score-matching.

#### Compilation/runtime
- KAEM is compiled using Reactant.jl and trained with EnzymeMLIR for autodifferentiation.
- These are bleeding-edge, experimental tools that offer first-in-class speed and allocations, faster than any other machine learning framework.

---

### 🔍 How is it interpretable?

#### Deterministic representation
- KAEM has completely redefined generative modeling using the Kolmogorov-Arnold Representation theorem as its basis.
- While generative modeling is a probabilistic task, KAEM uses the inverse transform method to instead reframe the theorem without introducing stochasticity.

#### Latent distributions
- KAEM uses a Kolmogorov-Arnold Network prior.
- One can simply plot each latent feature's distribution and look at it.
- Complex covariance relationships are deferred to the generator.

---

## Why should you care?

Typically, generative models succumb to trade-offs amongst the following:

- Fast inference
- High quality
- Stable training
- Interpretability

KAEM does not, especially when platformed om our XPUs.

## Setup:

Need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Julia](https://github.com/JuliaLang/juliaup). Choose your favourite installer and run: 

```bash
bash <conda-installer-name>-latest-Linux-x86_64.sh
curl -fsSL https://install.julialang.org | sh
```

Then install

```bash
make install
```

[Optional;] Test all Julia scripts:

```bash
make test
```

### Note for windows users:

This repo uses shell scripts solely for convenience, you can run everything without them too. If you want to use the shell scripts, [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) is recommended.

## Quick start:

List commands:
```
make help
```

Edit the config files:

```bash
nvim config/nist_config.ini
```

For individual experiments run:

```bash
make train-vanilla DATASET=MNIST
make train-thermo DATASET=SVHN
```

To automatically run experiments one after the other:
```bash
nvim jobs.txt # Schedule jobs
make train-sequential CONFIG=jobs.txt
```

For benchmarking run:

```bash
make bench
```

## Julia flow:

With trainer (preferable):

```julia
using ConfParser, Random

include("src/pipeline/trainer.jl")
using .trainer

t = init_trainer(
      rng, 
      conf, # See config directory for examples
      dataset_name; 
      img_resize = (16,16), # Resize for prototyping
      file_loc = loc
)
train!(t)
```

Without trainer:

```julia
using Random, Lux, Enzyme, ComponentArrays, Accessors

include("src/KAEM/KAEM.jl")
include("src/KAEM/model_setup.jl")
include("src/utils.jl")
using .KAEM_model
using .ModelSetup
using .Utils

model = init_KAEM(
      dataset, 
      conf, 
      x_shape; 
      file_loc = file_loc, 
      rng = rng
)

# MLIR-compiled loss, (slow to compile, fast to run, see https://mlir.llvm.org/).
x, loader_state = iterate(model.train_loader)
x = pu(x)
model, ps, st_kan, st_lux = prep_model(model, x; rng = rng) 
loss, grads, st_ebm, st_gen = model.loss_fcn(
      ps,
      st_kan,
      st_lux,
      model,
      x;
      train_idx = 1, # Only affects temperature scheduling in thermo model
      rng = Random.default_rng()
)

# States reset with Accessors.jl:
@reset st.ebm = st_ebm
@reset st.gen = st_gen
```

## Citation/license [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

```bibtex
@misc{raj2025kolmogorovarnoldenergymodelsfast,
      title={Kolmogorov-Arnold Energy Models: Fast and Interpretable Generative Modeling}, 
      author={Prithvi Raj},
      year={2025},
      eprint={2506.14167},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.14167}, 
}
```
