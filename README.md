# Safe Exploration via Policy Priors
This repository contains an implementation of SOOPER, as introduced in the paper Safe Exploration via Policy Priors.

## Requirements

- Python == 3.11.6
- `venv` or `Poetry`

## Installation

### Using pip

```bash
git clone https://github.com/anon/safe-sim2real
cd safe-sim2real
python3 -m venv venv
source venv/bin/activate
pip install -e .
````

### Using Poetry

```bash
git clone https://github.com/anon/safe-sim2real
cd safe-sim2real
poetry install
poetry shell
```

## Usage

Our code uses [Hydra](https://hydra.cc/) to configure experiments. Each experiment is defined as a `yaml` file in `ss2r/configs/experiments`. For example, to train the RCCar sim-to-real experiment, run:

```bash
python train_brax.py +experiment=rccar_sim_to_real
```

<!-- ## Citation

If you find our repository useful in your work, please consider citing:

```bibtex
``` -->

<!-- ## Learn More

* **Project Webpage**: 
* **Paper**:
* **Contact**: 

