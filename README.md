# Safe Sim2Real
A collection of algorithms and experiment tools for safe sim to real transfer in robotics.

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

