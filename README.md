# Inverse Prefrence Learning: Preference-based RL without a Reward Function

Code for [*Inverse Prefrence Learning: Preference-based RL without a Reward Function*](https://arxiv.org/pdf/2305.15363.pdf) by Joey Hejna and Dorsa Sadigh.

This repository is based on a frozen version of [research-lightning](https://github.com/jhejna/research-lightning). For detailed information about how to use the repository, see the research lightning repository.

## Installation

Complete the following steps:
1. Clone the repository to your desired location using `git clone`.
2. Create the conda environment using `conda env create -f environment_<cpu or gpu>.yaml`.
3. Install the repository research package via `pip install -e research`.
4. Modify the `setup_shell.sh` script by updating the appropriate values as needed. The `setup_shell.sh` script should load the environment, move the shell to the repository directory, and additionally setup any external dependencies. All the required flags should be at the top of the file. This is necesary for support with the SLURM launcher, which we use for all experiments.

When using the repository, you should be able to setup the environment by running `. path/to/setup_shell.sh`.

Some experiments require extra setup steps, which we detail below.

### Robomimic Tasks
In order to run the robomimic experiments, you need to install the [robomimic](https://robomimic.github.io/docs/introduction/installation.html) package and the [robosuite](https://robosuite.ai/) package. We install these dependencies in the following manner:

Robosuite:
1. Git clone the robosuite repository, found [here](https://github.com/ARISE-Initiative/robosuite).
2. Checkout the `offline_study` branch
3. install the package to the conda environment without dependencies via `pip install -e . --no-dependencies`.
4. Relevant as of 2/14/2023: Robosuite has not updated their package to be compatible with Python 3.10. Change `from collections import Iterable` to `from collections.abc import Iterable` in `robosuite/models/arenas/multi_table_arena.py` and `robosuite/utils/placement_samplers.py`.
5. In the python interpreter, run `import robosuite` until it completes. If any errors show up, install the missing packages via pip.

Robomimic:
1. Git clone the robomimic repository, found [here](https://github.com/ARISE-Initiative/robosuite).
2. Install the package to the conda environment without dependencies via `pip install -e . --no-dependencies`.
3. In the python interpreter, run `import robomimic` until it completes. If any errors show up, install the missing packages via pip.
4. Download the datasets per the instructions [here](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html)

Finally, make sure to edit the files in `configs/robomimic` to correctly point to the download locations of the dataset.

### MetaWorld Setup
By default, the MetaWorld benchmark has a discrepancy in environment naming. For dataset creation, we modify this to better support automatically getting oracle policies from the environment name. Specifically, we modify `miniconda3/envs/<env name>/lib/python3.8/site-packages/metaworld/envs/mujoco/env_dict.py` to have `('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2)`.

## Usage
You should be able to activate the development enviornment by running `. path/to/setup_shell.sh`. This is the same environment that will be activated when running jobs locally or on SLURM.

To train a model, simply run `python scripts/train.py --config path/to/config --path path/to/save/folder`

All experiments in the paper can be replicated using the `.json` sweep files.  To run a sweep (files ending in `.json`), use the scripts in the `tools` folder for running locally or on SLURM. Our experiments were run on SLURM.

To launch any job via a script in the `tools` folder specify the config and path by adding the argument `--arguments config=path/to/config path=path/to/save/folder`.

### Local Jobs
Launching jobs locally can easily be done by specifying `--valid-cpus`, which system CPUs to use, in the same manner you would for `taskset` and `--valid-gpus` via nvidia devices in the same way you would set `CUDA_VISIBLE_DEVICES`. Then, specify the number of CPUs and GPUs per job with `--cpus` and `--gpus`. Note that multi-gpu training is not yet supported. The script will automatically balance jobs on the provided hardware. Here is an example:
```
python tools/run_local.py scripts/my_custom_script.py --valid-cpus 0-8 --valid-gpus 0 1 --cpus 2 --gpus 1 --arguments config=path/to/config path=path/to/save/folder`
```
This will run one job on cpu core range 0-2 with GPU 0 and one job on cpu core range 2-4 with GPU 1.

### SLURM
Launching jobs on SLURM is done via the `tools/run_slurm.py` script. In addition to the base arguments for job launching, the slurm script takes several additional arguments for slurm. Here is an example command that includes all of the required arguments and launches training jobs from `scripts/train.py`. Additional optional arguments can be found in the `tools/run_slurm.py` file.
```
python tools/run_slurm.py --partition <partition> --cpus 2 --gpus 1080ti:1 --mem 8G --job-name example --arguments config=path/to/config path=path/to/save/
```
The `gpu` argument takes in the GRES specification of the GPU resource. One unfortunate problem with GRES it doesn't allow sharing GPUs between slurm jobs.  The `--jobs-per-instance` argument allows you to train multiple models on a single SLURM allocation in parallel on the same GPU. You just need to make sure to specify enough CPU and memory resources to run both at once.
