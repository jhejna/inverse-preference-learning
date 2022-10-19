# Few-Shot Preference Learning for Human-in-the-Loop RL

## Installation

First cloen the repository. Then complete the following steps:
1. Edit `environment_cpu.yaml` and `environment_gpu.yaml` as desired to include any additional dependencies via conda or pip. You can also change the name if desired. This is needed if you are going to use PolyMetis for real robot experiments. You can view their documentation [here](https://facebookresearch.github.io/fairo/polymetis/installation.html).
2. Create the conda environment using `conda env create -f environment_<cpu or gpu>.yaml`.
3. Install the research package via `pip install -e research`.
4. Modify the `setup_shell.sh` script by updated the appropriate values as needed. The `setup_shell.sh` script should load the environment, move the shell to the repository directory, and additionally setup any external dependencies. You can add any extra code here.

## Usage
You should be able to activate the development enviornment by running `. path/to/setup_shell.sh`. This is the same environment that will be activated when running jobs locally or on SLURM.

To train a model, simply run `python scripts/train.py --config path/to/config --path path/to/save/folder`

To run a sweep (files ending in `.json`), use the scripts in the `tools` folder for running locally or on SLURM. Our experiments were run on SLURM.
Results can be viewed on tensorboard.

To launch any job via a script in the `tools` folder, use the `--entry-point <path to script>` argument to specify the path to the target script (`scripts/train.py` by default) and the `--arguments <arg1>=<value1>  <arg2>=<value2> ..  <argk>=<valuek>` to specify the arguments for the script. Multiple different jobs can be stacked. For example, `--arguments` can be provided more than once to specify different sets of arguments.

## MetaWorld Setup
By default, the MetaWorld ML10 benchmark includes some tasks used for evaluation in Standard preference based learning papers. Thus, we modify ML10 to slightly tweak the default environment configuration.  Specifically, we modify `miniconda3/envs/<env name>/lib/python3.8/site-packages/metaworld/envs/mujoco/env_dict.py` to have `('button-press-topdown-v2', SawyerButtonPressTopdownEnvV2)`.

### Local Jobs
Launching jobs locally can easily be done by specifying `--cpus` in the same manner you would for `taskset` and `--gpus` via nvidia devices. Note that multi-gpu training is not yet supported. The script will automatically balance jobs on the provided hardware. Here is an example:
```
python tools/run_local.py scripts/my_custom_script.py --cpus 0-8 --gpus 0 1 --seeds-per-job 2 --arguments <arg1>=<value1>  <arg2>=<value2>
```
This will run one job on cores 0-3 with GPU 0 and one job on cpus 4-7 with GPU 1.

### SLURM
Launching jobs on SLURM is done via the `tools/run_slurm.py` script. In addition to the base arguments for job launching, the slurm script takes several additional arguments for slurm. Here is an example command that includes all of the required arguments and launches training jobs from `scripts/train.py`. Additional optional arguments can be found in the `tools/run_slurm.py` file.
```
python tools/run_slurm.py --partition <partition> --cpus 8 --gpus 1080ti:1 --mem 16G --job-name example --arguments config=configs/example.yaml path=../output/test
```
The `gpu` argument takes in the GRES specification of the GPU resource. One unfortunate problem with GRES it doesn't allow sharing GPUs between slurm jobs. This often means that if you want to run a small model that only consumes, say 25% of the GPUs max FLOPS, everyone else on the cluster will still be blocked from using the GPU. The `--jobs-per-instance` argument allows you to train multiple models on the same SLURM node in parallel on the same GPU! You just need to make sure to specify enough CPU and memory resources to run both at once. Doing so drastically saves GPU resources on the cluster if you are running parameter sweeps.

### Using the Sweeper
The default training script `scripts/train.py` also supports parameter sweeps. Parameter sweeps are specified using `json` files. Any of the scripts in `tools` will automatically detect that a sweep is being run based on the `entry-point` and type of config file (json) being specified. An example sweep file is found on the `vision` branch. Keys in the json file are specified via strings, with periods to separate nested structions, and values are provided as lists. For example to specify a learning rate sweep, one would add `"optim_kwargs.lr" : [0.01, 0.001]`.

There are two special keys. The first `"base"` is required and specifies the path to the base config that will be modified by the sweep file. The second, `"paired_keys"` allows you to pair the values of differnet parameters in the sweep.

## Reproducing Results

In order to reproduce the results from the paper, pretraining datsets must be generated. This can be done in parallel using the scripts in the `scripts` folder via the following commands:

MetaWorld:
```
python tools/run_slurm.py --partition <partition> --cpus 12 --mem 32G --job-name ml10-dataset --entry-point scripts/metaworld/collect_policy_dataset.py --arguments benchmark=ml10 tasks-per-env=25 cross-env-ep=10 within-env-ep=25 expert-ep=15 random-ep=2 epsilon=0.1 num-workers=10 noise-type=gaussian path=datasets/mw
```

Point Mass:
```
python tools/run_slurm.py --partition <partition> --cpus 2 --mem 8G --job-name reaching-dataset --entry-point scripts/goal_reaching/collect_random_dataset.py --arguments num-steps=25000 env=point_mass_random path=datasets/pm
```

Reacher:
```
python tools/run_slurm.py --partition <partition> --cpus 2 --mem 8G --job-name reaching-dataset --entry-point scripts/goal_reaching/collect_random_dataset.py --arguments num-steps=25000 env=reacher path=datasets/reacher
```

Panda:
```
python tools/run_slurm.py --partition <partition> --cpus 1 --mem 4G --job-name panda-dataset --seeds-per-job 15 --entry-point scripts/panda/collect_reach_dataset.py --arguments num-steps=10000 num-tasks=5 noise-magnitude=0.8 path=datasets/panda_reach

# Then train the SAC Models for block pushing
python tools/run_slurm.py --partition <partition> --cpus 8 --mem 28G --job-name reach-sac --gpus 1 --jobs-per-instance 4 --arguments config=configs/panda/block_push_sac.json path=output/block_sac

python tools/run_slurm.py --partition <partition> --cpus 1 --mem 4G --job-name panda-dataset --seeds-per-job 16 --entry-point scripts/panda/collect_block_dataset.py --arguments random-ep=8 expert-ep=50 cross-ep=5 policy-noise=0.15 policies=../output/human_rl/05_30/panda/block_push_sweep/ init-noise=0.3 path=output/block_sac
```

Then, train the MAML models using the provided configs after modifying them to include the dataset path.

Finally, our models can be trained by editting the sweep files to include the trained MAML checkpoint.
