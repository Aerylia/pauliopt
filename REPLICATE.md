# How to replicate the results from the paper:
These are the instructions to replicate the results from the paper available [as preprint](https://arxiv.org/abs/2304.08814).

The experiments were performed using ``python 3.11``  on a SLURM compute cluster using a bash script similar to this:

```[bash]
#!/bin/bash
#SBATCH --job-name=pauliopt_Johannesburg
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1

srun python run_experiment.py Johannesburg 1 2 5 10 50 100
```

Where run_experiment.py can be found in [the scripts folder](scripts/run_experiment.py).

To ensure that the experiments are using the same versions of libraries, the dependencies and their versions can be found in [scripts/requirements.txt](scripts/requirements.txt).
Once these libraries are installed, the pauliopt library can be installed using
```
pip install .
```
from the folder where this file is located.

The obtained results can be found in [scripts/results](scripts/results/) and they can be parsed into the figures from the paper using the provided [jupyter notebook](scripts/parse_results.ipynb).

## Using the run_experiment.py script
The script takes two arguments: the name of the target device and a sequence of the number of gadgets to generate. It will generate 1 circuit for each provided number and compile it using all methods. The results are concatenated to the output csv file. The script is designed to be executed multiple times in parallel on a SLURM cluster.

## Other
I believe this is sufficient to replicate all results from the paper, if you run into any issues, please contact me.