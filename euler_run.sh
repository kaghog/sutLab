module load stack/2024-06
module load gcc/12.2.0
module load python/3.10.13
module load openjdk/21.0.3_9
module load maven
module load eth_proxy

source /cluster/project/adey/kaghog/env_sutlab/bin/activate


#sbatch -n 1 --cpus-per-task=24 --time=12:30:00 --mem-per-cpu=8192 --wrap="python3 -m synpp config_local_hannover.yml"
sbatch -n 1 --cpus-per-task=8 --time=2:30:00 --mem-per-cpu=32192 --wrap="python3 -m synpp config_bogota.yml"

