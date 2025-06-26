#!/bin/bash
#
#SBATCH --job-name="cross-lin-viz"
#SBATCH --comment="Test von LLMs"
#SBATCH --mail-type=ALL
#SBATCH --mail-user="adrian.muelthaler@campus.lmu.de"
#SBATCH --chdir=/home/m/muelthaler/Projects/cross-lin-viz
#SBATCH --output=/home/m/muelthaler/Projects/cross-lin-viz/slurm.%j.%N.out
#SBATCH --ntasks=1

source venv/bin/activate
python3 -u test_model.py --model "Qwen/Qwen2.5-3B-Instruct" --dataset "nairdanus/VEC_prompt_en" --category "color"