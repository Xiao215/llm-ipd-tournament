#!/bin/bash
#SBATCH --job-name=IPD_sim
#SBATCH --gres=gpu:a40:2
#SBATCH --time=5:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --mail-user=xiaoo.zhang@mail.utoronto.ca
#SBATCH --mail-type=END,FAIL

# Run
source venv/bin/activate
export PYTHONPATH=.

python3 script/run_game.py --config prisoner_dilemma_toy.yaml --log