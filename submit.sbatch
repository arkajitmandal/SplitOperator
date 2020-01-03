#!/bin/bash
#SBATCH -p action -A action
#SBATCH -o my_output_%j
#SBATCH --mem-per-cpu=60GB
#SBATCH -t 50:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
python split.py
