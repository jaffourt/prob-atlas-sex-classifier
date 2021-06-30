#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --mem 200G
#SBATCH -n 6
#SBATCH -c 2
python gridsearch_random_forest.py
