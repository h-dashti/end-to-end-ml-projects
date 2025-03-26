#!/bin/bash

#SBATCH --time=24:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=32      
#SBATCH --mem=36G                
#SBATCH --partition=sapphire      


module load Python
source ~/.venvs/env_ml/bin/activate


python xgb_gridsearch.py
