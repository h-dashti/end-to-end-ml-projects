#!/bin/bash

#SBATCH --time=02:00:00          
#SBATCH --nodes=1
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=16      
#SBATCH --mem=16G                
#SBATCH --partition=sapphire      

module load Python
source ~/.venvs/env_ml/bin/activate


python xgb_gridsearch.py
