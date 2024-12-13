#!/bin/bash -l
#SBATCH --chdir ./slurm/
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 00:30:00
#SBATCH --gres gpu:1

cd ../
#python ../test.py
# The --reservation line only works during the two 1-week periods
# when 80 GPUs are available. Remove otherwise.
module load gcc python

# Ensure the desired Python version is loaded, or install it if necessary
# This might be specific to your cluster's module system
#module load python/3.10.12

# You only need to create this virtualenv once
# Feel free to replace the name "course_py-3.10" with your own environemnt name 
# virtualenv --system-site-packages ~/venvs/course_py-3.10
virtualenv --system-site-packages -p python3.10 ~/venvs/mnlp_project

# Activate the virtualenv everytime before you run the job
# Make sure the name matches the one you created
source ~/venvs/mnlp_project/bin/activate

# upgrade pip the first time you load the environment
pip install --upgrade pip

# For the first time, you need to install the dependencies
# You don't have to do the installation for every job.
# Only when you need to update any packages you already installed

# NOTE: Paths are relative to the chdir: /scratch/izar/<put-your-username-here> and the cd
#pip install torch

pip install -r requirements.txt

echo "Dependencies have been installed, Now running the given code"
source .env 

# Insert code
python src/fine_tune_trocr.py 

echo "Batch is complete"
sleep 2
