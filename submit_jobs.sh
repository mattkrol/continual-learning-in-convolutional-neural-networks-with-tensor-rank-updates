#!/bin/bash
#
# Submit all job sbatch scripts in the jobs/scripts directory to slurm.
#
# Author: Matt Krol

num_jobs=$(ls jobs/scripts | wc -l)

[ "$num_jobs" = "0" ] && echo "No jobs were found." && exit

echo "Attempting to submit $num_jobs jobs ..."

for job in $(ls jobs/scripts)
do
    sbatch "jobs/scripts/${job}"
done

echo "Successfully submitted $num_jobs jobs!"
