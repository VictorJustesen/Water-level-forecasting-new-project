#!/bin/bash
# LSF Batch Job Script for running automl.py

### General LSF options ###

# -- Specify the queue --
# Use a GPU queue appropriate for your models (e.g., gpuv100 or gpua100)
# Remember A100 requires code compiled with CUDA >= 11.0
#BSUB -q gpuv100

# -- Set the job Name --
#BSUB -J AutoML_WaterLevel

# -- Ask for number of cores (CPU slots) --
# Adjust based on data loading/preprocessing needs. 8 is a reasonable start.
#BSUB -n 8

# -- Request GPU resources --
# Request 1 GPU in exclusive process mode.
#BSUB -gpu "num=1:mode=exclusive_process"

# -- Specify that all cores/GPU must be on the same host/node --
#BSUB -R "span[hosts=1]"

# -- Specify memory requested PER CORE/SLOT --
# Example: 8GB RAM per core (total 64GB). ADJUST BASED ON YOUR NEEDS!
#BSUB -R "rusage[mem=8GB]"

# -- Specify memory limit PER CORE/SLOT (Job killed if exceeded) --
# Example: 10GB per core (total 80GB limit). ADJUST BASED ON YOUR NEEDS!
#BSUB -M 9GB

# -- Set walltime limit: hh:mm --
# Max 24:00 for GPU queues. START SHORT (e.g., 1:00) FOR TESTING!
# Adjust based on expected runtime for the full job.
#BSUB -W 16:00

# -- Specify output and error files (%J expands to Job ID) --
# We'll create the 'logs' directory below.
#BSUB -o logs/automl_%J.out
#BSUB -e logs/automl_%J.err

# -- Email notifications (Optional) --
# Uncomment and set your DTU email if desired
##BSUB -u s224296@dtu.dk  # Use your actual email
# Send email on job start (-B) and job end/failure (-N)
##BSUB -B
##BSUB -N

### End of LSF options ###

# --- Environment Setup and Execution --- #

# Create directories for logs and results if they don't exist
# Output (-o) and Error (-e) files go here
mkdir -p logs
# Python script results (e.g., JSON) go here
mkdir -p results

echo "=========================================================="
echo "Job Started on $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Working Directory: $(pwd)"
echo "Requested Cores: $LSB_DJOB_NUMPROC"
echo "Allocated Hosts: $LSB_HOSTS"
echo "Queue: $LSB_QUEUE"
echo "Start Time: $(date)"
echo "=========================================================="

# 1. Load necessary modules
echo "Loading required modules..."
module purge # Start with a clean environment is good practice
# --- !!! IMPORTANT: CHECK AND USE CORRECT MODULE NAMES/VERSIONS FOR DTU HPC !!! ---
source ~/.bashrc
module load cuda/11.8      # Or specific version compatible with GPU queue & TF
# --- !!! END OF MODULE CHECK SECTION !!! ---
echo "Modules loaded:"
module list

# 2. Activate your Conda environment
# Make sure 'forecasting' environment was created successfully with the corrected YAML∂
echo "Activating Conda environment 'forecasting'..."
# Use 'conda activate' if 'source activate' gives issues
conda activate forecastinghpc
# Check activation
echo "Conda environment: $CONDA_DEFAULT_ENV"
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "forecastinghpc" ]; then
    echo "ERROR: Failed to activate conda environment 'forecastinghpc'."
    exit 1
fi
echo "Python path: $(which python)"


# 3. Navigate to the project directory (if submitting from elsewhere)
# If you submit using 'bsub < submit_automl.sh' from WITHIN the project dir,
# this cd might be redundant, but it's safer to include it.
cd ~/school/Water-level-forecasting-new-project || exit 1 # Exit if cd fails
echo "Working directory set to: $(pwd)"

# 4. Run your Python script
echo "Running automl.py..."
# --- !!! IMPORTANT: ADJUST ARGUMENTS FOR YOUR automl.py SCRIPT !!! ---
# Assuming automl.py takes arguments for input and output files.
# Using masterdata.csv as input, adjust if needed (e.g., processed_data.parquet).
# Saving results to a unique file in the 'results' directory.
python -u automl2.py \
    

# Capture the exit status of the python script
status=$?
if [ $status -ne 0 ]; then
    echo "ERROR: Python script automl.py failed with exit status $status"
    exit $status # Exit the batch job with the same error code
fi
# --- !!! END OF PYTHON EXECUTION SECTION !!! ---

echo "Python script finished successfully."

echo "=========================================================="
echo "Job Finished: $(date)"
echo "=========================================================="

exit 0
