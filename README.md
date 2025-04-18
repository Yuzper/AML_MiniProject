Getting started
============
Follow the steps in this README to get your project up and running.


1 Install UV (the new python management tool)
--------------------------
UV is the new python management tool we are using. You can check a demo of basic usage [here](https://docs.astral.sh/uv/).

1. Install uv (check this [page](https://docs.astral.sh/uv/getting-started/installation/) for more details):
* MacOs and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
* Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Verify that the command `uv --version` is recognized.


2 Install the dependencies and test instalation
---------------- 

The dependencies are managed by uv, that installs everything (including python) in a local environment.

1) In the root of your newly created project (where this README is) run `uv sync` in your terminal.* 
2) Test if everything is working as intended by running `uv run pytest`. This will run all the tests available. 
3) Test again that everything is running by importing your new package.
    1. Open a python shell in the created virtual environment by running `uv run python` in your shell.
    2. Try to import your package in that python sheel: `>>> import <YOUR NEW PACKAGE>`.

If all these steps work, the installation was succesfull! 
    
*Note: The sync command will create a virtual environment in your repo (in .venv) and will install all the dependencies in that environment. You can start a python shell from that enviroment with `uv run python`. If using PyCharm or VSCode, make sure this virtual environment is the one used by the python entrepeter of PyCharm or VScode.

3 A few notes on UV usage
----------------------

If you want to install the dependencies run `uv sycn`

IF YOU'RE RUNNING IT LOCALLY, MAKE SURE TO SWITCH FROM TENSORFLOW-GPU TO TENSORFLOW

To add new packages, you just need to do `uv add <package_name>`.

If you want to run a python command using the virtual environment created by uv with your project dependencies, do `uv run python <your_python_file>.py`

4 Preprocessing the RAID Dataset
----------------------

The preprocessing pipeline prepares the RAID dataset for training and evaluation. It supports two modes:

- **Local mode**: Processes a small, stratified sample of the dataset for quick testing.
- **Production mode**: Processes the full dataset with official splits.

To run the preprocessing:

- Local mode: `python -m src.data_preprocessing --local --sample-size 3000`
- Production mode: `python -m src.data_preprocessing --run-name raid_full --prod`

Processed datasets are saved under `data/processed/<run_name>/<split>/`.

5 Training the BERT Detector
----------------------

The training script trains a BERT-based model on the preprocessed RAID dataset. It supports saving model checkpoints and training metrics.

To train the model:

- Local debug run:
  ```bash
    python -m src.train \
        --dataset-root data/processed/raid_local \
        --model-name distilbert-base-uncased \
        --epochs 2 --run-name distil_local
  ```

- Full production run:
  ```bash
    python -m src.train \
        --dataset-root data/processed/raid_full \
        --model-name bert-base-uncased \  
        --epochs 5
  ```

Model checkpoints are saved under `outputs/checkpoints/<run_name>/`, and training metrics are saved as JSON under `outputs/metrics/`.




