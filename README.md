# Project Documentation

## Submission
Submit via link to your github.itu.dk repo (provide access to stehe, chur, and star) or Colab notebook.

## Documentation
Provide informal documentation in this README or directly in a Jupyter notebook to document/describe:
- **Names of all involved team members**
- **Central problem, domain, data characteristics**
- **Central method**: chosen architecture and training mechanisms, with a brief justification if non-standard
- **Key experiments & results**: present and explain results, e.g., in simple accuracy tables, error graphs, or visualizations of representations and/or edge cases – keep it crisp.
- **Discussion**: summarize the most important results and lessons learned (what is good, what can be improved).

### Central Problem, Domain, Data Characteristics (AI Generated - gotta modify)
The central problem addressed in this project is detecting AI-generated text using the RAID dataset. The RAID dataset is a benchmark dataset designed for anti-AI-text detection, containing examples of both human-written and machine-generated text. The dataset is split into training, validation, and test sets, and includes text samples with binary labels (`human` or `machine`). The preprocessing pipeline combines relevant fields (e.g., `title` and `generation`) into a unified `text` column for consistency across splits. The dataset supports both local debugging with small samples and full-scale production runs.

### Central Method: Chosen Architecture and Training Mechanisms (AI Generated - gotta modify)
The project uses BERT-based architectures (`bert-base-uncased` and `distilbert-base-uncased`) for sequence classification. These models are fine-tuned on the RAID dataset using the Hugging Face Transformers library. The training pipeline includes:
- Tokenization with the specified tokenizer (e.g., `bert-base-uncased`).
- Binary label encoding (`human` → 0, `machine` → 1).
- Training with PyTorch and the Hugging Face `Trainer` API, which supports features like early stopping and checkpointing.

The choice of BERT-based models is motivated by their strong performance on text classification tasks. DistilBERT is used for faster experimentation due to its reduced size, while BERT is used for full-scale production runs to achieve higher accuracy. The training script ensures reproducibility by setting random seeds and saving the best model checkpoint for evaluation.

---

## TO RUN THE PROJECT

### ENV Setup - For Local Development

#### 1. Install UV (the new Python management tool)
UV is the Python management tool we are using. You can check a demo of basic usage [here](https://docs.astral.sh/uv/).

1. Install UV (check this [page](https://docs.astral.sh/uv/getting-started/installation/) for more details):
   - MacOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Verify that the command `uv --version` is recognized.

#### 2. Install the dependencies and test installation
1. In the root of your project (where this README is), run `uv sync` in your terminal.
2. Activate the environment by running `source .venv/bin/activate`.
3. Test that everything is running by importing your new package:
   - Open a Python shell in the created virtual environment by running `uv run python` in your shell.
   - Try to import your package: `>>> import <YOUR NEW PACKAGE>`.

#### Notes on UV usage
- To install dependencies: `uv sync`
- To add new packages: `uv add <package_name>`
- To run Python commands using the virtual environment: `uv run python <your_python_file>.py`

### ENV Setup - For Cluster
Run `env.slurm` to install the environment.

---

## TASKS

### 1. Preprocessing the RAID Dataset
The preprocessing pipeline prepares the RAID dataset for training and evaluation. It supports two modes:
- **Local mode**: Processes a small, stratified sample of the dataset for quick testing.
- **Production mode**: Processes the full dataset with official splits.

To run the preprocessing:
- Local mode:
  ```bash
  python -m src.data_preprocessing --local --sample-size 100 --tokenizer distilbert-base-uncased --run-name raid_local
  ```
- Production mode:
  ```bash
  python -m src.data_preprocessing --run-name raid_full --tokenizer bert-base-uncased --prod
  ```

Processed datasets are saved under `data/processed/<run_name>/<split>/`.

---

### 2. Training the BERT Detector
The training script trains a BERT-based model on the preprocessed RAID dataset. It supports saving model checkpoints and training metrics.

To train the model:
- Local debug run:
  ```bash
  python -m src.train_torch \
      --dataset-root data/processed/raid_local \
      --model-name distilbert-base-uncased \
      --epochs 2 \
      --batch-size 4 \
      --run-name distil_local
  ```
- Full production run:
  ```bash
  python -m src.train_torch \
      --dataset-root data/processed/raid_full \
      --model-name bert-base-uncased \
      --epochs 5 \
      --batch-size 8 \
      --learning-rate 2e-5 \
      --run-name bert_full
  ```

Model checkpoints are saved under `outputs/checkpoints/<run_name>/`, and training metrics are saved as JSON under `outputs/metrics/`.

---

### 3. Evaluation
Evaluate the fine-tuned BERT/DistilBERT classifier on RAID dataset splits.

To evaluate the model:
```bash
python -m src.evaluate \
    --dataset-root data/processed/raid_full \
    --run-name bert_full \
    --split test \
    --batch-size 8 \
    --save-preds
```

Evaluation metrics are saved as JSON under `outputs/metrics/<run_name>_eval_metrics.json`. Optionally, per-example predictions are saved as CSV under `outputs/predictions/`.