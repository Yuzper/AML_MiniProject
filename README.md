# Project Documentation

## Team Members
Ivan Petrov, Jesper Terkildsen and Ana Ana Vera Vázquez

### Central Problem, Domain, Data Characteristics
Problem at hand: Detecting AI generated text to distinguish human-written and machine-generated content.

Domain: NLP/Text classification, specifically AI generated text detection.

Data: RAID (Robust AI Detection) dataset, containing binary-labeled text samples:
 - Class 0: Human-written text
 - Class 1: AI-generated text

The full dataset contains 6 million text samples combined from both human written and AI generated content, spanning 8 different domains such as, abstracts, books, Reddit posts and recipes. The data contains additional meta data for the AI generated samples about the underliying model (varies GPT models, Llama etc), decoding strategies (Greedy, Random, Greedy w/ Rep Penalty, and Sampling w/ Rep Penalty) and adversarial attacks used to modify text to decieve the detectors.
After pre-processing the data is structured into training, validation, and test splits, with combined fields (e.g., title, generation) into unified text samples. The dataset enables both rapid debugging (small subsets) and larger-scale experimentation for the cluster.

![alt text](https://github.com/Yuzper/AML_MiniProject/tree/main/readme_helper/RAID_data_distribution.PNG "RAID_data_distribution.PNG")

### Central Method: Chosen Architecture and Training Mechanisms
The project uses BERT-based architectures (`distilbert-base-uncased` for local, `bert-base-uncased` and `RoBERTa-base-uncased` in cluster training) for sequence classification. These models are fine-tuned on the previous mentioned RAID dataset using the Hugging Face Transformers library.

The training pipeline includes:
- Tokenization with the specified tokenizer (e.g., `bert-base-uncased`).
- Binary label encoding (`human` → 0, `machine` → 1).
- Training with PyTorch and the Hugging Face `Trainer` API, which supports features like early stopping and checkpointing.

DistilBERT is used for low resource local experimentation due to its lighter computational load and faster training times.
BERT is used as baseline for full-scale training on the HPC cluster. The choice of BERT models is motivated by their strong performance on various text classification tasks.
Additionally, RoBERTa is introduced to explore the benefits of its optimized pre-training and larger training corpus. RoBERTa has demonstrated superior generalization in many NLP benchmarks, making it a promising candidate for better detection task.

The training script ensures reproducibility with random seeds sat and saving the best model checkpoint for later evaluation. In addition the `RoBERTa` model is introduced with the hope of utilizing its larger corpus and training data to the task.


### Key Experiments & Results
Both RoBERTa and bert_full achieved high accuracies on our test set, being 97.14% and 99.59% respectfully. RoBERTa was trained for an epoch less and therefore could possibly achieve an accuracy more on par with the BERT model, however due to time limitations and compute resource constrain this is left as further work.  


### Discussion
Our bert model made 304 mispredictions with 6 of them being false positives. Meaning that it more often than not overpredicted AI text. Depending on the domain this could be a good trade-off and the current model currently minimizes the amount of AI generated text that can get through it's detection, at the cost of more manual review of tagged content.


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
    --split val \
    --batch-size 8 \
    --save-preds
```

Evaluation metrics are saved as JSON under `outputs/metrics/<run_name>_eval_metrics.json`. Optionally, per-example predictions are saved as CSV under `outputs/predictions/`.
