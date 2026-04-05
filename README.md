# English-French Translation (10701 Final Project)

## Setup

```bash
# setup virtual environment, if you want; will be automatically ignored
python3.12 -m venv .venv
source .venv/bin/activate

# install requirements -- you may need to reinstall torch separately for CUDA/MPS/ROCm support, not sure
pip install -r requirements.txt 

# setup project (currently, just downloads data)
python project_setup.py

# split data into train and test
python src/train.py preprocess
```

## Structure
`src/`: core project scripts
  - `lstm.py`: LSTM-based model, layers, and helpers
  - `transformer.py`: Transformer-based model, layers, and helpers
  - `ssm.py`: SSM-based model, layers, and helpers
  - `tokenizer.py`: Tokenization of strings. Currently, just vocab-based tokenization.
  - `train.py`: preprocessing + training code.
    - Preprocess (split into train and test `.csv`s):
      ```
      python src/train.py preprocess
      ```
    - Train:
      ```
      python src/train.py train --train_config configs/<my_train_config>.yaml --model_config configs/<my_model_config>.csv --model <ssm | transformer | lstm>
      ```
    To train, you must make a wandb account for logging, and be added to the project organization (contact Griffin).

`tests/`: pytests
`configs/`: Model and training configs, in yaml format.
  - Training config schema:
  ```python
  class TrainConfig:
    lr: int
    num_epochs: int
    verbose: bool
    train_val_split: float
    batch_size: int
    data_nrows: int | None
  ```

  - Model config schema: depends on model.

