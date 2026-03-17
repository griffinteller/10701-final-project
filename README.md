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
```

## Structure
`src/`: core project scripts
  - `lstm.py`: LSTM-based model, layers, and helpers
  - `transformer.py`: Transformer-based model, layers, and helpers
  - `ssm.py`: SSM-based model, layers, and helpers
  - `tokenizer.py`: Tokenization of strings. Currently, just vocab-based tokenization.

`tests/`: pytests