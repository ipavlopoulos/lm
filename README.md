# Language Modeling Toolkit

Small research-oriented language modeling code for:

- character-level n-gram language models
- word-level n-gram language models
- neural next-word models built with PyTorch LSTM or GRU
- evaluation workflows for predictive medical text experiments

This repository is lightweight and practical. It is most useful if you want to train simple language models, compare count-based and neural approaches, or reproduce the predictive-text experiments used in the accompanying medical-text work.

## What Is In This Repo

The codebase has two main modeling implementations:

- [`markov/models.py`](/Users/iopa3492/vs/lm/markov/models.py): count-based n-gram language models for characters and words
- [`neural/models.py`](/Users/iopa3492/vs/lm/neural/models.py): neural language models based on LSTM / GRU

The main experiment entrypoint is:

- [`run_mlm.py`](/Users/iopa3492/vs/lm/run_mlm.py): dataset loading, preprocessing, repeated train/test sampling, evaluation, and exploratory analysis

Supporting utilities live in:

- [`toolkit.py`](/Users/iopa3492/vs/lm/toolkit.py): preprocessing, accuracy, keystroke-discount, and precision/recall helpers

## Quick Start

### 1. Install dependencies

This project uses the dependencies listed in [`requirements.txt`](/Users/iopa3492/vs/lm/requirements.txt), including PyTorch for the neural language model.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using a newer Python stack, some dependency pinning may still need adjustment.

### 2. Try the n-gram model

```python
from markov.models import LM

wlm = LM(gram="WORD", n=2).train(["this", "is", "an", "example", "this", "is"])
print(wlm.generate_text(grams_num=8))
print(wlm.cross_entropy("this is an example".split()))
```

### 3. Try the neural model

```python
from neural.models import RNN

rnn = RNN(epochs=1, vocab_size=1000)
text = "Jack and Jill went up the hill to fetch a pail of water"
rnn.train(text)
print(rnn.generate_next_gram("Jack and Jill went"))
```

## Models

### Character N-Gram Model

Use [`LM`](/Users/iopa3492/vs/lm/markov/models.py) with `gram="CHAR"` when you want character-level generation or character-based cross-entropy / bits-per-character style evaluation.

```python
from markov.models import LM

clm = LM(gram="CHAR", n=4).train("this is an example this is another")
print(clm.generate_text(grams_num=60))
print(clm.bpc("this is an example"))
```

### Word N-Gram Model

Use [`LM`](/Users/iopa3492/vs/lm/markov/models.py) with `gram="WORD"` for next-word prediction and word-level evaluation.

```python
from markov.models import LM

wlm = LM(gram="WORD", n=3).train("this is an example this is another example".split())
print(wlm.generate_text(grams_num=12))
print(wlm.cross_entropy("this is an example".split()))
```

### Neural Word Model

Use [`RNN`](/Users/iopa3492/vs/lm/neural/models.py) for an LSTM-based model, or set `use_gru=True` for GRU.

```python
from neural.models import RNN

rnn = RNN(epochs=1, vocab_size=5000, use_gru=False)
rnn.train("a small amount of training text goes here")
print(rnn.cross_entropy("a small amount of text"))
```

## Running Experiments

The main CLI is [`run_mlm.py`](/Users/iopa3492/vs/lm/run_mlm.py).

Common modes:

- `--method counts`: evaluate word n-gram models
- `--method lstm`: evaluate the neural LSTM model
- `--method gru`: evaluate the neural GRU model
- `--method explore`: run exploratory vocabulary / lexicon analysis

### Basic examples

Evaluate count-based models:

```bash
python run_mlm.py \
  --method counts \
  --dataset iuxray \
  --repetitions 5 \
  --vocab_size 1000
```

Evaluate GRU:

```bash
python run_mlm.py \
  --method gru \
  --dataset iuxray \
  --epochs 10 \
  --repetitions 3 \
  --vocab_size 5000
```

Run exploratory vocabulary analysis:

```bash
python run_mlm.py \
  --method explore \
  --dataset iuxray \
  --explore_vocab_sensitivity 1
```

### Supported datasets

`run_mlm.py` expects one of:

- `iuxray`
- `mimic`
- a direct path to a `.csv` file

By default, the code looks for:

- `./DATA/iuxray.csv.gz`
- `./DATA/NOTEEVENTS.csv.gz`

### Useful flags

Some of the most important runtime flags are:

- `--dataset`: dataset name or CSV path
- `--dataset_size`: maximum number of reports sampled
- `--test_size`: number of held-out words used for evaluation
- `--vocab_size`: tokenizer vocabulary size for neural models or vocabulary-study ceiling
- `--epochs`: neural training epochs
- `--min_word_freq`: map rare tokens to the OOV token
- `--report_type`: mainly relevant for MIMIC filtering
- `--section_name`: restrict to one report section such as `findings` or `impression`
- `--preprocess`: enable text preprocessing
- `--explore_vocab_sensitivity`: run vocabulary sensitivity analysis for count or neural models
- `--stop_lexicon_path`: CSV lexicon with a `term` column
- `--meds_lexicon_path`: CSV lexicon with a `term` column

## Outputs And Evaluation

Depending on the mode, the code reports or saves:

- next-word prediction accuracy
- keystroke reduction / keystroke discount metrics
- cross-entropy / bits-per-character style metrics
- vocabulary sensitivity studies as CSV
- precision/recall plots for stopword vs medical lexicons

Generated analysis files are typically written into the working directory.

## Repository Layout

```text
.
├── markov/
│   └── models.py
├── neural/
│   └── models.py
├── LAB/
│   └── predictive_keyboard/
├── run_mlm.py
├── toolkit.py
├── mlm.sh
├── mlm_ades.sh
└── requirements.txt
```

## Notes

- This is research code, so the CLI and data assumptions are more experiment-oriented than product-oriented.
- The shell scripts [`mlm.sh`](/Users/iopa3492/vs/lm/mlm.sh) and [`mlm_ades.sh`](/Users/iopa3492/vs/lm/mlm_ades.sh) show example batch runs.
- The notebooks [`LM.ipynb`](/Users/iopa3492/vs/lm/LM.ipynb) and [`nlm_example.ipynb`](/Users/iopa3492/vs/lm/nlm_example.ipynb) provide additional examples.

## Citation

If this repository is useful in your work, please cite:

```bibtex
@inproceedings{pavlopoulos2021customized,
  title={Customized Neural Predictive Medical Text: A Use-Case on Caregivers},
  author={Pavlopoulos, John and Papapetrou, Panagiotis},
  booktitle={Artificial Intelligence in Medicine: 19th International Conference on Artificial Intelligence in Medicine, AIME 2021, Virtual Event, June 15--18, 2021, Proceedings},
  pages={438--443},
  year={2021},
  organization={Springer}
}
```
