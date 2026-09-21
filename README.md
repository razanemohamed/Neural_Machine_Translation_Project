# French-to-English Neural Machine Translation with PyTorch

This project contains a notebook-based implementation of a neural machine translation (NMT) system using a Transformer-inspired architecture. The notebook focuses on translating from French to English by combining:

- PyTorch
- Hugging Face `transformers`
- Hugging Face `datasets`
- custom multi-head attention implementation
- parallel corpus preprocessing and tokenization

## Overview

The notebook demonstrates a sequence-to-sequence translation pipeline for a French-to-English dataset. It includes:

- downloading and extracting a parallel corpus
- loading train, validation, and test splits
- inspecting dataset samples
- loading tokenizer files from local resources
- tokenizing source and target sentences
- preparing encoder and decoder input IDs
- building the attention mechanism used in a Transformer-style model

This project is intended as an educational and research-oriented implementation of core machine translation concepts.

## Project Goals

- understand how a parallel translation dataset is structured
- prepare bilingual text for model training
- tokenize French and English text for sequence modeling
- create encoder/decoder input representations
- build a custom multi-head attention layer with masking
- use the notebook as a base for extending into a full training pipeline

## Technical Stack

- Python
- PyTorch
- Hugging Face `datasets`
- Hugging Face `transformers`
- `requests` and `zipfile` for dataset downloading

## Repository Structure

```text
Neural_Machine_Translation_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── Neural_Machine_Translation.ipynb
└── resources/
    ├── parallel_en_fr_corpus/
    ├── tokenizer_fr/
    └── tokenizer_en/
```

> The notebook downloads the dataset into a `resources/` directory when run and expects the extracted corpus to contain train, validation, and test splits.

## Dataset

The notebook downloads a zip file from a public project resource and extracts it locally. The expected folder structure is:

```text
resources/
├── parallel_en_fr_corpus/
│   ├── train/
│   ├── validation/
│   └── test/
├── tokenizer_fr/
└── tokenizer_en/
```

The dataset contains parallel sentence pairs, where:

- `text_fr` contains French sentences
- `text_en` contains English sentences

The notebook then maps each example into:

- `encoder_input_ids`
- `decoder_input_ids`

## Notebook Workflow

The notebook is organized in the following stages:

1. Install required dependencies
2. Download and extract the data
3. Load the dataset splits using `datasets.Dataset`
4. Print sample examples from each split
5. Load source and target tokenizers
6. Inspect tokenization behavior with example text
7. Map raw text samples to token IDs for model inputs
8. Define a custom `MultiHeadAttention` module
9. Add logic for:
   - linear projections for queries, keys, and values
   - attention head reshaping
   - masking for padding and causality
   - softmax attention and context aggregation

## Model Details

The model implementation in the notebook includes a custom attention block with:

- query, key, and value projections
- multi-head splitting
- scaled dot-product attention
- padding mask handling
- causal mask handling for decoder self-attention
- output projection

The notebook uses PyTorch `nn.Module` classes and follows a Transformer-inspired approach to sequence modeling.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Neural_Machine_Translation_Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook notebooks/Neural_Machine_Translation.ipynb
```

Or in VS Code:

- open the notebook file
- select the correct Python interpreter
- run the cells from top to bottom

## Sample Translations

The current model is still experimental and produces imperfect translations. The examples below reflect the actual behavior observed in the notebook output.

| Input | Gold | Prediction |
|---|---|---|
| vous me mettez mal a l aise . | you re embarrassing me . | you re putting me . |
| c est toi le professeur . | you re the teacher . | you re the teacher . |
| elle sourit avec bonheur . | she smiled happily . | she stared him . |
| je ne suis pas devin . | i m not a psychic . | i m not low . |
| vous allez perdre . | you re going to lose . | you re going to love it . |

These examples show that the model can capture some basic sentence patterns and short translations, but it still struggles with nuance, contextual meaning, and grammatical accuracy. This is expected for an early notebook-based sequence-to-sequence translation implementation.

## Notes

- This project is intended for learning and experimentation.
- The notebook includes a dataset download step, so internet access is required when running it for the first time.
- The project is best run in a notebook environment such as Jupyter, Google Colab, or VS Code notebooks.
- Some parts of the notebook are designed as implementation exercises and can be expanded into a full training loop.

## Possible Improvements

This notebook can be extended with:

- full encoder-decoder training loop
- validation metric tracking
- BLEU score evaluation
- model checkpoint saving
- beam search decoding for translation generation
- a script version outside the notebook
- a cleaner command-line interface for experimentation

## License

This project is provided for educational and research use. If you plan to publish it publicly, consider adding an open-source license such as MIT.
