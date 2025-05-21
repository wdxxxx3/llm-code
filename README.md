# Subword Tokenization Algorithms

This repository contains Python implementations of several common subword tokenization algorithms used in Natural Language Processing (NLP). These algorithms are fundamental for preparing text data for large language models.

## Implemented Algorithms

The following tokenization algorithms have been implemented:

1.  **BPE (Byte Pair Encoding)** (`BPE.py`):
    *   A character-level implementation of BPE.
    *   Learns a vocabulary of subword units by iteratively merging the most frequent pair of adjacent characters or character sequences.
    *   Includes functionality for training, tokenizing, and saving/loading models.

2.  **BBPE (Byte-level BPE)** (`BBPE.py`):
    *   A byte-level implementation of BPE.
    *   Operates directly on UTF-8 byte sequences, making it suitable for handling diverse languages and unknown characters.
    *   Includes training, tokenization (on byte sequences), and model saving/loading.

3.  **WordPiece** (`WordPiece.py`):
    *   A character-level implementation of the WordPiece algorithm.
    *   Builds a vocabulary by merging pairs of subwords that maximize the likelihood of the training data.
    *   Features include training, tokenization (with `##` prefixes for subword continuations and `[UNK]` for unknown characters), and model saving/loading.

4.  **ULM (Unigram Language Model)** (`ULM.py`):
    *   An implementation of the Unigram Language Model tokenizer.
    *   Starts with a large set of candidate subwords and iteratively prunes them using an EM-like algorithm to find a vocabulary that maximizes the likelihood of the corpus when segmented using Viterbi.
    *   Includes training, Viterbi-based tokenization, and model saving/loading.

## Setup and Usage

No external libraries beyond standard Python are required.

### 1. Training Tokenizer Models

Each tokenizer can be trained by running its respective Python script directly. This will typically train a model on a sample corpus defined within the script and save the model files to the project's root directory.

*   **BPE:**
    ```bash
    python BPE.py
    ```
    (This will create `bpe_model.vocab` and `bpe_model.merges` or similar, as defined in the script's main block)

*   **BBPE:**
    ```bash
    python BBPE.py
    ```
    (This will create `bbpe_model.bvocab` and `bbpe_model.bmerges` or similar)

*   **WordPiece:**
    ```bash
    python WordPiece.py
    ```
    (This will create `wp_model.wp_vocab` or similar)

*   **ULM:**
    ```bash
    python ULM.py
    ```
    (This will create `ulm_model.ulm_model` or similar)

**Note:** The exact filenames for saved models are defined in the `if __name__ == "__main__":` block of each script. Ensure these match the paths expected by the comparison script if you modify them. The test scripts also create model files, typically in subdirectories within `tests/`.

### 2. Comparing Tokenizers

A script `compare_tokenizers.py` is provided to demonstrate and compare the outputs of the different tokenizers on sample sentences.

Before running the comparison script, ensure that you have trained the models for each tokenizer you wish to compare, as it loads these pre-trained models. The comparison script expects model files to be in the project root, matching the names used during training (e.g., `bpe_char_model_test.vocab`, `bbpe_test_model.bvocab`, etc., as per the `if __name__ == "__main__"` blocks of the individual tokenizer scripts).

```bash
python compare_tokenizers.py
```

The script will print the tokenized output for each algorithm for a set of predefined sentences.

### 3. Running Unit Tests

Unit tests are provided in the `tests/` directory to verify the correctness of each tokenizer implementation.

To run all tests:
```bash
python -m unittest discover tests
```

To run tests for a specific module (e.g., BPE):
```bash
python tests/test_bpe.py
```

## Directory Structure

```
.
├── BPE.py              # Character-level BPE implementation
├── BBPE.py             # Byte-level BPE implementation
├── WordPiece.py        # WordPiece implementation
├── ULM.py              # Unigram Language Model tokenizer implementation
├── compare_tokenizers.py # Script to compare tokenizer outputs
├── README.md           # This file
├── LICENSE             # Project License (currently contains default text)
└── tests/              # Directory for unit tests
    ├── test_bpe.py
    ├── test_bbpe.py
    ├── test_wordpiece.py
    └── test_ulm.py
    # (Plus, subdirectories like test_bpe_model_files/ for test model files, created during tests)
```

## Contributing

Contributions and suggestions are welcome! Please feel free to fork the repository, make changes, and submit pull requests. If you encounter issues or have ideas for improvements, please open an issue.
```
