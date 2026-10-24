import os
import sys
import re
import math # For ULM if -math.inf is used, though not directly here

# Ensure scripts from the current directory can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Attempt to import from tokenizer scripts
try:
    from BPE import load_bpe_model, tokenize_word_with_bpe
    BPE_EOW = '</w>' # From BPE.py
except ImportError:
    print("Error: BPE.py not found or cannot import BPE functions.")
    sys.exit(1)

try:
    from BBPE import load_bbpe_model, tokenize_word_with_bbpe
except ImportError:
    print("Error: BBPE.py not found or cannot import BBPE functions.")
    sys.exit(1)

try:
    from WordPiece import load_wordpiece_model, tokenize_word_with_wordpiece, UNK_TOKEN as WP_UNK_TOKEN, WORD_SUFFIX as WP_WORD_SUFFIX
except ImportError:
    print("Error: WordPiece.py not found or cannot import WordPiece functions.")
    sys.exit(1)

try:
    from ULM import load_ulm_model, tokenize_word_with_ulm, UNK_TOKEN as ULM_UNK_TOKEN
except ImportError:
    print("Error: ULM.py not found or cannot import ULM functions.")
    sys.exit(1)

# Define Model Paths (these should match the filenames used in the individual training scripts)
BPE_MODEL_BASE = "bpe_char_model_test" # From BPE.py example
BBPE_MODEL_BASE = "bbpe_test_model"   # From BBPE.py example
WORDPIECE_MODEL_BASE = "wordpiece_test_model" # From WordPiece.py example
ULM_MODEL_BASE = "ulm_test_model"     # From ULM.py example

# For WordPiece, define a basic character set for tokenization fallback
# This should ideally match the alphabet used during its training for consistency.
# If the WordPiece training derived its alphabet, we use a similar derivation here for OOV handling by the tokenizer.
# For this comparison script, we'll define a broad set.
DEFAULT_WP_INITIAL_ALPHABET_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz0123456789.,!?'\"- " + 
    "你好世界こんにちは" # Add some common multi-byte characters from test sentences
)


def format_bbpe_tokens(byte_tokens_list):
    """Helper to format BBPE byte tokens for readable printing."""
    formatted = []
    for token_tuple in byte_tokens_list:
        try:
            # Attempt to decode for readability, show bytes as backup
            decoded_str = bytearray(token_tuple).decode('utf-8', errors='replace')
            formatted.append(f"'{decoded_str}'{token_tuple}")
        except Exception:
            formatted.append(str(token_tuple))
    return formatted


def main():
    print("--- Tokenizer Comparison Script ---")

    # 1. Load Models
    print("\n--- Loading Models ---")
    bpe_vocab, bpe_merge_rules = None, None
    bbpe_vocab, bbpe_merge_rules = None, None
    wp_vocab_list, wp_vocab_set = None, None
    ulm_vocab_probs, ulm_max_len = None, None

    try:
        print(f"Loading BPE model from base: {BPE_MODEL_BASE}...")
        bpe_vocab, bpe_merge_rules = load_bpe_model(BPE_MODEL_BASE)
        if bpe_vocab is None: raise FileNotFoundError
        print("BPE model loaded.")
    except FileNotFoundError:
        print(f"BPE model files ({BPE_MODEL_BASE}.vocab, {BPE_MODEL_BASE}.merges) not found.")
        print("Please train the BPE model by running BPE.py first.")
        # sys.exit(1) # Option to exit, or continue without this tokenizer

    try:
        print(f"Loading BBPE model from base: {BBPE_MODEL_BASE}...")
        bbpe_vocab, bbpe_merge_rules = load_bbpe_model(BBPE_MODEL_BASE)
        if bbpe_vocab is None: raise FileNotFoundError
        print("BBPE model loaded.")
    except FileNotFoundError:
        print(f"BBPE model files ({BBPE_MODEL_BASE}.bvocab, {BBPE_MODEL_BASE}.bmerges) not found.")
        print("Please train the BBPE model by running BBPE.py first.")
        # sys.exit(1)

    try:
        print(f"Loading WordPiece model from base: {WORDPIECE_MODEL_BASE}...")
        wp_vocab_list = load_wordpiece_model(WORDPIECE_MODEL_BASE)
        if wp_vocab_list is None: raise FileNotFoundError
        wp_vocab_set = set(wp_vocab_list)
        print("WordPiece model loaded.")
    except FileNotFoundError:
        print(f"WordPiece model file ({WORDPIECE_MODEL_BASE}.wp_vocab) not found.")
        print("Please train the WordPiece model by running WordPiece.py first.")
        # sys.exit(1)

    try:
        print(f"Loading ULM model from base: {ULM_MODEL_BASE}...")
        ulm_vocab_probs = load_ulm_model(ULM_MODEL_BASE)
        if ulm_vocab_probs is None: raise FileNotFoundError
        ulm_max_len = max(len(sw) for sw in ulm_vocab_probs.keys()) if ulm_vocab_probs else 0
        print(f"ULM model loaded (max subword len: {ulm_max_len}).")
    except FileNotFoundError:
        print(f"ULM model file ({ULM_MODEL_BASE}.ulm_model) not found.")
        print("Please train the ULM model by running ULM.py first.")
        # sys.exit(1)

    # 2. Input Text
    sample_sentences = [
        "This is a simple test sentence.",
        "hello world",
        "你好世界 from BBPE and ULM.", #你好世界
        "こんにちは an unlikelihood", # こんにちは
        "antidisestablishmentarianism",
        "點樣 ??", # Cantonese example, might be OOV for some
        "여러가지 다양한 단어들" # Korean example
    ]
    
    # Or allow user input:
    # user_input = input("Enter a sentence to tokenize (or press Enter for default samples): ")
    # if user_input:
    #     sample_sentences = [user_input]

    print("\n--- Tokenization Results ---")

    for sentence in sample_sentences:
        print(f"\nInput Sentence: \"{sentence}\"")
        
        # Simple whitespace split for words, as tokenizers are word-based
        # More sophisticated sentence-to-word tokenization could be used here (e.g., handling punctuation better)
        # For now, use regex to get "words" similar to how training scripts might define them.
        words = re.findall(r'\S+', sentence) # Basic split by any whitespace
                                             # Or use re.findall(r'\b\w+\b|[\.,\?\!]', sentence.lower()) for more refined words + punctuation

        # BPE
        if bpe_merge_rules:
            print("\nBPE Tokens:")
            full_bpe_tokens = []
            for word in words:
                # BPE expects words without EOW, adds it internally
                bpe_word_tokens = tokenize_word_with_bpe(word.lower(), bpe_merge_rules)
                print(f"Word: {word} -> {bpe_word_tokens}")
                full_bpe_tokens.append(bpe_word_tokens)
            print(f"Full: {full_bpe_tokens}")
        
        # BBPE
        if bbpe_merge_rules:
            print("\nBBPE Tokens:")
            full_bbpe_tokens = []
            for word in words:
                bbpe_word_tokens = tokenize_word_with_bbpe(word, bbpe_merge_rules) # BBPE handles case internally if needed, but usually trained on lower
                print(f"Word: {word} -> {format_bbpe_tokens(bbpe_word_tokens)}")
                full_bbpe_tokens.append(bbpe_word_tokens)
            # print(f"Full (raw): {full_bbpe_tokens}") # Raw output might be too verbose

        # WordPiece
        if wp_vocab_set:
            print("\nWordPiece Tokens:")
            full_wp_tokens = []
            # WordPiece tokenizer needs the initial character alphabet for OOV handling.
            # Using a default derived set for this comparison script.
            wp_initial_alphabet = DEFAULT_WP_INITIAL_ALPHABET_CHARS 
            for word in words:
                # WordPiece usually handles casing by having separate tokens or by lowercasing.
                # Our implementation in WordPiece.py's train_wordpiece implies lowercasing at input.
                # The tokenizer itself might not lowercase, relying on vocab content.
                # For safety, let's assume words are lowercased if vocab was built on lowercase.
                # However, the `initial_alphabet_chars` in `get_initial_vocab_and_word_counts` filters chars.
                # The `tokenize_word_with_wordpiece` needs `initial_alphabet_chars` to know basic chars.
                wp_word_tokens = tokenize_word_with_wordpiece(word, wp_vocab_set, wp_initial_alphabet)
                print(f"Word: {word} -> {wp_word_tokens}")
                full_wp_tokens.append(wp_word_tokens)
            print(f"Full: {full_wp_tokens}")

        # ULM
        if ulm_vocab_probs:
            print("\nULM Tokens:")
            full_ulm_tokens = []
            for word in words:
                # ULM training also lowercased words.
                ulm_word_tokens = tokenize_word_with_ulm(word.lower(), ulm_vocab_probs, ulm_max_len)
                print(f"Word: {word} -> {ulm_word_tokens}")
                full_ulm_tokens.append(ulm_word_tokens)
            print(f"Full: {full_ulm_tokens}")
            
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
    # Add a newline at the end of the script file itself
    # (This comment is for the agent; the script will have a newline by construction)
