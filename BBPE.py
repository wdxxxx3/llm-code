import collections
import re

def get_initial_vocab_and_word_byte_counts(corpus_text):
    """
    1. Splits the raw corpus text into words.
    2. Encodes each word into its UTF-8 byte sequence.
    3. Represents each word's byte sequence as a tuple of integers (0-255).
    4. Counts the frequency of these byte-sequence representations of words.
    5. Builds an initial vocabulary of all unique single bytes found.

    Args:
        corpus_text (str): The raw text corpus.

    Returns:
        tuple: (
            word_byte_counts (collections.Counter): Frequencies of words, where words
                are represented as tuples of byte integers (UTF-8).
                e.g., {(104, 101, 108, 108, 111): 5, ...}.
            initial_byte_vocab (set): The initial set of unique single bytes (integers)
                found in the corpus.
        )
    """
    # Split into words using a similar approach to BPE.py for consistency
    # It's important to define how "words" are segmented before byte conversion.
    raw_words = re.findall(r'\b\w+\b', corpus_text.strip().lower())

    word_to_byte_tokens_counts = collections.Counter()
    initial_byte_token_vocab = set()

    for word in raw_words:
        # Encode word to UTF-8 bytes
        word_bytes = word.encode('utf-8')
        # Represent the word as a tuple of single-byte tokens (tuples).
        # e.g., "hi" (104, 105) -> ((104,), (105,))
        byte_tokens_sequence = tuple((byte,) for byte in word_bytes)
        
        word_to_byte_tokens_counts[byte_tokens_sequence] += 1
        
        # Add individual single-byte tokens to the initial vocabulary
        for byte_token in byte_tokens_sequence:
            initial_byte_token_vocab.add(byte_token)

    print("--- Step 0: Preprocessing and Initial Byte Token Vocabulary ---")
    printable_counts = {
        " ".join( ".".join(map(str,tok)) for tok in k) : v 
        for k,v in word_to_byte_tokens_counts.items()
    }
    print(f"Word to byte-tokens sequence frequencies (first 5): {list(printable_counts.items())[:5]}")
    # Sort vocab for consistent printing. Vocab contains tuples like (104,).
    sorted_vocab_for_print = sorted(list(initial_byte_token_vocab), key=lambda x: x[0])
    print(f"Initial byte-token vocabulary (size {len(initial_byte_token_vocab)}): {sorted_vocab_for_print[:20]}...\n")
    
    return word_to_byte_tokens_counts, initial_byte_token_vocab

if __name__ == '__main__':
    # Basic test for the first function
    sample_corpus = "hello world hello你好" # \w+ makes "hello你好" one word
    # Expected words: ['hello', 'world', 'hello你好']

    word_token_counts, initial_vocab_tokens = get_initial_vocab_and_word_byte_counts(sample_corpus)
    
    print("\n--- Test Output for get_initial_vocab_and_word_byte_counts (tokenized structure) ---")
    print("Word to Byte Tokens Counts:")
    for token_seq, count in word_token_counts.items():
        # Reconstruct word string for readability if possible
        try:
            # Each token in token_seq is a tuple, e.g. (104,) or (104, 101)
            # Flatten list of tuples and then convert bytes to string
            byte_values = [b for token_tuple in token_seq for b in token_tuple]
            reconstructed_word = bytearray(byte_values).decode('utf-8', errors='ignore')
            print(f"  Sequence: {token_seq} (Word: '{reconstructed_word}') -> Count: {count}")
        except:
            print(f"  Sequence: {token_seq} -> Count: {count} (Error reconstructing word)")

    
    print(f"\nInitial Byte Token Vocab (sorted, first 20): {sorted(list(initial_vocab_tokens), key=lambda x: x[0])[:20]}")
    
    # Verify structure and content
    # For "hello": ( (104,), (101,), (108,), (108,), (111,) )
    hello_token_seq = tuple((b,) for b in "hello".encode('utf-8'))
    assert hello_token_seq in word_token_counts
    assert word_token_counts[hello_token_seq] == 1 # because "hello你好" is a different word.

    # For "world": ( (119,), (111,), (114,), (108,), (100,) )
    world_token_seq = tuple((b,) for b in "world".encode('utf-8'))
    assert world_token_seq in word_token_counts
    assert word_token_counts[world_token_seq] == 1

    # For "hello你好":
    hello_nihao_token_seq = tuple((b,) for b in "hello你好".encode('utf-8'))
    assert hello_nihao_token_seq in word_token_counts
    assert word_token_counts[hello_nihao_token_seq] == 1
    
    # Check initial vocab tokens
    assert (104,) in initial_vocab_tokens # h
    assert (228,) in initial_vocab_tokens # first byte of 你
    assert (101,) in initial_vocab_tokens # e

    # Test with corpus where "hello" appears twice as a standalone word
    sample_corpus_spaced = "hello world hello 你好"
    words_from_spaced = re.findall(r'\b\w+\b', sample_corpus_spaced.strip().lower())
    print(f"\nWords from spaced corpus: {words_from_spaced}") # Should be ['hello', 'world', 'hello', '你好']

    counts_spaced, vocab_spaced = get_initial_vocab_and_word_byte_counts(sample_corpus_spaced)
    print("Spaced Word Byte Token Counts:")
    for token_seq, count_val in counts_spaced.items():
        byte_values = [b for token_tuple in token_seq for b in token_tuple]
        reconstructed_word = bytearray(byte_values).decode('utf-8', errors='ignore')
        print(f"  {token_seq} (Word: '{reconstructed_word}'): {count_val}")
    
    assert counts_spaced[hello_token_seq] == 2

    nihao_bytes = "你好".encode('utf-8')
    nihao_token_seq = tuple((b,) for b in nihao_bytes)
    assert nihao_token_seq in counts_spaced
    assert counts_spaced[nihao_token_seq] == 1
    
    print(f"Initial byte token vocab for spaced (size {len(vocab_spaced)}): {sorted(list(vocab_spaced), key=lambda x:x[0])[:20]}")
    print("\nget_initial_vocab_and_word_byte_counts structural change seems OK.")
    print("Note: word_byte_counts keys are now tuples of single-byte tokens (tuples).")
    print("Initial vocabulary is a set of single-byte tokens (tuples like (byte_val,)).")


# Placeholder for next functions
def get_byte_pair_stats(word_byte_counts):
    """
    Counts the frequency of adjacent byte pairs in all word byte sequences.
    
    Args:
        word_byte_counts (collections.Counter): Frequencies of words, where words
            are represented as tuples of byte integers.
            e.g., {(104, 101, 108, 108, 111): 5, ...}
            Each element in the tuple is an individual byte (integer 0-255).

    Returns:
        collections.Counter: Frequencies of adjacent byte pairs.
                             Each key is a tuple of two integers (byte1, byte2).
                             e.g., {(104, 101): 7, (101, 108): 9, ...}
    """
    pair_stats = collections.defaultdict(int)
    for byte_sequence, freq in word_byte_counts.items():
        # byte_sequence is a tuple of integers, e.g., (104, 101, 108, 108, 111)
        # We are looking for pairs of individual bytes.
        for i in range(len(byte_sequence) - 1):
            # Each element byte_sequence[i] is an int (a byte)
            pair = (byte_sequence[i], byte_sequence[i+1])
            pair_stats[pair] += freq
    return pair_stats

def merge_byte_pair(target_byte_token_pair, word_to_byte_tokens_counts_in):
    """
    Merges a specified pair of byte tokens within all word byte-token sequences.

    Args:
        target_byte_token_pair (tuple): The pair of byte tokens to merge.
            Each token in the pair is a tuple of bytes, e.g., ((b1, b2), (b3,)).
        word_to_byte_tokens_counts_in (collections.Counter): Input frequencies of
            word representations (tuples of byte tokens).

    Returns:
        collections.Counter: New frequencies after merging the pair.
            Keys are the updated tuples of byte tokens.
    """
    word_to_byte_tokens_counts_out = collections.defaultdict(int)
    
    # Create the new merged token.
    # If target_byte_token_pair is ((104,), (101,)), merged_token is (104, 101).
    # If target_byte_token_pair is ((104, 101), (108,)), merged_token is (104, 101, 108).
    merged_token = target_byte_token_pair[0] + target_byte_token_pair[1]

    for current_byte_token_sequence, freq in word_to_byte_tokens_counts_in.items():
        new_sequence_tokens = []
        i = 0
        while i < len(current_byte_token_sequence):
            # Check if the current token and the next one form the target pair
            if i < len(current_byte_token_sequence) - 1 and \
               (current_byte_token_sequence[i], current_byte_token_sequence[i+1]) == target_byte_token_pair:
                new_sequence_tokens.append(merged_token)
                i += 2 # Skip the two merged tokens
            else:
                new_sequence_tokens.append(current_byte_token_sequence[i])
                i += 1
        word_to_byte_tokens_counts_out[tuple(new_sequence_tokens)] += freq
        
    return word_to_byte_tokens_counts_out

def train_bbpe(corpus_text, num_merges):
    """
    Trains the Byte-BPE model.

    Args:
        corpus_text (str): The raw text corpus.
        num_merges (int): The number of merge operations to perform.

    Returns:
        tuple: (
            final_byte_token_vocab (set): The final vocabulary of byte tokens.
                                          Each token is a tuple of bytes (integers).
                                          e.g., {(104,), (101,), (104, 101), ...}
            merge_rules (list): A list of merge rules applied.
                                Each rule is a tuple of two byte tokens that were merged.
                                e.g., [((104,), (101,)), ((101, 108), (108,)), ...]
        )
    """
    # 1. Initial Setup from corpus_text
    # word_to_byte_tokens_counts: keys are word representations (tuples of byte tokens), values are frequencies.
    # current_byte_token_vocab: set of unique byte tokens (initially single-byte tokens like (104,)).
    word_to_byte_tokens_counts, current_byte_token_vocab = get_initial_vocab_and_word_byte_counts(corpus_text)
    
    merge_rules = [] # Stores merge rules in order of learning.

    print("--- Starting BBPE Training Iterations ---\n")
    for i in range(num_merges):
        print(f"--- Merge Iteration {i + 1}/{num_merges} ---")

        # 2. Count frequencies of adjacent byte token pairs.
        # pair_stats: keys are tuples like ((tok1_bytes), (tok2_bytes)), values are frequencies.
        pair_stats = get_byte_pair_stats(word_to_byte_tokens_counts)

        if not pair_stats:
            print("No more byte token pairs to merge. Stopping training early.")
            break

        # 3. Find the most frequent byte token pair.
        best_byte_token_pair = max(pair_stats, key=pair_stats.get)
        best_pair_freq = pair_stats[best_byte_token_pair]
        
        # For printing: convert byte tuples in pair to readable strings if desired
        # pair_str = (best_byte_token_pair[0], best_byte_token_pair[1]) # actual tokens
        print(f"Most frequent pair: {best_byte_token_pair} (Frequency: {best_pair_freq})")

        # 4. Merge this pair to form a new byte token.
        # Add this new token to the vocabulary.
        # merged_token is already a tuple of bytes: token1_bytes + token2_bytes
        merged_byte_token = best_byte_token_pair[0] + best_byte_token_pair[1]
        current_byte_token_vocab.add(merged_byte_token)
        merge_rules.append(best_byte_token_pair) # Record this merge rule.

        # 5. Update the corpus representation by applying the merge.
        word_to_byte_tokens_counts = merge_byte_pair(best_byte_token_pair, word_to_byte_tokens_counts)
        print(f"After merge, new token '{merged_byte_token}' added to vocabulary.")
        # Optional: print details for debugging
        # print(f"Current vocabulary size: {len(current_byte_token_vocab)}\n")
        print("-" * 30)

    print("\n--- BBPE Training Complete ---")
    print(f"Final byte token vocabulary size: {len(current_byte_token_vocab)}")
    # Sort for printing part of the vocab
    # print_vocab = sorted(list(current_byte_token_vocab), key=lambda x: (len(x), x))
    # print(f"Final vocabulary (sample): {print_vocab[:20]} ...")
    print(f"Learned merge rules (Total: {len(merge_rules)}):")
    # for idx, rule in enumerate(merge_rules):
    #     rule_str = (rule[0], rule[1]) # actual tokens
    #     merged_form = rule[0] + rule[1]
    #     print(f"  {idx+1}. {rule_str} -> {merged_form}")

    return current_byte_token_vocab, merge_rules

def tokenize_word_with_bbpe(word_string, merge_rules):
    """
    Tokenizes a single word string using the learned BBPE merge rules.

    Args:
        word_string (str): The word to tokenize.
        merge_rules (list): A list of merge rules (pairs of byte tokens)
                             in the order they were learned.
                             e.g., [ ((b1,), (b2,)), ((b1,b2), (b3,)), ... ]

    Returns:
        list: A list of byte tokens (tuples of bytes) representing the tokenized word.
              e.g., [(108, 111, 119), (101, 115, 116)] for "lowest"
    """
    if not word_string:
        return []

    # 1. Encode word to UTF-8 bytes and convert to initial sequence of single-byte tokens
    word_bytes = word_string.encode('utf-8')
    # e.g., "low" (108,111,119) -> [(108,), (111,), (119,)]
    current_tokens = [(byte,) for byte in word_bytes]

    if not current_tokens: # Handles empty string case after encoding
        return []

    # 2. Iteratively apply merge rules
    for token_pair_to_merge in merge_rules:
        # token_pair_to_merge is like ((b1,), (b2,)) or ((b1,b2), (b3,))
        new_tokens = []
        i = 0
        while i < len(current_tokens):
            if i < len(current_tokens) - 1 and \
               (current_tokens[i], current_tokens[i+1]) == token_pair_to_merge:
                # Merge the pair
                merged_token = token_pair_to_merge[0] + token_pair_to_merge[1]
                new_tokens.append(merged_token)
                i += 2 # Skip two processed tokens
            else:
                new_tokens.append(current_tokens[i])
                i += 1
        current_tokens = new_tokens
        # print(f"Applied rule {token_pair_to_merge} -> {merged_token if 'merged_token' in locals() else ''}, Tokens: {current_tokens}")
    
    return current_tokens

# Helper to convert token (tuple of ints) to string "b1 b2 b3"
def _token_to_str(token_tuple):
    return " ".join(map(str, token_tuple))

# Helper to convert string "b1 b2 b3" to token (tuple of ints)
def _str_to_token(token_str):
    return tuple(map(int, token_str.split()))

def save_bbpe_model(byte_token_vocab, merge_rules, base_filename):
    """
    Saves the BBPE model (vocabulary and merge rules) to files.

    Args:
        byte_token_vocab (set): The vocabulary of byte tokens (tuples of ints).
        merge_rules (list): A list of merge rules (pairs of byte tokens).
        base_filename (str): The base name for the output files.
                             ".bvocab" and ".bmerges" will be appended.
    """
    vocab_file = base_filename + ".bvocab"
    merges_file = base_filename + ".bmerges"

    # Save vocabulary
    # Sort for consistency: by length, then by content
    sorted_vocab = sorted(list(byte_token_vocab), key=lambda x: (len(x), x))
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in sorted_vocab:
            f.write(_token_to_str(token) + '\n')
    print(f"Byte token vocabulary saved to: {vocab_file}")

    # Save merge rules
    # Each rule is ((b1,b2), (b3,)) stored as "b1 b2 --- b3"
    with open(merges_file, 'w', encoding='utf-8') as f:
        for rule_token_pair in merge_rules:
            # rule_token_pair is (token1, token2)
            f.write(f"{_token_to_str(rule_token_pair[0])} --- {_token_to_str(rule_token_pair[1])}\n")
    print(f"Merge rules saved to: {merges_file}")


def load_bbpe_model(base_filename):
    """
    Loads a BBPE model (vocabulary and merge rules) from files.

    Args:
        base_filename (str): The base name of the model files.
                             Assumes ".bvocab" and ".bmerges" extensions.

    Returns:
        tuple: (
            byte_token_vocab (set): The loaded byte token vocabulary.
            merge_rules (list): The loaded list of merge rules.
        )
        Returns None, None if files are not found or parsing fails.
    """
    vocab_file = base_filename + ".bvocab"
    merges_file = base_filename + ".bmerges"
    
    loaded_vocab = set()
    loaded_merge_rules = []

    try:
        # Load vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded_vocab.add(_str_to_token(line))
        print(f"Byte token vocabulary loaded from: {vocab_file}")

        # Load merge rules
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' --- ')
                if len(parts) == 2:
                    try:
                        token1 = _str_to_token(parts[0])
                        token2 = _str_to_token(parts[1])
                        loaded_merge_rules.append((token1, token2))
                    except ValueError as e:
                        print(f"Warning: Malformed rule on line {line_num} in {merges_file}: '{line}'. Error: {e}")
                else:
                    print(f"Warning: Malformed rule structure on line {line_num} in {merges_file}: '{line}'")
        print(f"Merge rules loaded from: {merges_file}")
        
        return loaded_vocab, loaded_merge_rules

    except FileNotFoundError:
        print(f"Error: Model files not found for base_filename: {base_filename}")
        return None, None
    except Exception as e:
        print(f"Error loading BBPE model: {e}")
        return None, None

if __name__ == '__main__':
    # --- Test get_initial_vocab_and_word_byte_counts ---
    print("--- Running initial tests for get_initial_vocab_and_word_byte_counts ---")
    sample_corpus_initial_test = "hello world hello你好"
    word_token_counts, initial_vocab_tokens = get_initial_vocab_and_word_byte_counts(sample_corpus_initial_test)
    print("\n--- Test Output for get_initial_vocab_and_word_byte_counts (tokenized structure) ---")
    # ... (previous test printouts for this function can be kept or summarized)
    hello_token_seq = tuple((b,) for b in "hello".encode('utf-8'))
    assert word_token_counts[hello_token_seq] == 1
    print("Initial function tests passed.\n")

    # --- Full BBPE Training, Save, Load, and Tokenization Demo ---
    print("--- Starting Full BBPE Demo ---")
    corpus = """
    hello world
    hello你好
    你好世界
    Byte Pair Encoding
    Byte Pair Example
    こんにちは世界
    """
    num_merges = 50 # Adjust as needed for demonstration

    # 1. Train BBPE model
    print("--- Training Original BBPE Model ---")
    original_byte_vocab, original_merge_rules = train_bbpe(corpus, num_merges)

    # Print some details of the trained model
    print("\n--- Original Trained Model Details ---")
    # print(f"Original Vocab (sample): {list(original_byte_vocab)[:20]}")
    # print(f"Original Merge Rules (sample): {original_merge_rules[:5]}")


    # 2. Save the trained model
    model_basename = "bbpe_test_model"
    print(f"\n--- Saving BBPE Model to '{model_basename}.*' ---")
    save_bbpe_model(original_byte_vocab, original_merge_rules, model_basename)

    # 3. Load the model
    print(f"\n--- Loading BBPE Model from '{model_basename}.*' ---")
    loaded_byte_vocab, loaded_merge_rules = load_bbpe_model(model_basename)

    if loaded_byte_vocab is not None and loaded_merge_rules is not None:
        print("\n--- Verifying Loaded Model ---")
        # Compare vocabularies (sets, order doesn't matter for content)
        if original_byte_vocab == loaded_byte_vocab:
            print("Vocabulary successfully saved and loaded: Identical.")
        else:
            print("Vocabulary mismatch after loading!")
            if len(original_byte_vocab) != len(loaded_byte_vocab):
                 print(f"Length diff: Orig {len(original_byte_vocab)}, Loaded {len(loaded_byte_vocab)}")
            # print(f"In original but not loaded: {original_byte_vocab - loaded_byte_vocab}")
            # print(f"In loaded but not original: {loaded_byte_vocab - original_byte_vocab}")


        # Compare merge rules (lists, order matters)
        if original_merge_rules == loaded_merge_rules:
            print("Merge rules successfully saved and loaded: Identical.")
        else:
            print("Merge rules mismatch after loading!")
            # for i, (orig_r, load_r) in enumerate(zip(original_merge_rules, loaded_merge_rules)):
            #     if orig_r != load_r:
            #         print(f"Rule mismatch at index {i}: Orig: {orig_r}, Loaded: {load_r}")
            #         break
            # if len(original_merge_rules) != len(loaded_merge_rules):
            #     print(f"Rules length diff: Orig {len(original_merge_rules)}, Loaded {len(loaded_merge_rules)}")


        print("\n--- Tokenizing with Loaded Model ---")
        test_words = ["hello", "你好", "world", "BytePair", "こんにちは", "newoovword"]
        
        for word in test_words:
            tokenized_output = tokenize_word_with_bbpe(word, loaded_merge_rules)
            
            # Make tokenized output more readable: convert byte tuples to strings
            readable_tokens = []
            for token_tuple in tokenized_output:
                try:
                    # Attempt to decode each token (tuple of bytes) as UTF-8
                    # This is for display only; some tokens might not be valid UTF-8 on their own
                    readable_tokens.append(f"'{bytearray(token_tuple).decode('utf-8', errors='replace')}'{token_tuple}")
                except:
                    readable_tokens.append(str(token_tuple)) # Fallback to string of tuple
            
            print(f"Word '{word}' -> Tokenized (Loaded Model): {readable_tokens}")

        # Verification of tokenization against original model (optional, but good for ensuring consistency)
        print("\n--- Comparing Tokenization with Original Model vs Loaded Model ---")
        all_matched = True
        for word in test_words:
            tokens_original = tokenize_word_with_bbpe(word, original_merge_rules)
            tokens_loaded = tokenize_word_with_bbpe(word, loaded_merge_rules)
            if tokens_original != tokens_loaded:
                all_matched = False
                print(f"Mismatch for '{word}':")
                print(f"  Original: {tokens_original}")
                print(f"  Loaded:   {tokens_loaded}")
        if all_matched:
            print("Tokenization results are IDENTICAL for original and loaded models.")
        else:
            print("Tokenization results DIFFER for original and loaded models.")
            
    else:
        print("Failed to load the BBPE model. Skipping verification and tokenization demo.")

    print("\n--- BBPE Demo Complete ---")
