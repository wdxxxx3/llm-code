import collections
import re
import math
import heapq # For priority queue if needed for Viterbi or pruning, not strictly for get_initial_candidates

# Define special tokens if any (ULM typically doesn't use ## or </w> in the same way as BPE/WordPiece)
# However, an UNK token might be useful.
UNK_TOKEN = "[UNK]"

def get_initial_candidates(corpus_text, max_subword_len=10, min_freq=2):
    """
    Generates an initial pool of candidate subwords and their frequencies from the corpus.
    Candidates include all unique characters and all substrings up to max_subword_len.

    Args:
        corpus_text (str): The raw text corpus.
        max_subword_len (int): Maximum length for extracted substrings.
        min_freq (int): Minimum frequency for a candidate to be kept.

    Returns:
        collections.Counter: Frequencies of candidate subwords (strings).
    """
    print("--- Generating Initial Candidates ---")
    # Normalize text: lowercase and split into words
    # Using \w+ to capture sequences of alphanumeric characters as words.
    # This means punctuation might be ignored unless handled separately.
    words = re.findall(r'\b\w+\b', corpus_text.lower().strip())

    candidate_freqs = collections.Counter()

    print(f"Processing {len(words)} words to extract candidates...")
    for i, word in enumerate(words):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(words)} words...")

        # Add all single characters from the word
        # This ensures all individual characters from actual words are candidates
        # and their frequencies are based on their occurrences within words.
        for char_idx in range(len(word)):
            candidate_freqs[word[char_idx]] += 1
            
        # Add all substrings up to max_subword_len
        for j in range(len(word)):
            for k in range(j + 1, min(j + 1 + max_subword_len, len(word) + 1)):
                # Substring length is k - j. We want length >= 2 here since single chars are covered.
                # Or, allow length 1 substrings here too, Counter will sum them up.
                # Let's ensure single characters are only added once per occurrence logic above.
                # So, substrings here should be length > 1 if chars are already counted.
                # However, the prompt implies "all substrings", so let's include length 1 here
                # and the Counter will naturally sum them.
                # If a char 'a' appears 5 times, its count will be 5.
                # If we add it via char loop and then again via substring loop, it will be 10.
                # Let's adjust: count single chars first, then substrings of length > 1.
                # The current loop for single chars does: candidate_freqs[word[char_idx]] += 1 for each char.
                # The substring loop:
                # if k - j > 1: # Only consider substrings of length 2 or more if single chars are separate
                substring = word[j:k]
                candidate_freqs[substring] += 1
                
    print(f"Extracted {len(candidate_freqs)} unique candidate subwords (pre-filtering).")

    # Filter candidates by minimum frequency
    filtered_candidates = collections.Counter(
        {subword: freq for subword, freq in candidate_freqs.items() if freq >= min_freq}
    )
    # Also ensure all single characters from the original alphabet are present, even if rare.
    # (The problem implies this by "all unique characters from the training corpus")
    # The min_freq might remove some. Let's add back single characters that were in words.
    initial_chars = set(char for word in words for char in word)
    for char in initial_chars:
        if char not in filtered_candidates and char in candidate_freqs: # It was seen, but filtered by min_freq
             filtered_candidates[char] = candidate_freqs[char] # Add it back with original freq
        elif char not in filtered_candidates: # Should not happen if words is not empty
             filtered_candidates[char] = 1 # Add with minimal count if somehow missed

    print(f"Filtered to {len(filtered_candidates)} candidates with min_freq={min_freq} (single chars preserved).")
    # print(f"Sample candidates (top 10): {filtered_candidates.most_common(10)}")
    return filtered_candidates

if __name__ == '__main__':
    sample_corpus = "this is a simple corpus with simple words for testing this simple algorithm."
    sample_corpus += " hello world 你好 世界 how are you" # Add multi-byte
    
    max_len = 5
    min_f = 1 # Keep all for this small sample

    print(f"--- Testing get_initial_candidates (max_len={max_len}, min_freq={min_f}) ---")
    candidates = get_initial_candidates(sample_corpus, max_subword_len=max_len, min_freq=min_f)

    print("\n--- Initial Candidates (Sample) ---")
    # Print some candidates and their frequencies
    for subword, freq in candidates.most_common(10):
        print(f"'{subword}': {freq}")

    # Verify some expected candidates
    assert 's' in candidates
    assert 'simple' in candidates
    assert candidates['simple'] >= 3 # "simple" appears 3 times
    assert candidates['s'] >= 6 # s in simple (3), is (2), this (2), testing (1) = 8

    assert 'is' in candidates
    assert candidates['is'] >= 2 # "this", "is"

    assert '你好' in candidates # Full multi-byte word if <= max_len
    assert candidates['你好'] == 1
    assert '你' in candidates
    assert candidates['你'] == 1
    assert '好' in candidates
    assert candidates['好'] == 1
    
    # Substring "impl" from "simple"
    if max_len >= 4:
        assert 'impl' in candidates
        assert candidates['impl'] >= 3

    # Substring "o w" is not a candidate because words are split by \b\w+\b
    assert 'o w' not in candidates # Space is not part of \w+

    print("\nget_initial_candidates function implemented and basic tests passed.")
    print(f"Total unique candidates found: {len(candidates)}")


# Placeholders for next functions
def viterbi_segment(text_chars_list, vocab_probs, max_subword_len_in_vocab=None):
    """
    Segments a list of characters into subwords using the Viterbi algorithm.

    Args:
        text_chars_list (list): A list of characters representing the word to segment.
        vocab_probs (dict): A dictionary mapping subword strings to their log probabilities.
                            e.g., {'a': -1.0, 'b': -1.2, 'ab': -0.5}
        max_subword_len_in_vocab (int, optional): The maximum length of a subword present
                                                  in vocab_probs. If None, it will be
                                                  calculated, or a practical limit assumed.

    Returns:
        tuple: (
            segmented_subwords (list): List of subword strings.
            total_log_prob (float): Total log probability of the best segmentation.
        )
        Returns ([UNK_TOKEN], -infinity) or similar if segmentation is not possible.
    """
    n = len(text_chars_list)
    if n == 0:
        return [], 0.0

    # dp_scores[i] stores the maximum log-probability for segmenting text_chars_list[:i]
    dp_scores = [-math.inf] * (n + 1)
    # dp_pointers[i] stores the length of the last subword in the optimal segmentation of text_chars_list[:i]
    dp_pointers = [0] * (n + 1)
    dp_scores[0] = 0.0  # Base case: empty string has log_prob 0

    # Determine max_subword_len to check if not provided
    # This is an optimization; without it, the inner loop for j can go up to i.
    _max_len_check = n 
    if max_subword_len_in_vocab:
        _max_len_check = max_subword_len_in_vocab
    # else: # Optionally calculate from vocab_probs if not too large
    #     if vocab_probs: _max_len_check = max(len(k) for k in vocab_probs.keys())

    for i in range(1, n + 1):  # Current end position (exclusive) of the prefix
        # j is the length of the potential last subword
        # Iterate j from 1 up to min(i, _max_len_check)
        for j in range(1, min(i, _max_len_check) + 1):
            start_pos = i - j
            subword_str = "".join(text_chars_list[start_pos:i])

            if subword_str in vocab_probs:
                log_prob_subword = vocab_probs[subword_str]
                current_path_score = dp_scores[start_pos] + log_prob_subword
                
                if current_path_score > dp_scores[i]:
                    dp_scores[i] = current_path_score
                    dp_pointers[i] = j  # Store length of this subword

    # Backtrack to reconstruct the best path
    segmented_subwords = []
    if dp_scores[n] == -math.inf:
        # Cannot segment the word with the given vocabulary.
        # Fallback: treat the whole word as UNK or segment into known single chars if possible.
        # For now, let's return a generic UNK if UNK_TOKEN is in vocab, else the original word.
        # A more sophisticated fallback would try to segment using only single characters from vocab.
        if UNK_TOKEN in vocab_probs:
             return [UNK_TOKEN], vocab_probs.get(UNK_TOKEN, -math.inf) # Use UNK_TOKEN's prob
        else: # No UNK token, return original word as unsegmentable (or its chars if they are in vocab)
             # This case needs careful handling based on desired ULM behavior for OOV.
             # A simple strategy: if unsegmentable, return the original word as one token.
             return ["".join(text_chars_list)], -math.inf


    current_pos = n
    while current_pos > 0:
        subword_len = dp_pointers[current_pos]
        if subword_len == 0: # Should not happen if dp_scores[n] was not -inf
            print(f"Error in Viterbi backtracking: zero length subword at pos {current_pos} for word {''.join(text_chars_list)}")
            # Fallback to UNK if this state is reached.
            return [UNK_TOKEN if UNK_TOKEN in vocab_probs else "".join(text_chars_list)], -math.inf
            
        subword = "".join(text_chars_list[current_pos - subword_len : current_pos])
        segmented_subwords.insert(0, subword) # Prepend to maintain order
        current_pos -= subword_len
        
    return segmented_subwords, dp_scores[n]


def train_ulm(corpus_text, target_vocab_size, num_iterations=5, 
              initial_seed_vocab_size_factor=5, # Smaller factor for faster tests
              max_subword_len_candidates=8, 
              min_freq_candidates=2,
              pruning_percentage=0.2): # Percentage of non-char pieces to remove each relevant iteration
    """
    Trains the Unigram Language Model tokenizer.

    Args:
        corpus_text (str): The raw text corpus.
        target_vocab_size (int): The desired final vocabulary size.
        num_iterations (int): Number of EM-like iterations.
        initial_seed_vocab_size_factor (int): Factor to determine initial seed vocab size.
        max_subword_len_candidates (int): Max length for initial candidate substrings.
        min_freq_candidates (int): Min frequency for initial candidate substrings.
        pruning_percentage (float): Percentage of lowest-probability non-character 
                                    subwords to consider removing in pruning steps.
    Returns:
        dict: The final vocabulary mapping subwords to their log probabilities (scores).
    """
    print("--- Starting ULM Training ---")

    # 1. Initialization
    print("Step 1: Generating initial candidates...")
    initial_candidates_freqs = get_initial_candidates(
        corpus_text, max_subword_len_candidates, min_freq_candidates
    )

    # Extract all single characters
    words_list = re.findall(r'\b\w+\b', corpus_text.lower().strip())
    all_single_chars = set(char for word in words_list for char in word)
    all_single_chars.add(UNK_TOKEN) # Ensure UNK is treated as a char for preservation

    # Create seed vocabulary: all single chars + most frequent N other candidates
    seed_vocab_target_size = min(
        len(initial_candidates_freqs),
        target_vocab_size * initial_seed_vocab_size_factor 
    )
    
    current_vocab_probs = {}
    # Add all single characters first
    total_freq_sum_for_init_probs = 0
    temp_probs_for_init = {}

    for char_token in all_single_chars:
        freq = initial_candidates_freqs.get(char_token, 1) # Default freq 1 if somehow missed
        temp_probs_for_init[char_token] = freq
        total_freq_sum_for_init_probs += freq
        
    # Add other most frequent candidates
    # Sort by frequency, descending.
    sorted_candidates = initial_candidates_freqs.most_common()
    
    for subword, freq in sorted_candidates:
        if len(current_vocab_probs) >= seed_vocab_target_size:
            break
        if subword not in current_vocab_probs: # Add if not already (e.g. not a single char)
            temp_probs_for_init[subword] = freq
            total_freq_sum_for_init_probs += freq
            
    # Normalize initial probabilities (log probs)
    if total_freq_sum_for_init_probs == 0: total_freq_sum_for_init_probs = 1 # Avoid div by zero
    
    for subword, freq in temp_probs_for_init.items():
        # Using log(frequency) as score, as true probs P(s) are tricky without full LM objective
        # A common simplification is that score(s) is related to its frequency or count.
        # Let's use log of normalized frequency for now.
        # Ensure extremely low prob for UNK initially if using normalized freqs.
        if subword == UNK_TOKEN:
             current_vocab_probs[subword] = -20.0 # Very low log_prob for UNK
        else:
             current_vocab_probs[subword] = math.log(freq / total_freq_sum_for_init_probs)

    print(f"Initial seed vocabulary size: {len(current_vocab_probs)}")
    max_len_in_vocab = max(len(sw) for sw in current_vocab_probs) if current_vocab_probs else 0


    # 2. Iterative EM-like Pruning
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
        
        # E-step: Segment corpus and update subword counts
        print("  E-step: Segmenting corpus and updating subword counts...")
        subword_counts_this_iter = collections.Counter()
        total_words = len(words_list)
        for i, word_str in enumerate(words_list):
            if (i+1) % 2000 == 0: print(f"    Segmented {i+1}/{total_words} words...")
            char_list = list(word_str)
            if not char_list: continue
            
            segmented_subwords, _ = viterbi_segment(char_list, current_vocab_probs, max_len_in_vocab)
            for subword in segmented_subwords:
                subword_counts_this_iter[subword] += 1
        
        # M-step: Update probabilities (scores) based on new counts
        # Here, we use log(count) as a score. This is a simplification.
        # A true ULM uses probabilities that sum to 1 over the vocabulary,
        # and the EM algorithm maximizes likelihood.
        # For this simplified version, we'll update scores for items in vocab based on counts.
        print("  M-step: Updating scores (log probabilities)...")
        
        # Keep only subwords that were actually used in this iteration's segmentations
        # plus all single characters (to ensure they are not lost)
        next_vocab_probs = {}
        total_counts_for_probs_this_iter = sum(subword_counts_this_iter.values())
        if total_counts_for_probs_this_iter == 0: total_counts_for_probs_this_iter = 1

        for subword in list(current_vocab_probs.keys()): # Iterate over existing vocab keys
            if subword in all_single_chars: # Always keep single characters
                count = subword_counts_this_iter.get(subword, 1) # Min count 1 for chars
                if subword == UNK_TOKEN:
                    next_vocab_probs[subword] = current_vocab_probs.get(UNK_TOKEN, -20.0) # Keep UNK prob stable or very low
                else:
                    next_vocab_probs[subword] = math.log(count / total_counts_for_probs_this_iter)

            elif subword in subword_counts_this_iter: # Was seen in segmentation
                 count = subword_counts_this_iter[subword]
                 next_vocab_probs[subword] = math.log(count / total_counts_for_probs_this_iter)
            # Else: subword from previous vocab was not used, so it's implicitly dropped unless a single char.

        current_vocab_probs = next_vocab_probs
        
        # Pruning step: Reduce vocab to target_vocab_size
        if len(current_vocab_probs) > target_vocab_size:
            print(f"  Pruning: Current vocab size {len(current_vocab_probs)} > target {target_vocab_size}")
            
            # Separate single characters (always keep) from multi-character subwords
            single_char_tokens_in_vocab = {sw:p for sw,p in current_vocab_probs.items() if sw in all_single_chars}
            multi_char_subwords = {sw:p for sw,p in current_vocab_probs.items() if sw not in all_single_chars}

            # How many multi-char subwords we want to keep:
            num_multi_char_to_keep = target_vocab_size - len(single_char_tokens_in_vocab)
            if num_multi_char_to_keep < 0: num_multi_char_to_keep = 0 # Should not happen if target_vocab_size is reasonable

            if len(multi_char_subwords) > num_multi_char_to_keep:
                # Sort multi-char subwords by their probability (score) in descending order
                sorted_multi_char_subwords = sorted(multi_char_subwords.items(), key=lambda item: item[1], reverse=True)
                
                # Keep the top `num_multi_char_to_keep`
                kept_multi_char_subwords = dict(sorted_multi_char_subwords[:num_multi_char_to_keep])
                
                # Combine kept single characters and top multi-character subwords
                current_vocab_probs = {**single_char_tokens_in_vocab, **kept_multi_char_subwords}
            
            print(f"    New vocab size after pruning: {len(current_vocab_probs)}")

        max_len_in_vocab = max(len(sw) for sw in current_vocab_probs) if current_vocab_probs else 0
        # print(f"Sample vocab after iter {iteration+1} (top 5 by prob): {sorted(current_vocab_probs.items(), key=lambda x: x[1], reverse=True)[:5]}")


    print("\n--- ULM Training Complete ---")
    print(f"Final vocabulary size: {len(current_vocab_probs)}")
    return current_vocab_probs

def tokenize_word_with_ulm(word_string, vocab_probs, max_subword_len_in_vocab=None):
    """
    Tokenizes a single word string using the trained ULM vocabulary and Viterbi segmentation.

    Args:
        word_string (str): The word to tokenize.
        vocab_probs (dict): A dictionary mapping subwords to their log probabilities.
        max_subword_len_in_vocab (int, optional): Max length of subwords in vocab.
                                                  Calculated if not provided.

    Returns:
        list: A list of subword strings.
    """
    if not word_string:
        return []
    
    char_list = list(word_string) # Convert word to list of characters

    if not max_subword_len_in_vocab and vocab_probs:
        # Calculate if not provided, can be slow if vocab is huge and called often
        max_subword_len_in_vocab = max(len(sw) for sw in vocab_probs.keys()) if vocab_probs else 0
        if max_subword_len_in_vocab == 0 and char_list: # Vocab empty or only empty string
             if UNK_TOKEN in vocab_probs: return [UNK_TOKEN]
             return [word_string]


    segmented_subwords, log_prob = viterbi_segment(
        char_list, 
        vocab_probs,
        max_subword_len_in_vocab
    )
    
    # print(f"Tokenizing '{word_string}': {segmented_subwords} (log_prob: {log_prob:.4f})")
    return segmented_subwords

def save_ulm_model(vocab_probs, base_filename):
    """
    Saves the ULM model (vocabulary and log probabilities) to a file.

    Args:
        vocab_probs (dict): A dictionary mapping subwords to their log probabilities.
        base_filename (str): The base name for the output file.
                             ".ulm_model" will be appended.
    """
    model_file = base_filename + ".ulm_model"
    
    # Sort items for consistent output
    sorted_vocab_items = sorted(vocab_probs.items())
    
    with open(model_file, 'w', encoding='utf-8') as f:
        for subword, log_prob in sorted_vocab_items:
            f.write(f"{subword}\t{log_prob}\n")
    print(f"ULM model saved to: {model_file}")


def load_ulm_model(base_filename):
    """
    Loads a ULM model (vocabulary and log probabilities) from a file.

    Args:
        base_filename (str): The base name of the model file.
                             Assumes ".ulm_model" extension.

    Returns:
        dict: A dictionary mapping subwords to their log probabilities.
              Returns None if the file is not found or parsing fails.
    """
    model_file = base_filename + ".ulm_model"
    loaded_vocab_probs = {}
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    subword = parts[0]
                    try:
                        log_prob = float(parts[1])
                        loaded_vocab_probs[subword] = log_prob
                    except ValueError:
                        print(f"Warning: Invalid log_prob on line {line_num} in {model_file}: '{parts[1]}'")
                else:
                    print(f"Warning: Malformed line {line_num} in {model_file}: '{line}'")
        print(f"ULM model loaded from: {model_file}")
        return loaded_vocab_probs
    except FileNotFoundError:
        print(f"Error: Model file not found: {model_file}")
        return None
    except Exception as e:
        print(f"Error loading ULM model: {e}")
        return None

if __name__ == '__main__':
    print("--- ULM (Unigram Language Model) Tokenizer Demo ---")

    # 1. Define Corpus
    corpus = """
    this is a simple corpus.
    it contains simple words for testing.
    this simple algorithm needs to be tested.
    hello world. 你好 世界. how are you?
    the quick brown fox jumps over the lazy dog.
    여러가지 다양한 언어의 단어들.
    """
    target_vocab_size = 100  # Small for demo; typical: 8k-32k
    num_train_iterations = 5 # Fewer iterations for faster demo
    
    # 2. Train ULM model
    print(f"\n--- Training ULM Model (Target Vocab Size: {target_vocab_size}, Iterations: {num_train_iterations}) ---")
    trained_vocab_probs = train_ulm(
        corpus, 
        target_vocab_size, 
        num_iterations=num_train_iterations,
        initial_seed_vocab_size_factor=3, # Smaller for faster init
        max_subword_len_candidates=6,   # Shorter max candidate length for speed
        min_freq_candidates=1           # Lower min freq for small corpus
    )

    print("\n--- Trained Model Details ---")
    print(f"Final Vocabulary size: {len(trained_vocab_probs)}")
    # Print some high-probability tokens
    # sorted_by_prob = sorted(trained_vocab_probs.items(), key=lambda item: item[1], reverse=True)
    # print(f"Sample of final vocabulary (top 10 by log_prob): {sorted_by_prob[:10]}")

    # 3. Save the trained model
    model_basename = "ulm_test_model"
    print(f"\n--- Saving ULM Model to '{model_basename}.ulm_model' ---")
    save_ulm_model(trained_vocab_probs, model_basename)

    # 4. Load the model
    print(f"\n--- Loading ULM Model from '{model_basename}.ulm_model' ---")
    loaded_vocab_probs = load_ulm_model(model_basename)

    if loaded_vocab_probs:
        print(f"Successfully loaded ULM model with vocabulary size: {len(loaded_vocab_probs)}")
        
        # Pre-calculate max subword length in loaded vocab for tokenizer efficiency
        max_len_loaded_vocab = 0
        if loaded_vocab_probs:
            max_len_loaded_vocab = max(len(sw) for sw in loaded_vocab_probs.keys())
        print(f"Max subword length in loaded vocab: {max_len_loaded_vocab}")


        # 5. Tokenize sample words
        print("\n--- Tokenizing Sample Words with Loaded ULM Model ---")
        test_words = [
            "simple", "corpus", "testing", "你好世界", "algorithm", 
            "unknownword", "fox", "단어들", "여러가지", "dog." # Test with punctuation if it was part of vocab
        ]
        
        for word in test_words:
            # Normalize word for tokenization (e.g. lowercase) if training was on lowercase
            # Our get_initial_candidates and train_ulm use .lower()
            word_to_tokenize = word.lower() 
            
            tokens = tokenize_word_with_ulm(word_to_tokenize, loaded_vocab_probs, max_len_loaded_vocab)
            # Reconstruct for printing if needed, or show subwords directly
            reconstructed = "".join(tokens)
            status = "(Perfect reconstruction)" if reconstructed == word_to_tokenize else f"(Reconstruction: {reconstructed})"
            if UNK_TOKEN in tokens: status = f"(Contains {UNK_TOKEN})"
                
            print(f"Word '{word}' (as '{word_to_tokenize}') -> Tokens: {tokens} {status}")
            
        # Test with a word containing characters potentially not in vocab
        oov_word = "xyz123" 
        print(f"\nTokenizing OOV word '{oov_word.lower()}'")
        tokens_oov = tokenize_word_with_ulm(oov_word.lower(), loaded_vocab_probs, max_len_loaded_vocab)
        reconstructed_oov = "".join(tokens_oov)
        status_oov = "(Perfect reconstruction)" if reconstructed_oov == oov_word.lower() else f"(Reconstruction: {reconstructed_oov})"
        if UNK_TOKEN in tokens_oov: status_oov = f"(Contains {UNK_TOKEN})"
        print(f"Word '{oov_word}' (as '{oov_word.lower()}') -> Tokens: {tokens_oov} {status_oov}")


    else:
        print("Failed to load ULM model. Skipping tokenization demo.")

    print("\n--- ULM Demo Complete ---")
