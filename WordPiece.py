import collections
import re

# Define special tokens
UNK_TOKEN = "[UNK]"
WORD_SUFFIX = "</w>" # Marks the end of a word

def get_initial_vocab_and_word_counts(corpus_text, initial_alphabet_chars):
    """
    1. Splits raw text into words.
    2. Represents words as lists of characters, adding WORD_SUFFIX.
    3. Counts frequency of these character-list representations.

    Args:
        corpus_text (str): The raw text corpus.
        initial_alphabet_chars (set): Set of unique characters forming the initial alphabet.
                                      (Note: WORD_SUFFIX is implicitly part of alphabet for processing)

    Returns:
        tuple: (
            word_counts (collections.Counter): Frequencies of words, where words
                are represented as tuples of characters (e.g., ('h','e','l','l','o','</w>')).
        )
    """
    # Simple word splitting: lowercase, split by whitespace.
    # More sophisticated splitting (e.g. by regex like in BPE/BBPE) can be used
    # but let's start with a basic approach.
    # We also need to handle punctuation if it's not part of initial_alphabet_chars.
    # For now, assume words are sequences of chars from initial_alphabet_chars, or split by spaces.
    
    # Let's use a regex that captures sequences of letters and numbers, similar to BPE.
    # This also helps in implicitly defining what a "word" is before character splitting.
    raw_words = re.findall(r'\b\w+\b', corpus_text.lower().strip())

    word_counts = collections.Counter()
    
    processed_words_details = [] # For debugging or deeper inspection if needed

    for word_str in raw_words:
        # Convert word to list of characters.
        # Only include characters that are in the provided initial_alphabet_chars.
        # Any character not in initial_alphabet_chars will be skipped or could be mapped to UNK_TOKEN here
        # if we wanted to handle UNK at this stage.
        # However, WordPiece typically builds its vocab from an initial alphabet,
        # and tokenization handles true OOV characters later.
        chars = [char for char in word_str if char in initial_alphabet_chars]
        
        if not chars: # Word might consist only of chars not in the alphabet
            processed_words_details.append({'original': word_str, 'processed': [], 'status': 'empty_after_filtering'})
            continue
            
        # Append WORD_SUFFIX
        word_char_list = chars + [WORD_SUFFIX]
        word_counts[tuple(word_char_list)] += 1
        processed_words_details.append({'original': word_str, 'processed': tuple(word_char_list), 'status': 'ok'})

    # print(f"Processed word details (sample): {processed_words_details[:5]}") # For diagnostics
    print(f"--- Initial Word Counts (get_initial_vocab_and_word_counts) ---")
    print(f"Total unique word forms (after char filtering & suffix): {len(word_counts)}")
    # print(f"Sample word counts: {list(word_counts.items())[:5]}")
    
    return word_counts

if __name__ == '__main__':
    sample_corpus = "hello world hello a b c ab ac bc abc testing"
    # Define an initial alphabet (typically all unique chars in corpus + special tokens)
    unique_chars = set(char for word in sample_corpus.split() for char in word)
    initial_alphabet = unique_chars # WORD_SUFFIX is handled implicitly by its addition
                                    # UNK_TOKEN will be added to vocab later in train_wordpiece

    print(f"Initial character alphabet from corpus: {sorted(list(initial_alphabet))}")

    word_counts_data = get_initial_vocab_and_word_counts(sample_corpus, initial_alphabet)

    print("\n--- Test Output for get_initial_vocab_and_word_counts ---")
    print("Word Counts (character list representation):")
    for char_list_tuple, count in word_counts_data.items():
        print(f"  {char_list_tuple} : {count}")

    # Example: ('h', 'e', 'l', 'l', 'o', '</w>') should have count 1 (if "hello" appears once and "hello" is a word)
    # Actually, "hello" appears twice.
    # The regex \b\w+\b splits "hello world hello" into "hello", "world", "hello".
    
    expected_hello_tuple = ('h', 'e', 'l', 'l', 'o', WORD_SUFFIX)
    if expected_hello_tuple in word_counts_data:
        print(f"Count for {expected_hello_tuple}: {word_counts_data[expected_hello_tuple]}")
        assert word_counts_data[expected_hello_tuple] == 2
    else:
        print(f"Error: {expected_hello_tuple} not found in word_counts_data.")
        assert False

    expected_ab_tuple = ('a', 'b', WORD_SUFFIX)
    assert word_counts_data[expected_ab_tuple] == 1

    # Test with characters not in initial alphabet
    corpus_with_foreign_chars = "hello world café 123"
    simple_alpha = set(['h', 'e', 'l', 'o', 'w', 'r', 'd', '1', '2', '3']) # 'c', 'a', 'f', 'é' are not here
    print(f"\nTesting with restricted alphabet: {simple_alpha}")
    word_counts_foreign = get_initial_vocab_and_word_counts(corpus_with_foreign_chars, simple_alpha)
    
    print("\nWord Counts (foreign chars, restricted alphabet):")
    for char_list_tuple, count in word_counts_foreign.items():
        print(f"  {char_list_tuple} : {count}")
    
    # 'café' -> \w+ gives 'café'. Filtered by simple_alpha, it might become empty or partial.
    # Current logic: [char for char in word_str if char in initial_alphabet_chars]
    # 'café' filtered by simple_alpha -> [] because c,a,f,é are not in simple_alpha.
    # So, it should be skipped.
    # '123' filtered by simple_alpha -> ['1','2','3'] -> ('1','2','3','</w>')
    expected_123_tuple = ('1', '2', '3', WORD_SUFFIX)
    assert expected_123_tuple in word_counts_foreign
    assert word_counts_foreign[expected_123_tuple] == 1
    
    # Check that 'café' (or its parts) didn't make it if its chars weren't in alphabet
    found_cafe_related = False
    for k_tuple in word_counts_foreign.keys():
        if 'c' in k_tuple or 'a' in k_tuple or 'f' in k_tuple or 'é' in k_tuple:
            found_cafe_related = True
            break
    assert not found_cafe_related

    print("\nget_initial_vocab_and_word_counts function created and basic tests passed.")

# Placeholders for next functions
def calculate_pair_scores(word_counts, current_vocab):
    """
    Calculates scores for all adjacent pairs of subwords/tokens.
    Score(s1, s2) = count(s1s2) / (count(s1) * count(s2))

    Args:
        word_counts (collections.Counter): Frequencies of words, where words
            are represented as tuples of current subword strings.
            e.g., {('h', 'e', 'l', 'l', 'o', '</w>'): 5, ('p', 'l', 'ay', '</w>'): 2}
        current_vocab (set): The current vocabulary of known subword strings.
                             (Not directly used in this implementation as pairs are derived
                              from word_counts, which should only contain known subwords.
                              Could be used for validation if needed.)

    Returns:
        dict: A dictionary where keys are pairs of subwords (tuple) and
              values are their calculated scores.
    """
    pair_frequencies = collections.defaultdict(int)
    individual_token_frequencies = collections.defaultdict(int)

    # Step 1 & 2: Calculate frequencies of pairs and individual tokens
    for word_tuple, count in word_counts.items():
        # word_tuple is like ('h', 'e', 'l', 'l', 'o', '</w>')
        # or after merges, like ('pl', 'ay', '</w>')
        if not word_tuple:
            continue
            
        for i in range(len(word_tuple) - 1):
            s1 = word_tuple[i]
            s2 = word_tuple[i+1]
            pair_frequencies[(s1, s2)] += count
        
        for token in word_tuple:
            individual_token_frequencies[token] += count
            
    pair_scores = {}
    # Step 3: Calculate scores for each pair
    for (s1, s2), count_s1s2 in pair_frequencies.items():
        count_s1 = individual_token_frequencies[s1]
        count_s2 = individual_token_frequencies[s2]
        
        if count_s1 == 0 or count_s2 == 0:
            # This should ideally not happen if tokens are in word_counts
            # print(f"Warning: Zero frequency for token in pair ({s1}, {s2}). count_s1={count_s1}, count_s2={count_s2}")
            score = 0.0 # Or handle as an error, or skip
        else:
            score = count_s1s2 / (count_s1 * count_s2)
        pair_scores[(s1, s2)] = score
        
    # print(f"--- Pair Scores Calculation ---")
    # print(f"Total unique pairs found: {len(pair_frequencies)}")
    # print(f"Total unique individual tokens counted: {len(individual_token_frequencies)}")
    # print(f"Sample pair scores (first 5): {list(pair_scores.items())[:5]}")
    return pair_scores

def merge_best_pair(best_pair_tokens, word_counts_in):
    """
    Merges the best pair of subwords into a single subword in all word representations.

    Args:
        best_pair_tokens (tuple): The pair of subword strings to merge, e.g., ('p', 'l').
        word_counts_in (collections.Counter): Input frequencies of words, where words
            are represented as tuples of current subword strings.

    Returns:
        collections.Counter: New word frequencies after merging the pair.
    """
    word_counts_out = collections.defaultdict(int)
    
    # The new merged subword token is simply the concatenation of the pair.
    # e.g., if best_pair_tokens is ('p', 'l'), merged_token_str is "pl".
    s1, s2 = best_pair_tokens
    merged_token_str = s1 + s2

    for current_word_tuple, freq in word_counts_in.items():
        new_word_representation = []
        i = 0
        while i < len(current_word_tuple):
            # Check if the current token and the next one form the best_pair_tokens
            if i < len(current_word_tuple) - 1 and \
               (current_word_tuple[i], current_word_tuple[i+1]) == best_pair_tokens:
                new_word_representation.append(merged_token_str)
                i += 2 # Skip the two merged tokens
            else:
                new_word_representation.append(current_word_tuple[i])
                i += 1
        word_counts_out[tuple(new_word_representation)] += freq
        
    # print(f"--- Merged Pair {best_pair_tokens} into {merged_token_str} ---")
    # print(f"Input word_counts size: {len(word_counts_in)}")
    # print(f"Output word_counts size: {len(word_counts_out)}")
    # print(f"Sample output counts (first 5): {list(word_counts_out.items())[:5]}")
    return word_counts_out

def train_wordpiece(corpus_text, target_vocab_size, initial_alphabet_chars_param=None):
    """
    Trains the WordPiece model.

    Args:
        corpus_text (str): The raw text corpus.
        target_vocab_size (int): The desired final vocabulary size.
        initial_alphabet_chars_param (set, optional): A set of characters for the initial alphabet.
            If None, it's derived from the corpus.

    Returns:
        tuple: (
            final_vocab (list): The final vocabulary (list of subword strings).
            merge_rules (list): List of merge operations (optional, for debugging).
        )
    """
    print("--- Starting WordPiece Training ---")

    # 1. Initialize Vocabulary
    if initial_alphabet_chars_param is None:
        print("Deriving initial alphabet from corpus...")
        # Basic derivation: unique characters in the corpus
        # For WordPiece, it's common to start with all single characters found.
        # Using regex to find "words" first, then characters within those words.
        words_for_alphabet = re.findall(r'\b\w+\b', corpus_text.lower().strip())
        initial_alphabet_chars = set(char for word in words_for_alphabet for char in word)
        print(f"Derived initial alphabet of size {len(initial_alphabet_chars)}")
    else:
        initial_alphabet_chars = set(initial_alphabet_chars_param) # Ensure it's a copy

    # current_vocab will store subword strings
    current_vocab = set(initial_alphabet_chars)
    current_vocab.add(WORD_SUFFIX) # WORD_SUFFIX must be part of the processing alphabet
    current_vocab.add(UNK_TOKEN)   # UNK_TOKEN is a standard part of the vocab
    
    # The initial_alphabet_chars for get_initial_vocab_and_word_counts should be
    # the character set used to filter/construct initial words.
    # This should align with what's in current_vocab excluding special tokens unless they are single chars.
    # For simplicity, the get_initial_vocab_and_word_counts will use the provided/derived initial_alphabet_chars.
    
    print(f"Initial vocab size (incl. specials): {len(current_vocab)}")
    # print(f"Initial vocab sample: {list(current_vocab)[:10]}")


    # 2. Initialize Word Counts
    # This function expects initial_alphabet_chars to define valid characters for words.
    word_counts = get_initial_vocab_and_word_counts(corpus_text, initial_alphabet_chars)
    
    merge_rules = [] # Optional: to store the sequence of merges

    # 3. Iterative Merging
    num_merges_done = 0
    while len(current_vocab) < target_vocab_size:
        print(f"\n--- Iteration {num_merges_done + 1} ---")
        print(f"Current vocab size: {len(current_vocab)} / Target: {target_vocab_size}")

        # Calculate scores for current pairs
        pair_scores = calculate_pair_scores(word_counts, current_vocab)

        if not pair_scores:
            print("No more pairs to merge. Stopping training early.")
            break

        # Find the best pair to merge (highest score)
        # Sort by score (desc), then by pair lexicographically for tie-breaking (optional but good for consistency)
        best_pair_tokens = max(pair_scores, key=lambda p: (pair_scores[p], p))
        best_score = pair_scores[best_pair_tokens]
        
        if best_score == 0: # Or some other threshold if scores can be non-positive
            print("No more valid pairs with positive scores. Stopping.")
            break

        # Create the new subword
        new_subword = best_pair_tokens[0] + best_pair_tokens[1]

        # Add to vocabulary (if it's truly new - set handles this)
        if new_subword in current_vocab:
            # This might happen if a pair's score becomes highest again after other merges changed counts.
            # Or if tie-breaking leads to re-evaluating an existing subword.
            # It's generally unexpected if pairs are unique and merging always creates a new string.
            # However, to be safe, we can simply not add it if it exists and perhaps remove the pair from future consideration.
            # For now, let's assume this is rare or handled by max score logic.
            # A simple fix is to remove the pair from pair_scores and find the next best one.
            # But for this implementation, we'll proceed, and if vocab size doesn't increase, loop eventually ends.
             print(f"Warning: New subword '{new_subword}' from pair {best_pair_tokens} (score {best_score}) already in vocab. Trying to continue.")
             # To prevent an infinite loop if this happens and vocab size doesn't grow:
             # One strategy: remove this pair from current consideration and pick next best.
             # For now, we assume target_vocab_size condition or no_more_pairs will terminate.


        print(f"Best pair: {best_pair_tokens} -> New subword: '{new_subword}' (Score: {best_score:.4f})")
        current_vocab.add(new_subword)
        merge_rules.append(best_pair_tokens) # Store the pair that was merged

        # Update word_counts by merging this pair
        word_counts = merge_best_pair(best_pair_tokens, word_counts)
        
        num_merges_done += 1
        if len(current_vocab) >= target_vocab_size:
            print(f"Target vocabulary size {target_vocab_size} reached.")
            break
            
    print("\n--- WordPiece Training Complete ---")
    print(f"Final vocabulary size: {len(current_vocab)}")
    print(f"Total merges performed: {num_merges_done}")
    
    # Sort for consistent output, e.g., by length then alphabetically
    final_vocab_list = sorted(list(current_vocab), key=lambda s: (len(s), s))
    
    return final_vocab_list, merge_rules


def tokenize_word_with_wordpiece(word_string, vocab_set, initial_alphabet_chars):
    """
    Tokenizes a single word string using the WordPiece vocabulary.
    Applies '##' prefix to subwords that are not the first token of the word.

    Args:
        word_string (str): The word to tokenize (without WORD_SUFFIX).
        vocab_set (set): The WordPiece vocabulary (set of subword strings).
        initial_alphabet_chars (set): The set of characters considered part of the known alphabet.

    Returns:
        list: A list of subword strings, with '##' prefixes where appropriate.
              Returns ['[UNK]'] if the entire word cannot be tokenized.
    """
    if not word_string:
        return []

    # Append WORD_SUFFIX for tokenization logic
    current_word_segment = word_string + WORD_SUFFIX
    subword_tokens = []
    
    original_len = len(current_word_segment)
    
    while current_word_segment:
        longest_match = ""
        # Find the longest subword in vocab that is a prefix of current_word_segment
        for i in range(len(current_word_segment), 0, -1):
            prefix = current_word_segment[:i]
            if prefix in vocab_set:
                longest_match = prefix
                break
        
        if longest_match:
            subword_tokens.append(longest_match)
            current_word_segment = current_word_segment[len(longest_match):]
        else:
            # No subword found in vocab, this indicates an issue or unknown character(s)
            # Fallback: if the first char is in the initial alphabet (and thus should be in vocab)
            # try to tokenize it as a single character. This covers cases where single chars
            # from the alphabet might not have formed longer subwords or were not explicitly added
            # to vocab as part of every possible segment.
            # However, our vocab should contain all initial alphabet chars + WORD_SUFFIX.
            
            # If the entire word cannot be broken down by known vocab items,
            # it might be better to return UNK for the whole word.
            # For robust tokenization, single characters from initial_alphabet_chars
            # (and those chars + WORD_SUFFIX) should be in vocab.
            
            # Let's try a character-by-character fallback for unknown parts
            first_char = current_word_segment[0]
            # Check if the single character itself (or with suffix) is in vocab
            # This is to handle cases where the initial alphabet chars are the fallback.
            char_as_token_with_suffix = first_char + WORD_SUFFIX
            char_as_token = first_char

            if len(current_word_segment) == 1 and char_as_token in vocab_set: # Should not happen if suffix is used
                 subword_tokens.append(char_as_token)
                 current_word_segment = ""
            elif len(current_word_segment) > 1 and char_as_token_with_suffix == current_word_segment and char_as_token_with_suffix in vocab_set:
                 subword_tokens.append(char_as_token_with_suffix)
                 current_word_segment = ""
            elif char_as_token in vocab_set: # Check for single char if it's in vocab
                 subword_tokens.append(char_as_token)
                 current_word_segment = current_word_segment[1:]
            elif UNK_TOKEN in vocab_set: # If single char is not in vocab, use UNK
                subword_tokens.append(UNK_TOKEN)
                current_word_segment = current_word_segment[1:] # Consume one char
            else: # Should not happen if UNK_TOKEN is always in vocab
                print(f"Error: Cannot tokenize '{word_string}'. Stuck at '{current_word_segment}'. UNK_TOKEN not in vocab?")
                return [UNK_TOKEN if UNK_TOKEN in vocab_set else word_string] # Fallback for the whole word

    # Apply '##' prefix to non-initial tokens
    if not subword_tokens: # Should not happen if logic is correct
        if UNK_TOKEN in vocab_set: return [UNK_TOKEN]
        return [word_string + WORD_SUFFIX] # Should be caught by UNK logic

    formatted_tokens = [subword_tokens[0]] # First token as is
    for token in subword_tokens[1:]:
        formatted_tokens.append("##" + token)
        
    return formatted_tokens


def save_wordpiece_model(vocab_list, base_filename):
    """
    Saves the WordPiece vocabulary to a file.

    Args:
        vocab_list (list): A list of subword strings forming the vocabulary.
                           Assumed to be sorted or in the desired order for saving.
        base_filename (str): The base name for the output file.
                             ".wp_vocab" will be appended.
    """
    vocab_file = base_filename + ".wp_vocab"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for subword in vocab_list:
            f.write(subword + '\n')
    print(f"WordPiece vocabulary saved to: {vocab_file}")


def load_wordpiece_model(base_filename):
    """
    Loads a WordPiece vocabulary from a file.

    Args:
        base_filename (str): The base name of the model file.
                             Assumes ".wp_vocab" extension.

    Returns:
        list: A list of subword strings forming the vocabulary.
              Returns None if the file is not found.
    """
    vocab_file = base_filename + ".wp_vocab"
    loaded_vocab_list = []
    try:
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                loaded_vocab_list.append(line.strip())
        print(f"WordPiece vocabulary loaded from: {vocab_file}")
        return loaded_vocab_list
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found: {vocab_file}")
        return None

if __name__ == '__main__':
    print("--- WordPiece Implementation Demo ---")

    # 1. Define Corpus and Initial Alphabet
    corpus = """
    A simple sentence.
    Another sentence with more words.
    Learning WordPiece tokenization.
    unusual words like unlikelihood and antidisestablishmentarianism.
    日本語の単語もトークン化できます。
    """
    # For a more realistic scenario, derive initial_alphabet_chars from the corpus
    # or use a predefined set (e.g., all printable ASCII + some common Unicode characters)
    # Here, we derive it from the corpus for simplicity in this example.
    
    print("Deriving initial alphabet from corpus for training...")
    temp_words_for_alphabet = re.findall(r'\b\w+\b', corpus.lower().strip())
    derived_initial_alphabet = set(char for word in temp_words_for_alphabet for char in word)
    # Add specific characters if they might be missed by \w+ or are desired, e.g. '.'
    # derived_initial_alphabet.add('.') # Example if periods should be standalone tokens initially
    print(f"Derived initial alphabet (size {len(derived_initial_alphabet)}): {sorted(list(derived_initial_alphabet))[:30]}...")

    target_vocab_size = 200 # Small for demonstration; typical sizes are 30k-50k

    # 2. Train WordPiece model
    print(f"\n--- Training WordPiece Model (Target Vocab Size: {target_vocab_size}) ---")
    # Pass the derived alphabet. train_wordpiece will add UNK_TOKEN and WORD_SUFFIX.
    trained_vocab_list, merge_rules = train_wordpiece(corpus, target_vocab_size, initial_alphabet_chars_param=derived_initial_alphabet)
    
    print("\n--- Trained Model Details ---")
    print(f"Final Vocabulary size: {len(trained_vocab_list)}")
    # print(f"Sample of final vocabulary: {trained_vocab_list[:20]}")
    # print(f"Merge rules learned (first 5): {merge_rules[:5]}")

    # 3. Save the trained vocabulary
    model_basename = "wordpiece_test_model"
    print(f"\n--- Saving WordPiece Vocabulary to '{model_basename}.wp_vocab' ---")
    save_wordpiece_model(trained_vocab_list, model_basename)

    # 4. Load the vocabulary
    print(f"\n--- Loading WordPiece Vocabulary from '{model_basename}.wp_vocab' ---")
    loaded_vocab_list = load_wordpiece_model(model_basename)

    if loaded_vocab_list:
        print(f"Successfully loaded vocabulary of size: {len(loaded_vocab_list)}")
        # For tokenization, a set is faster for lookups
        loaded_vocab_set = set(loaded_vocab_list)

        # The initial_alphabet_chars used for tokenization should be consistent with training
        # This set is used by the tokenizer to know what constitutes a "known" single character
        # if a longer subword match isn't found. It should include chars that were in the initial vocab.
        # The `train_wordpiece` function adds UNK_TOKEN and WORD_SUFFIX to its internal vocab,
        # but `initial_alphabet_chars` for the tokenizer should be just the characters.
        
        # Re-affirm the alphabet for the tokenizer (characters that are considered "known" individually)
        # This should ideally be the same set of characters that `train_wordpiece` started with.
        tokenizer_initial_alphabet = derived_initial_alphabet # Use the same derived set.

        # 5. Tokenize sample words
        print("\n--- Tokenizing Sample Words with Loaded Model ---")
        test_words = [
            "sentence", "tokenization", "unlikelihood", "日本語", "antidisestablishmentarianism",
            "unknownword", "another", "##subword", "test.", "simple"
        ]
        
        for word in test_words:
            # Note: The tokenizer expects words *without* WORD_SUFFIX. It adds it internally.
            # Also, it needs the initial character alphabet to handle unknown characters correctly.
            tokens = tokenize_word_with_wordpiece(word, loaded_vocab_set, tokenizer_initial_alphabet)
            print(f"Word '{word}' -> Tokens: {tokens}")
            
        # Example to show handling of word not made of alphabet chars (if alphabet is restricted)
        print("\n--- Tokenizing with restricted alphabet for OOV demo ---")
        restricted_alpha = set(['a','b','c','d','e','f','g','h','i','j','k','l','m',
                                'n','o','p','q','r','s','t','u','v','w','x','y','z'])
        # We'd need a vocab trained only with this alphabet for a true test.
        # For now, use existing loaded_vocab_set but with restricted_alpha for tokenizer's fallback.
        # This mainly tests how UNK_TOKEN is applied for characters outside restricted_alpha.
        
        word_with_numbers = "word123" # '1','2','3' are not in restricted_alpha
        tokens_restricted = tokenize_word_with_wordpiece(word_with_numbers, loaded_vocab_set, restricted_alpha)
        print(f"Word '{word_with_numbers}' (restricted alpha) -> Tokens: {tokens_restricted}")
        # Expected: parts of "word" might be tokenized, "123" should become UNK if not in vocab
        # or if '1','2','3' are not in restricted_alpha and thus treated as unknown by tokenizer fallback.

    else:
        print("Failed to load vocabulary. Skipping tokenization demo.")

    print("\n--- WordPiece Demo Complete ---")
