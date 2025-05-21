import collections
import re # 用于分词

# 特殊的单词结束符
EOW = '</w>'

def get_initial_vocab_and_word_counts(corpus_text):
    """
    1. Splits the raw corpus text into words.
    2. Appends the EOW (End Of Word) symbol to each word.
    3. Converts words into space-separated character sequences (e.g., "low" -> "l o w </w>").
    4. Counts the frequency of these character sequences.
    5. Builds an initial vocabulary of unique individual characters.

    This BPE implementation operates at the character level.

    Args:
        corpus_text (str): The raw text corpus. It's recommended to be in lowercase.

    Returns:
        tuple: (
            word_char_counts (collections.Counter): Frequencies of words, where words
                are represented as space-separated character sequences
                e.g., {'l o w </w>': 5, ...}.
            initial_char_vocab (set): The initial set of unique individual characters
                found in the corpus.
        )
    """
    # Use regex to split into words (sequences of word characters).
    # .strip() handles leading/trailing whitespace for the whole corpus.
    # .lower() converts text to lowercase.
    raw_words = re.findall(r'\b\w+\b', corpus_text.strip().lower())

    # For each word, add EOW, split into characters, and join with spaces.
    # e.g., "low" -> "l o w </w>"
    # This character sequence representation is used for BPE operations.
    processed_word_char_sequences = []
    for word in raw_words:
        # Each word is converted to a list of its characters,
        # then joined by spaces, with EOW appended.
        processed_word_char_sequences.append(" ".join(list(word)) + " " + EOW)

    # Count frequencies of these character-sequence representations of words.
    word_char_counts = collections.Counter(processed_word_char_sequences)

    # Build the initial character vocabulary from the character sequences.
    # This vocabulary will initially contain all single characters.
    initial_char_vocab = set()
    for char_sequence_str in word_char_counts:
        initial_char_vocab.update(char_sequence_str.split()) # Splits "l o w </w>" into ['l','o','w','</w>']

    print("--- Step 0: Preprocessing and Initial Character Vocabulary ---")
    print(f"Word frequencies (represented as space-separated char sequences): {word_char_counts}")
    print(f"Initial character vocabulary: {sorted(list(initial_char_vocab))}\n")
    return word_char_counts, initial_char_vocab


def get_pair_stats(word_char_counts):
    """
    Counts the frequency of adjacent character pairs in all word representations.
    `word_char_counts` is a Counter where keys are space-separated character
    sequences (representing words or segments) and values are their frequencies.
    e.g., {'l o w </w>': 5, 'n e w e s t </w>': 6}

    Args:
        word_char_counts (collections.Counter): Current word/segment frequencies,
                                                where keys are space-separated
                                                character sequences.

    Returns:
        collections.Counter: Frequencies of adjacent character pairs,
                             e.g., {('l', 'o'): 7, ('e', 's'): 9}.
                             Each key is a tuple (char1, char2).
    """
    pair_stats = collections.defaultdict(int)
    for char_sequence_str, freq in word_char_counts.items():
        # Split the string "c1 c2 c3" into a list ['c1', 'c2', 'c3']
        chars_list = char_sequence_str.split()
        for i in range(len(chars_list) - 1):
            pair = (chars_list[i], chars_list[i+1])
            pair_stats[pair] += freq # Add frequency of the sequence containing this pair
    return pair_stats


def merge_pair(target_char_pair, word_char_counts_in):
    """
    Merges a specified character pair in all word/segment character sequence representations.
    For example, if target_char_pair = ('e', 's'), then a sequence "n e w e s t </w>"
    becomes "n e w es t </w>".

    Args:
        target_char_pair (tuple): The character pair to merge, e.g., ('e', 's').
        word_char_counts_in (collections.Counter): Input word/segment frequencies,
            where keys are space-separated character sequences.

    Returns:
        collections.Counter: New word/segment frequencies after merging the pair.
            Keys are the updated space-separated character sequences.
    """
    word_char_counts_out = collections.defaultdict(int)
    # String representation of the pair to find, e.g., ('e', 's') -> 'e s'
    pair_to_replace_str = ' '.join(target_char_pair)
    # The merged segment, e.g., ('e', 's') -> 'es'
    merged_segment = ''.join(target_char_pair)

    for char_sequence_str, freq in word_char_counts_in.items():
        # Replace the spaced pair string with the merged segment string
        # Example: "n e w e s t </w>".replace('e s', 'es') -> "n e w es t </w>"
        new_char_sequence_str = char_sequence_str.replace(pair_to_replace_str, merged_segment)
        word_char_counts_out[new_char_sequence_str] += freq

    return word_char_counts_out


def train_bpe(corpus_text, num_merges):
    """
    Trains the BPE model by iteratively merging the most frequent character pairs.

    Args:
        corpus_text (str): The raw text corpus.
        num_merges (int): The number of merge operations to perform.

    Returns:
        tuple: (
            final_char_vocab (set): The final vocabulary, containing single characters
                                     and merged character sequences (subwords).
            merge_rules (list): A list of merge rules (character pairs like ('c1','c2'))
                                 in the order they were learned.
        )
    """
    # 1. Initial Setup:
    # word_char_counts: keys are character sequences (e.g., "l o w </w>"), values are frequencies.
    # current_char_vocab: set of unique characters and, progressively, merged character sequences.
    word_char_counts, current_char_vocab = get_initial_vocab_and_word_counts(corpus_text)

    # Stores merge rules (tuples of character_pair_to_merge) in order of learning.
    merge_rules = []

    print("--- Starting BPE Training Iterations ---\n")
    for i in range(num_merges):
        print(f"--- Merge Iteration {i + 1}/{num_merges} ---")

        # 2. Count frequencies of adjacent character pairs in the current set of sequences.
        # pair_stats: keys are tuples like ('c1', 'c2'), values are their frequencies.
        pair_stats = get_pair_stats(word_char_counts)

        if not pair_stats:
            print("No more character pairs to merge. Stopping training early.")
            break # Stop if no pairs can be merged (e.g., all sequences are single tokens).

        # 3. Find the most frequent character pair.
        # max() on dict uses keys; key=pair_stats.get sorts by values.
        best_char_pair = max(pair_stats, key=pair_stats.get)
        best_pair_freq = pair_stats[best_char_pair]
        print(f"Most frequent pair: {best_char_pair} (Frequency: {best_pair_freq})")

        # 4. Merge this character pair to form a new character sequence (segment/subword).
        # Add this new segment to the vocabulary.
        merged_char_sequence = "".join(best_char_pair)
        current_char_vocab.add(merged_char_sequence)
        merge_rules.append(best_char_pair) # Record this merge rule.

        # 5. Update the corpus representation by applying the merge to all affected sequences.
        word_char_counts = merge_pair(best_char_pair, word_char_counts)
        print(f"After merge, new segment '{merged_char_sequence}' added to vocabulary.")
        # Optional: print details for debugging during development
        # print(f"Updated word_char_counts: {word_char_counts}")
        # print(f"Current vocabulary size: {len(current_char_vocab)}, Vocabulary (sample): {sorted(list(current_char_vocab))[:10]}\n")
        print("-" * 30)


    print("\n--- BPE Training Complete ---")
    print(f"Final character vocabulary size: {len(current_char_vocab)}")
    print(f"Final vocabulary (sample): {sorted(list(current_char_vocab))[:20]} ...") # Display a sample
    print(f"Learned merge rules (Total: {len(merge_rules)}):")
    for idx, rule in enumerate(merge_rules):
        # rule is like ('c1', 'c2'), ''.join(rule) is 'c1c2'
        print(f"  {idx+1}. {rule} -> {''.join(rule)}")

    return current_char_vocab, merge_rules


def tokenize_word_with_bpe(word_string, merge_rules):
    """
    Tokenizes a single word into subword units using the learned BPE merge rules.
    This process is character-based.

    Args:
        word_string (str): The word to tokenize (e.g., "lowest"). It should not contain EOW.
        merge_rules (list): A list of merge rules (character pairs, e.g., ('c1', 'c2'))
                             in the order they were learned.

    Returns:
        list: A list of tokens (characters or merged character sequences, e.g. ['low', 'est</w>']).
    """
    if not word_string:
        return []

    # 1. Preprocessing: Split the word into individual characters and append EOW.
    # e.g., "lowest" -> ['l', 'o', 'w', 'e', 's', 't', '</w>']
    char_tokens = list(word_string) + [EOW]
    # print(f"Initial character tokens: {char_tokens}")

    # 2. Iteratively apply merge rules in the learned order.
    #    For each rule, scan the current token sequence and apply the merge where possible.
    for char_pair_to_merge in merge_rules: # e.g., char_pair_to_merge = ('e', 's')
        new_char_tokens = []
        i = 0
        while i < len(char_tokens):
            # Check if the current token and the next one form the pair to merge.
            if i < len(char_tokens) - 1 and \
               (char_tokens[i], char_tokens[i+1]) == char_pair_to_merge:
                # If yes, merge them into a single token.
                new_char_tokens.append("".join(char_pair_to_merge))
                i += 2 # Move past the two merged tokens.
            else:
                # If no, keep the current token as is.
                new_char_tokens.append(char_tokens[i])
                i += 1
        char_tokens = new_char_tokens # Update token sequence for the next rule.
        # print(f"Applied rule {char_pair_to_merge} -> {''.join(char_pair_to_merge)}, Tokens: {char_tokens}")

    return char_tokens


# --- Save and Load BPE Model ---
def save_bpe_model(char_vocab, merge_rules, base_filename):
    """
    Saves the BPE model (vocabulary and merge rules) to files.

    Args:
        char_vocab (set): The character vocabulary (including merged sequences).
        merge_rules (list): A list of merge rules (tuples of char_pair_to_merge).
        base_filename (str): The base name for the output files.
                             ".vocab" and ".merges" will be appended.
    """
    vocab_file = base_filename + ".vocab"
    merges_file = base_filename + ".merges"

    # Save vocabulary
    # Store as a sorted list for consistency, one token per line.
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token in sorted(list(char_vocab)):
            f.write(token + '\n')
    print(f"Vocabulary saved to: {vocab_file}")

    # Save merge rules
    # Each rule is a tuple, e.g., ('e', 's'). Store as "e s" per line.
    with open(merges_file, 'w', encoding='utf-8') as f:
        for rule in merge_rules:
            f.write(f"{rule[0]} {rule[1]}\n")
    print(f"Merge rules saved to: {merges_file}")


def load_bpe_model(base_filename):
    """
    Loads a BPE model (vocabulary and merge rules) from files.

    Args:
        base_filename (str): The base name of the model files.
                             Assumes ".vocab" and ".merges" extensions.

    Returns:
        tuple: (
            char_vocab (set): The loaded character vocabulary.
            merge_rules (list): The loaded list of merge rules (tuples).
        )
    Returns None, None if files are not found.
    """
    vocab_file = base_filename + ".vocab"
    merges_file = base_filename + ".merges"
    
    loaded_vocab = set()
    loaded_merge_rules = []

    try:
        # Load vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                loaded_vocab.add(line.strip())
        print(f"Vocabulary loaded from: {vocab_file}")

        # Load merge rules
        # Each line is "c1 c2", convert back to tuple ('c1', 'c2').
        with open(merges_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    loaded_merge_rules.append((parts[0], parts[1]))
                elif line.strip(): # Non-empty line that isn't a pair
                    print(f"Warning: Malformed rule in {merges_file}: '{line.strip()}'")
        print(f"Merge rules loaded from: {merges_file}")
        
        return loaded_vocab, loaded_merge_rules

    except FileNotFoundError:
        print(f"Error: Model files not found for base_filename: {base_filename}")
        return None, None


# --- 主程序和示例 ---
if __name__ == "__main__":
    # 示例语料库 (来自Sennrich et al., 2016 BPE论文的简化例子)
    corpus = """
    low low low low low
    lower lower
    newest newest newest newest newest newest
    widest widest widest
    """
    # 为了更明显地看到效果，我们稍微增加一点数据
    corpus += """
    hugging hugging face face
    a new new algorithm
    the widest possible view
    """

    num_merges = 20 # 设定合并次数

    # 训练BPE
    print("--- Training Original BPE Model ---")
    original_char_vocab, original_merge_rules = train_bpe(corpus, num_merges)

    print("\n--- Tokenizing with Original Model ---")
    test_words = ["lowest", "newer", "widely", "huggingface", "unknownword", "a", "new", "algorithm", "face"]
    original_tokenized_results = {}
    for word in test_words:
        tokenized_output = tokenize_word_with_bpe(word, original_merge_rules)
        original_tokenized_results[word] = tokenized_output
        print(f"Word '{word}' -> Original Tokens: {tokenized_output}")

    # 检查一些特殊的token
    print("\n--- Checking Original Vocabulary and Rules ---")
    print(f"Is 'est</w>' in original vocabulary? : {'est</w>' in original_char_vocab}")
    print(f"Is 'low' in original vocabulary? : {'low' in original_char_vocab}")
    print(f"Is 'hugg' in original vocabulary? : {'hugg' in original_char_vocab}")

    # --- Save and Load Demonstration ---
    model_basename = "bpe_char_model_test"
    print(f"\n--- Saving BPE Model to '{model_basename}.*' ---")
    save_bpe_model(original_char_vocab, original_merge_rules, model_basename)

    print(f"\n--- Loading BPE Model from '{model_basename}.*' ---")
    loaded_char_vocab, loaded_merge_rules = load_bpe_model(model_basename)

    if loaded_char_vocab is not None and loaded_merge_rules is not None:
        print("\n--- Verifying Loaded Model ---")
        # 1. Compare vocabularies (order might differ due to set properties, so compare sizes and content)
        if len(original_char_vocab) == len(loaded_char_vocab) and \
           original_char_vocab == loaded_char_vocab:
            print("Vocabulary successfully saved and loaded: Identical.")
        else:
            print("Vocabulary mismatch after loading!")
            # print(f"Original vocab: {sorted(list(original_char_vocab))}")
            # print(f"Loaded vocab  : {sorted(list(loaded_char_vocab))}")


        # 2. Compare merge rules (order and content should be identical)
        if original_merge_rules == loaded_merge_rules:
            print("Merge rules successfully saved and loaded: Identical.")
        else:
            print("Merge rules mismatch after loading!")
            # print(f"Original rules: {original_merge_rules}")
            # print(f"Loaded rules  : {loaded_merge_rules}")

        print("\n--- Tokenizing with Loaded Model ---")
        loaded_tokenized_results = {}
        for word in test_words:
            tokenized_output = tokenize_word_with_bpe(word, loaded_merge_rules)
            loaded_tokenized_results[word] = tokenized_output
            print(f"Word '{word}' -> Loaded Tokens: {tokenized_output}")

        # 3. Compare tokenization results
        all_tokenization_matched = True
        for word in test_words:
            if original_tokenized_results[word] != loaded_tokenized_results[word]:
                all_tokenization_matched = False
                print(f"Mismatch for word '{word}':")
                print(f"  Original: {original_tokenized_results[word]}")
                print(f"  Loaded:   {loaded_tokenized_results[word]}")
        
        if all_tokenization_matched:
            print("\nTokenization results with original and loaded models are IDENTICAL.")
        else:
            print("\nTokenization results with original and loaded models DIFFER.")

    else:
        print("Failed to load the model. Skipping verification.")


    # 看看如果合并次数很少会怎样 (This part can remain as is, or be removed if not central to the test)
    # print("\n--- 测试较少合并次数 (例如 5 次) ---")
    # _, few_rules = train_bpe(corpus, 5) # Re-train for this specific test
    # for word in ["lowest", "newer"]:
    #     tokenized_output = tokenize_word_with_bpe(word, few_rules)
    #     print(f"单词 '{word}' (5次合并) -> 分词结果: {tokenized_output}")
