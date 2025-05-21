import unittest
import os
import sys
import collections

# Add project root to sys.path to allow importing tokenizer modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from BPE import (
    train_bpe, tokenize_word_with_bpe, save_bpe_model, load_bpe_model,
    get_initial_vocab_and_word_counts, get_pair_stats, merge_pair, EOW
)

class TestBPE(unittest.TestCase):

    def setUp(self):
        self.corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
        # A smaller corpus for more predictable merge behavior in some tests
        self.small_corpus = "low low low new new"
        self.test_model_base = os.path.join(project_root, "tests", "test_bpe_model_files", "test_bpe_model")
        
        # Ensure the directory for model files exists
        os.makedirs(os.path.dirname(self.test_model_base), exist_ok=True)

        # Clean up any previous test model files
        for ext in ['.vocab', '.merges']:
            if os.path.exists(self.test_model_base + ext):
                os.remove(self.test_model_base + ext)

    def tearDown(self):
        # Clean up test model files after tests
        for ext in ['.vocab', '.merges']:
            if os.path.exists(self.test_model_base + ext):
                os.remove(self.test_model_base + ext)
        # Attempt to remove the directory if it's empty
        try:
            os.rmdir(os.path.dirname(self.test_model_base))
        except OSError:
            pass # Directory not empty or other error, fine for cleanup

    def test_get_initial_vocab_and_word_counts(self):
        word_char_counts, initial_char_vocab = get_initial_vocab_and_word_counts(self.small_corpus)
        
        # Expected word_char_counts for "low low low new new"
        # "low" -> "l o w </w>" (freq 3)
        # "new" -> "n e w </w>" (freq 2)
        expected_low_key = " ".join(list("low")) + " " + EOW
        expected_new_key = " ".join(list("new")) + " " + EOW
        
        self.assertEqual(word_char_counts[expected_low_key], 3)
        self.assertEqual(word_char_counts[expected_new_key], 2)
        self.assertEqual(len(word_char_counts), 2)

        expected_vocab = set(['l', 'o', 'w', 'n', 'e', EOW])
        self.assertEqual(initial_char_vocab, expected_vocab)

    def test_get_pair_stats(self):
        # word_char_counts from "low low low new new"
        word_char_counts = collections.Counter({
            "l o w </w>": 3,
            "n e w </w>": 2
        })
        pair_stats = get_pair_stats(word_char_counts)
        # Expected pairs:
        # ('l','o'): 3
        # ('o','w'): 3
        # ('w',EOW): 3 (from low) + 2 (from new) = 5
        # ('n','e'): 2
        # ('e','w'): 2
        self.assertEqual(pair_stats[('l','o')], 3)
        self.assertEqual(pair_stats[('o','w')], 3)
        self.assertEqual(pair_stats[('w',EOW)], 5)
        self.assertEqual(pair_stats[('n','e')], 2)
        self.assertEqual(pair_stats[('e','w')], 2)
        self.assertEqual(len(pair_stats), 5)

    def test_merge_pair(self):
        word_char_counts_in = collections.Counter({
            "l o w </w>": 3,
            "n e w </w>": 2
        })
        # Merge ('o', 'w') -> "ow"
        target_pair = ('o', 'w')
        word_char_counts_out = merge_pair(target_pair, word_char_counts_in)
        
        expected_low_merged_key = "l ow </w>" # l o w -> l ow
        self.assertEqual(word_char_counts_out[expected_low_merged_key], 3)
        # "n e w </w>" is not affected by merging ('o','w') if 'o w' isn't present
        # Oh, wait, merge_pair replaces 'o w' with 'ow'.
        # If the target pair is ('o', 'w'), it will replace "o w" in "l o w </w>"
        # It will not affect "n e w </w>"
        original_new_key = "n e w </w>"
        self.assertEqual(word_char_counts_out[original_new_key], 2)
        self.assertEqual(len(word_char_counts_out), 2)

        # Merge ('e', 'w') -> "ew" on the output of previous merge
        target_pair_2 = ('e', 'w')
        word_char_counts_out_2 = merge_pair(target_pair_2, word_char_counts_out)
        expected_new_merged_key = "n ew </w>"
        self.assertEqual(word_char_counts_out_2[expected_new_merged_key], 2)
        self.assertEqual(word_char_counts_out_2[expected_low_merged_key], 3) # from previous merge
        self.assertEqual(len(word_char_counts_out_2), 2)


    def test_train_and_tokenize(self):
        # Test basic training and tokenization
        vocab, rules = train_bpe(self.corpus, num_merges=20) # Increased merges for more substantial test
        self.assertTrue(len(vocab) > 0)
        self.assertTrue(len(rules) <= 20)
        
        tokens_lowest = tokenize_word_with_bpe("lowest", rules)
        self.assertIsInstance(tokens_lowest, list)
        # Example: if 'est</w>' was a merge, 'est</w>' should be in tokens_lowest
        # This depends heavily on the merges that happen with the full corpus.
        # Let's test with the small corpus and few merges for predictability
        
        small_vocab, small_rules = train_bpe(self.small_corpus, num_merges=3)
        # Corpus: "low low low new new"
        # Initial counts: {"l o w </w>": 3, "n e w </w>": 2}
        # Initial vocab: {'l','o','w','n','e','</w>'}
        # Pairs: ('l','o'):3, ('o','w'):3, ('w','</w>'):5, ('n','e'):2, ('e','w'):2
        # 1. Merge ('w','</w>') -> "w</w>". Vocab adds "w</w>". Rules: [(('w','</w>'))]
        #    Counts: {"l o w</w>": 3, "n e w</w>": 2}
        # 2. Merge ('l','o') -> "lo". Vocab adds "lo". Rules: [(('w','</w>')), (('l','o'))]
        #    Counts: {"lo w</w>": 3, "n e w</w>": 2}
        # 3. Merge ('lo','w</w>') -> "low</w>". Vocab adds "low</w>". Rules: [..., (('lo','w</w>'))]
        #    Counts: {"low</w>": 3, "n e w</w>": 2}
        
        self.assertIn(('w', EOW), small_rules) # Based on freq 5
        self.assertIn('w'+EOW, small_vocab)

        tokens_low = tokenize_word_with_bpe("low", small_rules)
        # With 3 merges as above: rules = [('w','</w>'), ('l','o'), ('lo','w</w>')]
        # "low" -> ['l','o','w','</w>']
        # Rule 1 ('w','</w>'): ['l','o','w</w>']
        # Rule 2 ('l','o'): ['lo','w</w>']
        # Rule 3 ('lo','w</w>'): ['low</w>']
        self.assertEqual(tokens_low, ['low'+EOW])

        tokens_newer = tokenize_word_with_bpe("newer", small_rules)
        # "newer" -> ['n','e','w','e','r','</w>']
        # Rule 1 ('w','</w>'): ['n','e','w','e','r</w>']
        # Rule 2,3 no effect
        self.assertEqual(tokens_newer, ['n','e','w','e','r'+EOW]) # assuming 'r' is not in small_corpus vocab, but it is if not filtered
                                                              # The tokenizer does not filter, it just uses rules.
                                                              # initial_char_vocab from get_initial_vocab_and_word_counts
                                                              # for small_corpus is {'l', 'o', 'w', '</w>', 'n', 'e'}
                                                              # 'r' is not in this set.
                                                              # tokenize_word_with_bpe does list(word_string) + [EOW]
                                                              # so 'r' will be included.

    def test_save_load_model(self):
        original_vocab, original_rules = train_bpe(self.corpus, num_merges=5)
        save_bpe_model(original_vocab, original_rules, self.test_model_base)
        
        self.assertTrue(os.path.exists(self.test_model_base + ".vocab"))
        self.assertTrue(os.path.exists(self.test_model_base + ".merges"))

        loaded_vocab, loaded_rules = load_bpe_model(self.test_model_base)
        self.assertEqual(original_vocab, loaded_vocab)
        self.assertEqual(original_rules, loaded_rules)

        word = "newer"
        tokens_original = tokenize_word_with_bpe(word, original_rules)
        tokens_loaded = tokenize_word_with_bpe(word, loaded_rules)
        self.assertEqual(tokens_original, tokens_loaded)

    def test_empty_input_tokenize(self):
        _, rules = train_bpe(self.small_corpus, num_merges=2)
        tokens = tokenize_word_with_bpe("", rules)
        self.assertEqual(tokens, [])

    def test_unknown_chars_tokenize(self):
        # Train on "low new"
        _, rules = train_bpe(self.small_corpus, num_merges=3)
        # Tokenize "apple" - 'a', 'p', 'p', 'l', 'e' are not in the rules from "low new"
        # Rules learned were [('w','</w>'), ('l','o'), ('lo','w</w>')]
        # "apple" -> ['a','p','p','l','e','</w>']
        # After rule ('w','</w>'): ['a','p','p','l','e</w>']
        # After rule ('l','o'): no change
        # After rule ('lo', 'w</w>'): no change
        tokens = tokenize_word_with_bpe("apple", rules)
        self.assertEqual(tokens, ['a','p','p','l','e'+EOW])


if __name__ == '__main__':
    unittest.main()
