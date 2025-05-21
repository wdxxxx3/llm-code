import unittest
import os
import sys
import collections

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from BBPE import (
    train_bbpe, tokenize_word_with_bbpe, save_bbpe_model, load_bbpe_model,
    get_initial_vocab_and_word_byte_counts, get_byte_pair_stats, merge_byte_pair
)

class TestBBPE(unittest.TestCase):

    def setUp(self):
        self.corpus_ascii = "low low low new new"
        self.corpus_multibyte = "你好 你好 世界 世界 世界" # nihao nihao shijie shijie shijie
        self.test_model_base = os.path.join(project_root, "tests", "test_bbpe_model_files", "test_bbpe_model")

        os.makedirs(os.path.dirname(self.test_model_base), exist_ok=True)
        for ext in ['.bvocab', '.bmerges']:
            if os.path.exists(self.test_model_base + ext):
                os.remove(self.test_model_base + ext)

    def tearDown(self):
        for ext in ['.bvocab', '.bmerges']:
            if os.path.exists(self.test_model_base + ext):
                os.remove(self.test_model_base + ext)
        try:
            os.rmdir(os.path.dirname(self.test_model_base))
        except OSError:
            pass

    def test_get_initial_vocab_and_word_byte_counts(self):
        word_token_counts, initial_byte_token_vocab = get_initial_vocab_and_word_byte_counts(self.corpus_ascii)
        
        # "low" -> ((108,), (111,), (119,))
        low_bytes = "low".encode('utf-8')
        expected_low_key = tuple((b,) for b in low_bytes)
        
        # "new" -> ((110,), (101,), (119,))
        new_bytes = "new".encode('utf-8')
        expected_new_key = tuple((b,) for b in new_bytes)
        
        self.assertEqual(word_token_counts[expected_low_key], 3)
        self.assertEqual(word_token_counts[expected_new_key], 2)
        self.assertEqual(len(word_token_counts), 2)

        expected_vocab_tokens = set()
        for char_code in "lowne".encode('utf-8'): # Unique bytes from "low", "new"
            expected_vocab_tokens.add((char_code,))
        self.assertEqual(initial_byte_token_vocab, expected_vocab_tokens)

    def test_get_byte_pair_stats(self):
        # Using tokenized structure for word_token_counts
        # "low": ((108,), (111,), (119,)) freq 3
        # "new": ((110,), (101,), (119,)) freq 2
        l, o, w, n, e = (108,), (111,), (119,), (110,), (101,) # byte tokens
        word_token_counts = collections.Counter({
            (l, o, w): 3,
            (n, e, w): 2
        })
        pair_stats = get_byte_pair_stats(word_token_counts)
        # Expected pairs:
        # (l,o): 3
        # (o,w): 3
        # (n,e): 2
        # (e,w): 2
        self.assertEqual(pair_stats[(l,o)], 3)
        self.assertEqual(pair_stats[(o,w)], 3)
        self.assertEqual(pair_stats[(n,e)], 2)
        self.assertEqual(pair_stats[(e,w)], 2)
        self.assertEqual(len(pair_stats), 4)


    def test_merge_byte_pair(self):
        l, o, w, n, e = (108,), (111,), (119,), (110,), (101,)
        word_token_counts_in = collections.Counter({
            (l, o, w): 3,
            (n, e, w): 2
        })
        # Merge (o, w) -> (111, 119) which is token ow_token
        ow_token = (111, 119) 
        target_pair = (o, w)
        word_token_counts_out = merge_byte_pair(target_pair, word_token_counts_in)
        
        expected_low_merged_key = (l, ow_token) # ((108,), (111,119))
        self.assertEqual(word_token_counts_out[expected_low_merged_key], 3)
        
        original_new_key = (n, e, w) # ((110,), (101,), (119,))
        self.assertEqual(word_token_counts_out[original_new_key], 2)
        self.assertEqual(len(word_token_counts_out), 2)

    def test_train_and_tokenize_multibyte(self):
        vocab, rules = train_bbpe(self.corpus_multibyte, num_merges=10)
        self.assertTrue(len(vocab) > 0)
        self.assertTrue(len(rules) <= 10)

        # "你好" -> nihao_bytes = (228, 189, 160, 229, 165, 189)
        # If merges occur, this might be shorter.
        tokens_nihao = tokenize_word_with_bbpe("你好", rules)
        self.assertIsInstance(tokens_nihao, list)
        self.assertTrue(all(isinstance(token, tuple) for token in tokens_nihao))
        
        # Check if the sum of lengths of byte tokens equals original byte length
        original_bytes_len = len("你好".encode('utf-8'))
        tokenized_bytes_len = sum(len(token) for token in tokens_nihao)
        self.assertEqual(tokenized_bytes_len, original_bytes_len)


    def test_save_load_model(self):
        original_vocab, original_rules = train_bbpe(self.corpus_multibyte, num_merges=5)
        save_bbpe_model(original_vocab, original_rules, self.test_model_base)
        
        self.assertTrue(os.path.exists(self.test_model_base + ".bvocab"))
        self.assertTrue(os.path.exists(self.test_model_base + ".bmerges"))

        loaded_vocab, loaded_rules = load_bbpe_model(self.test_model_base)
        self.assertEqual(original_vocab, loaded_vocab)
        self.assertEqual(original_rules, loaded_rules)

        word = "世界" # shijie
        tokens_original = tokenize_word_with_bbpe(word, original_rules)
        tokens_loaded = tokenize_word_with_bbpe(word, loaded_rules)
        self.assertEqual(tokens_original, tokens_loaded)

    def test_empty_input_tokenize(self):
        _, rules = train_bbpe(self.corpus_ascii, num_merges=2)
        tokens = tokenize_word_with_bbpe("", rules)
        self.assertEqual(tokens, [])

    def test_oov_chars_tokenize(self):
        # Train on ASCII corpus
        _, rules = train_bbpe(self.corpus_ascii, num_merges=5)
        
        # Tokenize a word with characters not in training (e.g., multi-byte)
        # OOV characters will still be represented by their bytes.
        word_multibyte = "你好"
        tokens = tokenize_word_with_bbpe(word_multibyte, rules)
        
        # Expected: each byte of "你好" becomes a separate token if no relevant rules apply
        # Rules from ASCII corpus won't apply to these byte values.
        # So, it should be tokenized into its constituent single-byte tokens.
        expected_tokens = []
        for byte_val in word_multibyte.encode('utf-8'):
            expected_tokens.append((byte_val,))
        
        self.assertEqual(tokens, expected_tokens)

if __name__ == '__main__':
    unittest.main()
