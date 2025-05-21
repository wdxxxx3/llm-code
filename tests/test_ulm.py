import unittest
import os
import sys
import collections
import math

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ULM import (
    train_ulm, tokenize_word_with_ulm, save_ulm_model, load_ulm_model,
    get_initial_candidates, viterbi_segment, UNK_TOKEN
)

class TestULM(unittest.TestCase):

    def setUp(self):
        self.corpus = "this is a simple corpus. simple words for simple testing."
        self.corpus_multibyte = "你好 世界 你好 世界 hello"
        self.test_model_base = os.path.join(project_root, "tests", "test_ulm_model_files", "test_ulm_model")

        os.makedirs(os.path.dirname(self.test_model_base), exist_ok=True)
        if os.path.exists(self.test_model_base + ".ulm_model"):
            os.remove(self.test_model_base + ".ulm_model")

    def tearDown(self):
        if os.path.exists(self.test_model_base + ".ulm_model"):
            os.remove(self.test_model_base + ".ulm_model")
        try:
            os.rmdir(os.path.dirname(self.test_model_base))
        except OSError:
            pass

    def test_get_initial_candidates(self):
        candidates = get_initial_candidates("test test a", max_subword_len=3, min_freq=1)
        self.assertIn('t', candidates)
        self.assertIn('e', candidates)
        self.assertIn('s', candidates)
        self.assertIn('a', candidates)
        self.assertIn('te', candidates)
        self.assertIn('es', candidates)
        self.assertIn('st', candidates)
        self.assertIn('tes', candidates)
        self.assertIn('est', candidates)
        self.assertEqual(candidates['t'], 4) # t in test (2) * 2
        self.assertEqual(candidates['test'], 2)
        self.assertEqual(candidates['a'], 1)


    def test_viterbi_segment(self):
        # Vocab: a, b, ab, c, bc. Log probs for simplicity (scores)
        vocab_probs = {
            'a': math.log(0.4), 'b': math.log(0.3), 'c': math.log(0.1),
            'ab': math.log(0.5), # higher prob for 'ab' than 'a'+'b'
            'bc': math.log(0.4)  # higher prob for 'bc' than 'b'+'c'
        }
        max_len = 2

        # Test 1: "ab"
        tokens, score = viterbi_segment(list("ab"), vocab_probs, max_len)
        self.assertEqual(tokens, ['ab'])
        self.assertAlmostEqual(score, math.log(0.5))

        # Test 2: "abc" -> "ab", "c"
        tokens, score = viterbi_segment(list("abc"), vocab_probs, max_len)
        self.assertEqual(tokens, ['ab', 'c'])
        self.assertAlmostEqual(score, math.log(0.5) + math.log(0.1))
        
        # Test 3: "aabc" -> "a", "a", "bc" (assuming "aa" not in vocab)
        tokens, score = viterbi_segment(list("aabc"), vocab_probs, max_len)
        self.assertEqual(tokens, ['a', 'a', 'bc'])
        self.assertAlmostEqual(score, math.log(0.4) + math.log(0.4) + math.log(0.4)) # a, a, bc

        # Test 4: Unsegmentable
        vocab_simple = {'a': -1.0, 'b': -1.0, UNK_TOKEN: -100.0}
        tokens, score = viterbi_segment(list("xyz"), vocab_simple, 1)
        self.assertEqual(tokens, [UNK_TOKEN]) # Should fall back to UNK_TOKEN
        self.assertEqual(score, vocab_simple[UNK_TOKEN])


    def test_train_and_tokenize(self):
        # Training ULM can be less deterministic in terms of exact vocab items due to EM nature.
        # Focus on: runs, vocab size is reasonable, UNK is present, basic tokenization works.
        target_size = 30 # small target vocab for test
        vocab_probs = train_ulm(self.corpus, target_size, num_iterations=2, 
                                initial_seed_vocab_size_factor=2, 
                                max_subword_len_candidates=5, min_freq_candidates=1)
        
        self.assertIn(UNK_TOKEN, vocab_probs)
        self.assertTrue(len(vocab_probs) <= target_size + len(set(c for w in self.corpus.split() for c in w)) ) # Approx target size
        
        # Check if single characters from corpus are in vocab
        self.assertIn('s', vocab_probs)
        self.assertIn('i', vocab_probs)

        max_len_trained_vocab = max(len(sw) for sw in vocab_probs.keys()) if vocab_probs else 0

        tokens_simple = tokenize_word_with_ulm("simple", vocab_probs, max_len_trained_vocab)
        self.assertIsInstance(tokens_simple, list)
        self.assertTrue(len(tokens_simple) > 0)
        self.assertEqual("".join(tokens_simple), "simple") # Should perfectly reconstruct known words

        tokens_oov = tokenize_word_with_ulm("unknownxyz", vocab_probs, max_len_trained_vocab)
        self.assertTrue(any(UNK_TOKEN == t for t in tokens_oov) or "".join(tokens_oov) == "unknownxyz")


    def test_save_load_model(self):
        target_size = 25
        original_vocab_probs = train_ulm(self.corpus_multibyte, target_size, num_iterations=2,
                                         initial_seed_vocab_size_factor=2,
                                         max_subword_len_candidates=4, min_freq_candidates=1)
        
        save_ulm_model(original_vocab_probs, self.test_model_base)
        self.assertTrue(os.path.exists(self.test_model_base + ".ulm_model"))

        loaded_vocab_probs = load_ulm_model(self.test_model_base)
        self.assertIsNotNone(loaded_vocab_probs)
        # Compare keys and float values with tolerance
        self.assertEqual(set(original_vocab_probs.keys()), set(loaded_vocab_probs.keys()))
        for k in original_vocab_probs:
            self.assertAlmostEqual(original_vocab_probs[k], loaded_vocab_probs[k], places=5)

        word = "你好"
        max_len_orig = max(len(sw) for sw in original_vocab_probs.keys()) if original_vocab_probs else 0
        max_len_load = max(len(sw) for sw in loaded_vocab_probs.keys()) if loaded_vocab_probs else 0

        tokens_original = tokenize_word_with_ulm(word, original_vocab_probs, max_len_orig)
        tokens_loaded = tokenize_word_with_ulm(word, loaded_vocab_probs, max_len_load)
        self.assertEqual(tokens_original, tokens_loaded)

    def test_empty_input_tokenize(self):
        vocab_probs = {UNK_TOKEN: -20.0, 'a': -1.0}
        tokens = tokenize_word_with_ulm("", vocab_probs, 1)
        self.assertEqual(tokens, [])

if __name__ == '__main__':
    unittest.main()
