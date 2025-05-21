import unittest
import os
import sys
import collections

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from WordPiece import (
    train_wordpiece, tokenize_word_with_wordpiece, save_wordpiece_model, load_wordpiece_model,
    get_initial_vocab_and_word_counts, calculate_pair_scores, merge_best_pair,
    UNK_TOKEN, WORD_SUFFIX
)

class TestWordPiece(unittest.TestCase):

    def setUp(self):
        self.corpus = "this is a simple sentence. another simple sentence."
        self.initial_alphabet_chars = set("abcdefghijklmnopqrstuvwxyz.") # Explicitly define for tests
        self.test_model_base = os.path.join(project_root, "tests", "test_wp_model_files", "test_wp_model")
        
        os.makedirs(os.path.dirname(self.test_model_base), exist_ok=True)
        if os.path.exists(self.test_model_base + ".wp_vocab"):
            os.remove(self.test_model_base + ".wp_vocab")

    def tearDown(self):
        if os.path.exists(self.test_model_base + ".wp_vocab"):
            os.remove(self.test_model_base + ".wp_vocab")
        try:
            os.rmdir(os.path.dirname(self.test_model_base))
        except OSError:
            pass

    def test_get_initial_vocab_and_word_counts(self):
        # Corpus: "this is a simple sentence." (words: this, is, a, simple, sentence)
        # Alphabet: only 's', 'e', 'n', 't', 'c' for this test subset.
        small_corpus = "sentence sense"
        small_alpha = set("sentc")
        word_counts = get_initial_vocab_and_word_counts(small_corpus, small_alpha)
        
        # "sentence" -> s,e,n,t,e,n,c,e -> filter with small_alpha -> s,e,n,t,e,n,c,e
        # -> ('s','e','n','t','e','n','c','e', WORD_SUFFIX)
        expected_sentence_key = ('s','e','n','t','e','n','c','e', WORD_SUFFIX)
        self.assertEqual(word_counts[expected_sentence_key], 1)

        # "sense" -> s,e,n,s,e -> filter with small_alpha -> s,e,n,s,e
        # -> ('s','e','n','s','e', WORD_SUFFIX)
        expected_sense_key = ('s','e','n','s','e', WORD_SUFFIX)
        self.assertEqual(word_counts[expected_sense_key], 1)
        self.assertEqual(len(word_counts), 2)


    def test_calculate_pair_scores(self):
        # word_counts: {('s','e','n','t',WORD_SUFFIX): 5, ('s','e',WORD_SUFFIX):3}
        word_counts = collections.Counter({
            ('s','e','n','t',WORD_SUFFIX): 5, 
            ('s','e',WORD_SUFFIX):3
        })
        # Frequencies:
        # Pairs: (s,e): 8, (e,n): 5, (n,t): 5, (t,WD_SFX): 5, (e,WD_SFX):3
        # Tokens: s:8, e:8, n:5, t:5, WD_SFX:8
        pair_scores = calculate_pair_scores(word_counts, set(['s','e','n','t',WORD_SUFFIX]))
        
        # score(s,e) = 8 / (8*8) = 8/64 = 0.125
        self.assertAlmostEqual(pair_scores[('s','e')], 8 / (8*8))
        # score(e,n) = 5 / (8*5) = 5/40 = 0.125
        self.assertAlmostEqual(pair_scores[('e','n')], 5 / (8*5))
        # score(n,t) = 5 / (5*5) = 5/25 = 0.2
        self.assertAlmostEqual(pair_scores[('n','t')], 5 / (5*5))


    def test_merge_best_pair(self):
        word_counts_in = collections.Counter({
            ('s','e','n','t',WORD_SUFFIX): 5, 
            ('s','e',WORD_SUFFIX):3
        })
        # Assume ('n','t') is the best pair to merge into "nt"
        best_pair = ('n','t')
        word_counts_out = merge_best_pair(best_pair, word_counts_in)

        expected_key1 = ('s','e','nt',WORD_SUFFIX)
        self.assertEqual(word_counts_out[expected_key1], 5)
        
        expected_key2 = ('s','e',WORD_SUFFIX) # This one is not affected
        self.assertEqual(word_counts_out[expected_key2], 3)
        self.assertEqual(len(word_counts_out), 2)

    def test_train_and_tokenize(self):
        # Small target vocab for predictability
        # initial_alphabet_chars includes 's', 'i', 'm', 'p', 'l', 'e', 't', 'n', 'c', '.'
        # Corpus: "simple sentence. simple sentence."
        small_corpus = "simple sentence. simple sentence."
        small_alpha = set("simplet nc.")
        target_size = len(small_alpha) + 3 + 5 # Base chars + UNK + SUFFIX + few merges
        
        vocab_list, _ = train_wordpiece(small_corpus, target_size, initial_alphabet_chars_param=small_alpha)
        vocab_set = set(vocab_list)

        self.assertIn(UNK_TOKEN, vocab_set)
        self.assertIn(WORD_SUFFIX, vocab_set)
        for char_ in small_alpha:
            self.assertIn(char_, vocab_set)
        
        # Tokenize "simple"
        # Depends on merges. If "sim" and "ple" are learned:
        # "simple" + WORD_SUFFIX -> "simple</w>"
        # Possible tokenization: ["simple</w>"] or ["sim", "##ple</w>"]
        tokens_simple = tokenize_word_with_wordpiece("simple", vocab_set, small_alpha)
        self.assertIsInstance(tokens_simple, list)
        self.assertTrue(len(tokens_simple) > 0)
        if len(tokens_simple) > 1:
            self.assertTrue(tokens_simple[1].startswith("##"))
        
        # Tokenize word with OOV chars from outside small_alpha
        tokens_oov_char = tokenize_word_with_wordpiece("word", vocab_set, small_alpha)
        # 'w', 'o', 'r', 'd' are not in small_alpha.
        # Expected: [UNK_TOKEN, UNK_TOKEN, UNK_TOKEN, UNK_TOKEN+WORD_SUFFIX] or similar if UNK is per char
        # The current tokenizer consumes one char for UNK.
        # "word</w>" -> UNK (for w) + UNK (for o) + UNK (for r) + UNK (for d) + WORD_SUFFIX
        # The WORD_SUFFIX is appended to the word string *before* tokenization.
        # So "word</w>". First char 'w' is OOV -> UNK_TOKEN, remaining "ord</w>"
        # 'o' is OOV -> UNK_TOKEN, remaining "rd</w>"
        # 'r' is OOV -> UNK_TOKEN, remaining "d</w>"
        # 'd</w>' could be a token if 'd' was in alpha and 'd</w>' learned, or UNK+WORD_SUFFIX
        # If 'd' is OOV, then 'd' -> UNK_TOKEN, and '</w>' remains.
        # If '</w>' is a token (it is), then it's appended.
        # So, for "word", expecting [UNK_TOKEN, ##UNK_TOKEN, ##UNK_TOKEN, ##UNK_TOKEN, ##WORD_SUFFIX] if all chars are OOV
        # Or, if the tokenizer handles it as 4 UNKs and then the final token has suffix:
        # e.g. for "word": [UNK_TOKEN, ##UNK_TOKEN, ##UNK_TOKEN, ##UNK_TOKEN_WITH_SUFFIX]
        # Let's trace: word_string="word", current_word_segment="word</w>"
        # 'w' not in vocab_set, 'w' not in small_alpha -> UNK_TOKEN. subword_tokens=["[UNK]"], current_word_segment="ord</w>"
        # 'o' not in vocab_set, 'o' not in small_alpha -> UNK_TOKEN. subword_tokens=["[UNK]","[UNK]"], current_word_segment="rd</w>"
        # 'r' not in vocab_set, 'r' not in small_alpha -> UNK_TOKEN. subword_tokens=["[UNK]","[UNK]","[UNK]"], current_word_segment="d</w>"
        # 'd' not in vocab_set, 'd' not in small_alpha -> UNK_TOKEN. subword_tokens=["[UNK]","[UNK]","[UNK]","[UNK]"], current_word_segment="</w>"
        # '</w>' is in vocab_set. subword_tokens=["[UNK]","[UNK]","[UNK]","[UNK]","</w>"]. current_word_segment=""
        # Final formatting: ["[UNK]", "##[UNK]", "##[UNK]", "##[UNK]", "##</w>"]
        self.assertEqual(tokens_oov_char, [UNK_TOKEN, "##"+UNK_TOKEN, "##"+UNK_TOKEN, "##"+UNK_TOKEN, "##"+WORD_SUFFIX])


    def test_save_load_model(self):
        vocab_list_original, _ = train_wordpiece(self.corpus, 50, initial_alphabet_chars_param=self.initial_alphabet_chars) # Target 50
        save_wordpiece_model(vocab_list_original, self.test_model_base)
        
        self.assertTrue(os.path.exists(self.test_model_base + ".wp_vocab"))

        loaded_vocab_list = load_wordpiece_model(self.test_model_base)
        self.assertEqual(vocab_list_original, loaded_vocab_list)

        word = "sentence"
        vocab_set_original = set(vocab_list_original)
        vocab_set_loaded = set(loaded_vocab_list)
        
        tokens_original = tokenize_word_with_wordpiece(word, vocab_set_original, self.initial_alphabet_chars)
        tokens_loaded = tokenize_word_with_wordpiece(word, vocab_set_loaded, self.initial_alphabet_chars)
        self.assertEqual(tokens_original, tokens_loaded)

    def test_empty_input_tokenize(self):
        vocab_list, _ = train_wordpiece(self.corpus, 30, initial_alphabet_chars_param=self.initial_alphabet_chars)
        vocab_set = set(vocab_list)
        tokens = tokenize_word_with_wordpiece("", vocab_set, self.initial_alphabet_chars)
        self.assertEqual(tokens, [])


if __name__ == '__main__':
    unittest.main()
