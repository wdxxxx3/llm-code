import collections
import re # 用于分词

# 特殊的单词结束符
EOW = '</w>'

def get_initial_vocab_and_word_counts(corpus_text):
    """
    1. 将原始文本分割成单词。
    2. 在每个单词末尾添加 EOW 符号。
    3. 统计处理后单词的频率。
    4. 构建初始字符词典。

    Args:
        corpus_text (str): 原始文本语料。

    Returns:
        tuple: (
            word_counts (collections.Counter): 单词频率，如 {'l o w </w>': 5, ...}
                                            注意：这里的key是空格分隔的字符序列
            initial_char_vocab (set): 初始的单个字符集合
        )
    """
    # 使用正则表达式进行简单的分词（按空格和标点），并转换为小写
    # \w+ 匹配单词字符，'[^A-Za-z0-9\s]+' 匹配非字母数字空格的标点作为分隔符
    # 或者更简单地，只按空格分割，然后处理标点（这里为了简化，我们假设单词已经比较干净）
    raw_words = re.findall(r'\b\w+\b', corpus_text.lower())

    # 为每个单词添加 EOW，并将其拆分为字符，用空格连接
    # 例如 "low" -> "l o w </w>"
    # 我们需要存储原始单词及其频率，但为了BPE操作，我们会将其表示为字符列表
    processed_words_for_counting = []
    for word in raw_words:
        processed_words_for_counting.append(" ".join(list(word)) + " " + EOW)

    word_counts = collections.Counter(processed_words_for_counting)

    # 构建初始字符词典
    initial_char_vocab = set()
    for word_chars_str in word_counts:
        initial_char_vocab.update(word_chars_str.split())

    print("--- 步骤 0: 预处理和初始词典 ---")
    print(f"原始词频 (以空格分隔的字符序列表示): {word_counts}")
    print(f"初始字符词典: {sorted(list(initial_char_vocab))}\n")
    return word_counts, initial_char_vocab


def get_pair_stats(word_counts):
    """
    统计所有单词中相邻符号对的频率。
    `word_counts` 是一个 Counter，key 是空格分隔的符号序列，value 是频率。
    例如: {'l o w </w>': 5, 'n e w e s t </w>': 6}

    Args:
        word_counts (collections.Counter): 当前单词（表示为符号序列）及其频率。

    Returns:
        collections.Counter: 相邻符号对的频率，如 {('l', 'o'): 7, ('e', 's'): 9}
    """
    pair_stats = collections.defaultdict(int)
    for word_chars_str, freq in word_counts.items():
        symbols = word_chars_str.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_stats[pair] += freq # 乘以该单词的频率
    return pair_stats


def merge_pair(target_pair, word_counts_in):
    """
    在所有单词中合并指定的符号对。
    例如，如果 target_pair = ('e', 's')，则 "n e w e s t </w>" 会变成 "n e w es t </w>"。

    Args:
        target_pair (tuple): 要合并的符号对，如 ('e', 's')。
        word_counts_in (collections.Counter): 输入的单词频率。

    Returns:
        collections.Counter: 合并后的新的单词频率。
    """
    word_counts_out = collections.defaultdict(int)
    bigram = re.escape(' '.join(target_pair)) #  ('e', 's') -> 'e s'
    replacement = ''.join(target_pair)       #  ('e', 's') -> 'es'

    for word_chars_str, freq in word_counts_in.items():
        # 使用正则表达式替换，确保只替换独立的对（通过空格分隔）
        new_word_chars_str = re.sub(r'(?<!\S)' + bigram + r'(?!\S)', replacement, word_chars_str)
        # (?<!\S) 前面不是非空白字符（即前面是空白或行首）
        # (?!\S)  后面不是非空白字符（即后面是空白或行尾）
        # 上面的正则可能过于严格，对于 't e s t' 和 'es'，它不会匹配中间的 'e s'
        # 更简单的方式是直接在 split 后的 list 上操作，但 re.sub 更快
        # 这里我们用更简单的方式，因为 key 本身就是 'symbol1 symbol2 ...'
        # 直接用字符串替换
        new_word_chars_str = word_chars_str.replace(' '.join(target_pair), replacement)
        word_counts_out[new_word_chars_str] += freq

    return word_counts_out


def train_bpe(corpus_text, num_merges):
    """
    训练BPE模型。

    Args:
        corpus_text (str): 原始文本语料。
        num_merges (int): 要执行的合并操作次数。

    Returns:
        tuple: (
            final_vocab (set): 最终的词典（包含单字符和合并后的子词）。
            merge_rules (list): 按学习顺序列出的合并规则 (pair_to_merge)。
        )
    """
    # 1. 初始准备
    word_counts, current_vocab = get_initial_vocab_and_word_counts(corpus_text)
    # `word_counts` 的 key 是 'c h a r1 c h a r2 ... </w>' 形式

    merge_rules = [] # 存储合并规则，顺序很重要

    print("--- 开始 BPE 训练迭代 ---\n")
    for i in range(num_merges):
        print(f"--- 合并迭代 {i + 1}/{num_merges} ---")

        # 2. 统计相邻符号对频率
        pair_stats = get_pair_stats(word_counts)
        # print(f"当前符号对频率: {pair_stats}")

        if not pair_stats:
            print("没有更多可合并的符号对，提前停止。")
            break # 如果没有可合并的对了，就停止

        # 3. 找出频率最高的符号对
        # max(iterable, key=function)
        # pair_stats.get 会返回对应 key 的 value，作为排序依据
        best_pair = max(pair_stats, key=pair_stats.get)
        best_pair_freq = pair_stats[best_pair]
        print(f"最高频对: {best_pair} (频率: {best_pair_freq})")

        # 4. 合并该符号对并更新词典
        merged_symbol = "".join(best_pair)
        current_vocab.add(merged_symbol)
        merge_rules.append(best_pair) # 记录这条规则

        # 5. 更新语料库中的表示
        word_counts = merge_pair(best_pair, word_counts)
        print(f"合并后，新符号 '{merged_symbol}' 加入词典。")
        # print(f"更新后的词频表示: {word_counts}")
        # print(f"当前词典大小: {len(current_vocab)}, 词典: {sorted(list(current_vocab))}\n")
        print("-" * 30)


    print("\n--- BPE 训练完成 ---")
    print(f"最终词典大小: {len(current_vocab)}")
    print(f"最终词典 (部分): {sorted(list(current_vocab))[:20]} ...") # 只显示一部分
    print(f"学习到的合并规则 (共 {len(merge_rules)} 条):")
    for idx, rule in enumerate(merge_rules):
        print(f"  {idx+1}. {rule} -> {''.join(rule)}")

    return current_vocab, merge_rules


def tokenize_word_with_bpe(word_string, merge_rules):
    """
    使用学习到的BPE合并规则来对单个新词进行分词。

    Args:
        word_string (str): 要分词的单词（不含 </w>）。
        merge_rules (list): 按学习顺序列出的合并规则 (pair_to_merge)。

    Returns:
        list: 分词后的子词列表。
    """
    if not word_string:
        return []

    # 1. 预处理：拆分为字符，并添加 EOW
    tokens = list(word_string) + [EOW]
    # print(f"初始 tokens: {tokens}")

    # 2. 迭代应用合并规则 (按学习顺序)
    #    对于每个规则，我们扫描整个当前 token 序列，应用该规则
    for pair_to_merge in merge_rules:
        new_tokens = []
        i = 0
        while i < len(tokens):
            # 检查当前位置和下一个位置是否能形成要合并的对
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_merge:
                new_tokens.append("".join(pair_to_merge)) # 合并
                i += 2 # 跳过两个已处理的 token
            else:
                new_tokens.append(tokens[i]) # 不合并，直接添加
                i += 1
        tokens = new_tokens # 更新 token 序列以备下一条规则使用
        # print(f"应用规则 {pair_to_merge} -> {''.join(pair_to_merge)} 后: {tokens}")

    return tokens


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
    final_vocab, learned_merge_rules = train_bpe(corpus, num_merges)

    print("\n--- 使用BPE进行分词测试 ---")
    test_words = ["lowest", "newer", "widely", "huggingface", "unknownword", "a", "new", "algorithm", "face"]
    for word in test_words:
        tokenized_output = tokenize_word_with_bpe(word, learned_merge_rules)
        print(f"单词 '{word}' -> 分词结果: {tokenized_output}")

    # 检查一些特殊的token
    print("\n--- 检查词典和规则 ---")
    print(f"最终词典中是否有 'est</w>'? : {'est</w>' in final_vocab}")
    print(f"最终词典中是否有 'low'? : {'low' in final_vocab}")
    print(f"最终词典中是否有 'hugg'? : {'hugg' in final_vocab}")

    # 看看如果合并次数很少会怎样
    print("\n--- 测试较少合并次数 (例如 5 次) ---")
    _, few_rules = train_bpe(corpus, 5)
    for word in ["lowest", "newer"]:
        tokenized_output = tokenize_word_with_bpe(word, few_rules)
        print(f"单词 '{word}' (5次合并) -> 分词结果: {tokenized_output}")