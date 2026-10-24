# llm-code

自学大模型时写的代码，每个算法尽量用最少的依赖从零实现，方便读懂原理。

## 目录

```
.
├── tokenizer/                 # 分词算法（只用 Python 标准库）
│   ├── BPE.py                 # 字符级 BPE（Byte Pair Encoding）
│   ├── BBPE.py                # 字节级 BPE（GPT-2 系列使用）
│   ├── WordPiece.py           # WordPiece（BERT 使用）
│   ├── ULM.py                 # Unigram Language Model（SentencePiece 使用）
│   ├── compare_tokenizers.py  # 对比四种分词器在同一句子上的结果
│   └── tests/                 # 单元测试
└── rlhf/                      # 强化学习 / RLHF 相关
    └── ppo.py                 # PPO（RLHF 的核心算法），在 CartPole 上演示
```

## 分词器

| 算法 | 思路 | 典型模型 |
|---|---|---|
| BPE | 反复合并出现次数最多的相邻符号对 | GPT-1、早期机器翻译模型 |
| BBPE | 同 BPE，但在 UTF-8 字节上做，不会出现未知字符 | GPT-2 / GPT-3、RoBERTa、LLaMA 3 |
| WordPiece | 合并能让语料似然提升最多的符号对 | BERT |
| ULM | 先给一个大词表，再用 EM + Viterbi 逐步删掉贡献最小的子词 | T5、ALBERT（SentencePiece） |

运行（在 `tokenizer/` 目录下）：

```bash
cd tokenizer
python BPE.py                  # 训练并演示，模型文件写在当前目录
python BBPE.py
python WordPiece.py
python ULM.py
python compare_tokenizers.py   # 需要先运行上面四个脚本生成模型文件
python -m unittest discover -s tests
```

## PPO

`rlhf/ppo.py` 实现了离散动作空间的 PPO（Actor-Critic + clip 目标），用 CartPole-v1 演示，CPU 上几分钟能训练到平均 475 分以上。

```bash
pip install -r requirements.txt
python rlhf/ppo.py
```
