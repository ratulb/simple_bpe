# simple_bpe

A small byte-pair-encoding (BPE) tokenizer written in Mojo. It can train a vocabulary from a list of texts, encode text to token IDs, decode IDs back to text, and save and load the result.

**This project is for learning.** It follows the BPE implementation in the Hugging Face LLM Course, chapter 6, section 5: [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5). That section says its code "won't be an optimized version you can actually use on a big corpus" and exists so you can understand the algorithm. The same applies here. The code is short and every step is visible, and it is slow and incomplete on purpose. Do not use it to tokenize data for a real model.

`simple_bpe` is the starting point for the first chapter of the book *Tokens to Transformers in Mojo* ([chapter 1](https://ratulb.github.io/tokens-to-transformers/ch01/)). The companion project [`mbpe`](https://github.com/ratulb/mbpe) is a production-grade tokenizer that is compatible with tiktoken.

## What it does

A language model works on integers, not text. A tokenizer converts text to a list of integer IDs and back. BPE builds the table that maps pieces of text to IDs:

1. Split the training text into words and count how often each word occurs.
2. Start the vocabulary with every distinct character in those words.
3. Count every pair of adjacent tokens across all words, weighted by word frequency.
4. Take the most frequent pair, join it into one new token, and add it to the vocabulary. Remember this merge rule.
5. Repeat from step 3 until the vocabulary reaches the size you asked for.

To encode new text, split it into words, split each word into characters, and apply the merge rules in the order they were learned. Then look up each resulting token in the vocabulary.

### A small example

This is the example from the Hugging Face course, worked by hand. The corpus has five words with these counts:

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)
```

The starting vocabulary is the characters `b g h n p s u`. The pair `("u", "g")` occurs 20 times, more than any other, so the first merge rule is `("u", "g") -> "ug"`. After that, `("u", "n")` occurs 16 times and gives `"un"`, and then `("h", "ug")` gives `"hug"`. Each rule creates a token that later rules can build on, which is why encoding must apply the rules in the order they were learned.

## How this implementation differs from the course

The algorithm is the same: word frequencies, a character alphabet, a special token at the start of the vocabulary, pair counting, the first-seen pair winning ties, and merge rules replayed in order. A few things are different:

| | Hugging Face course | `simple_bpe` |
| --- | --- | --- |
| Language | Python | Mojo |
| Pre-tokenization | Uses the GPT-2 pre-tokenizer from `transformers` | A small built-in one (see below) |
| Special token | `<\|endoftext\|>` | `<UNK>` at ID 0 |
| Unknown character | Raises an error | Becomes `<UNK>` (ID 0) |
| Save and load | Not covered | JSON file |

Because the pre-tokenizer is simpler, the vocabulary learned from the course's example corpus will not necessarily match the course's output exactly.

### The pre-tokenizer

Merges only happen inside a word, so the text is cut into words first. This tokenizer does it with two rules:

- Each space is written as the marker `Ġ` at the start of the next word, so `"hello world"` becomes `["hello", "Ġworld"]`. Decoding turns `Ġ` back into a space. The marker is the one GPT-2 uses.
- Each period becomes its own word, so `"hello world."` becomes `["hello", "Ġworld", "."]`.

Nothing else is handled. Commas, other punctuation, digits, contractions and newlines stay attached to the words around them.

## Requirements

- [Mojo](https://www.modular.com/mojo) 1.1.0, installed through [pixi](https://pixi.sh)
- Python available to Mojo. Saving and loading use Python's `json` module through Mojo's Python interop.

## Run it

```bash
git clone https://github.com/ratulb/simple_bpe
cd simple_bpe
pixi install
pixi run mojo main.mojo
```

`main.mojo` runs six tests covering training, encoding, decoding, and saving and loading.

## Use it

```mojo
def main() raises:
    # Imports: see the top of main.mojo.
    var corpus: List[String] = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    var tok = BPETokenizer()
    tok.train(corpus, 50)              # target vocabulary size

    var ids = tok.encode("This is not a token.")
    print(tok.decode(ids))             # This is not a token.

    tok.save("tokenizer.json")
    var loaded = BPETokenizer.load("tokenizer.json")
```

`vocab_size` counts every entry: `<UNK>`, the single characters, and the merged tokens. Training stops early if no pair is left to merge.

## Files

| File | Contents |
| --- | --- |
| `tokenizer.mojo` | The tokenizer: `PreTokenizer`, `BPETokenizer`, and the two helper functions for counting and merging pairs. The file is commented line by line. |
| `main.mojo` | The tests. |

## Limitations

These are deliberate, to keep the code short enough to read in one sitting:

- **Characters, not bytes.** A character that did not appear in the training corpus becomes `<UNK>` and cannot be recovered. Production tokenizers such as the one in GPT-2 start from the 256 byte values instead, so nothing is ever unknown. The course calls this *byte-level BPE*.
- **A crude pre-tokenizer.** Only spaces and periods are handled.
- **Slow encoding.** Encoding checks every merge rule against every word, so the work grows with the number of rules times the length of the text.
- **Tie-breaking is not specified.** When two pairs have the same count, the one that comes first in the dictionary's iteration order is merged. The rule is simple, but it is not a guarantee, and a different implementation can learn a different vocabulary from the same corpus.
- **Training recounts everything.** Every merge recounts all pairs from scratch.
- **A private file format.** The saved JSON works only with this tokenizer. It is not the format used by tiktoken or Hugging Face.

The `mbpe` project addresses each of these.

## References

- [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5), Hugging Face LLM Course, chapter 6, section 5. The algorithm and the Python implementation this code follows.
- [Tokens to Transformers in Mojo, chapter 1](https://ratulb.github.io/tokens-to-transformers/ch01/). A walkthrough of this code and of `mbpe`.
- [`mbpe`](https://github.com/ratulb/mbpe). The production-grade tokenizer.
