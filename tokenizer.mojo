"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
"""A minimal byte-pair-encoding (BPE) tokenizer: train, encode, decode, save, load.

WHAT A TOKENIZER DOES

A language model works on integers, not text. A tokenizer converts a string
into a list of integer IDs (encode) and converts the IDs back into the
string (decode). The mapping between pieces of text and IDs is a fixed table
called the vocabulary, built once from a training corpus and then frozen.

HOW THE VOCABULARY IS BUILT (BPE)

1. Start with one token per distinct character found in the corpus.
2. Count every pair of adjacent tokens across the corpus.
3. Take the most frequent pair, join it into one new token, and add that
   token to the vocabulary. Remember the rule ("a" + "b" -> "ab").
4. Repeat from step 2 until the vocabulary reaches the requested size.

Frequent character sequences such as "the" or "ing" end up as single
tokens. Rare words stay split into smaller pieces.

THE THREE DATA STRUCTURES

    vocab   List[String]                    ID -> token. The ID is the index.
                                            Index 0 is "<UNK>", then one entry
                                            per character, then one entry per
                                            merge in the order it was learned.
    stoi    Dict[String, Int]               token -> ID (the reverse of vocab).
    merges  Dict[(String, String), String]  (left, right) -> joined token.
                                            Insertion order is the order the
                                            rules were learned, and encoding
                                            depends on that order.

THE PRE-TOKENIZER

Before BPE sees any text, PreTokenizer cuts it into words. Merges only
happen inside a word, never across two words, so "the cat" can never become
one token. A space is not discarded: it is written as the marker "Ġ" at the
start of the word that follows it, so "hello world" becomes the words
"hello" and "Ġworld". Decoding turns "Ġ" back into a space.

ENCODING

Pre-tokenize, split each word into characters, then apply the learned merge
rules in the order they were learned. Each resulting token is looked up in
the vocabulary. A token that is not in the vocabulary gets ID 0 ("<UNK>").

LIMITATIONS (deliberate, to keep the code short)

- The base alphabet is characters, not bytes. A character that never
  appeared in the training corpus becomes "<UNK>" and cannot be recovered.
- The pre-tokenizer is crude: it handles spaces and periods only.
- Encoding checks every merge rule against every word, which is slow.
- When two pairs have the same count, the one that comes first in the
  dictionary's iteration order wins. The rule is simple but it is not
  written down as a guarantee anywhere.
- The file format is this tokenizer's own JSON layout.
"""

from std.pathlib import Path
from std.python import Python

# A "Char" is a one-character String. The alias only makes type annotations
# easier to read: List[Char] is a list of single characters.
comptime Char = String


struct PreTokenizer:
    """Cuts text into words before BPE runs. Holds no state."""

    @staticmethod
    def split[
        spacer: StaticString = "Ġ",
    ](var text: String) raises -> List[String]:
        """Split text into words, marking each space with a prefix on the next word.

        Rules, applied in this order:
        1. Every space becomes a separator followed by the marker "Ġ", so the
           marker ends up attached to the front of the following word.
        2. Every "." gets a separator in front of it, so a period becomes
           its own word.
        3. The text is cut at the separators.

        Example:

            "hello world."  ->  ["hello", "Ġworld", "."]

        The first word has no marker because no space came before it. The
        marker lets decode() restore each space exactly.

        Args:
            text: The text to split.

        Returns:
            The words, in order. A text that starts with a space produces an
            empty string as its first word; later steps ignore it because it
            has no characters.
        """
        var splits = (
            StringSlice(text)
            # " " -> " Ġ": keep a space as the cut point and add the marker after it.
            .replace(" ", " " + spacer)
            # "." -> " .": add a cut point before every period.
            .replace(".", " .")
            # Cut at every space. The separators vanish; the marker stays.
            .split(" ")
        )
        # The pieces are string views into the text above. Copy each one into
        # its own String so the result owns its data.
        var result = List[String](capacity=len(splits))
        for split in splits:
            result.append(String(from_utf8=split.as_bytes()))
        return result^


struct BPETokenizer(Sized & Movable):
    """A BPE tokenizer. Call train() or load() before encode() or decode()."""

    # ID -> token. Position in the list is the ID.
    var vocab: List[String]
    # token -> ID. Built alongside vocab.
    var stoi: Dict[String, Int]
    # (left, right) -> joined. Insertion order is the order rules were learned.
    var merges: Dict[Tuple[String, String], String]

    def __init__(out self):
        """Create an empty tokenizer. It must be trained or loaded before use."""
        self.vocab = List[String]()
        self.stoi = Dict[String, Int]()
        self.merges = Dict[Tuple[String, String], String]()

    def train(mut self, corpus: List[String], vocab_size: Int) raises:
        """Learn the vocabulary and merge rules from a list of texts.

        Calling train() again replaces everything learned before.

        Args:
            corpus: The training texts. Each string is pre-tokenized separately.
            vocab_size: The target number of vocabulary entries, counting
                "<UNK>" and the single characters. If it is not larger than
                1 + the number of distinct characters, no merges are learned.
                Training also stops early if no pair is left to merge.

        Example: training on ["hello world"] gives the words "hello" and
        "Ġworld". The characters, sorted, are d e h l o r w Ġ, so the starting
        vocabulary has 9 entries: "<UNK>" at 0, d=1, e=2, h=3, l=4, o=5, r=6,
        w=7, Ġ=8. Every pair occurs once, so the first one counted ("h", "e")
        is merged first and "he" becomes ID 9.
        """
        # Step 1. Pre-tokenize and count words.
        # word_freqs maps each distinct word to how many times it occurs in
        # the corpus. Training works on distinct words with a count, which is
        # much cheaper than walking the full text on every round.
        var word_freqs = Dict[String, Int]()
        for text in corpus:
            var words = PreTokenizer.split(text)
            for word in words:
                word_freqs[word] = 1 + word_freqs.get(word, 0)

        # Step 2. Collect every distinct character that appears in any word.
        # codepoints() yields Unicode code points, so "é" is one character.
        # Sorting makes the ID assignment independent of the order in which
        # characters were first seen.
        var alphabet: List[Char] = []
        for word in word_freqs.keys():
            for cp in word.codepoints():
                var char = chr(Int(cp))
                if char not in alphabet:
                    alphabet.append(char)
        sort(alphabet)

        # Step 3. Start the vocabulary: "<UNK>" at ID 0, then each character.
        # "<UNK>" stands for any token that is not in the vocabulary.
        self.vocab = List[String](capacity=vocab_size)
        self.vocab.append(String("<UNK>"))
        self.stoi = Dict[String, Int]()
        self.stoi["<UNK>"] = 0
        for i, char in enumerate(alphabet):
            self.vocab.append(char)
            self.stoi[char] = i + 1

        # Step 4. Write each distinct word as a list of one-character tokens.
        # This is the state the merge loop edits: after a merge, the words that
        # contained the pair hold one token where they held two.
        var splits = Dict[String, List[Char]]()
        for word in word_freqs.keys():
            splits[word] = [chr(Int(cp)) for cp in word.codepoints()]

        # Step 5. The merge loop. Each pass adds exactly one vocabulary entry.
        self.merges = Dict[Tuple[String, String], String]()
        while len(self.vocab) < vocab_size:
            # Count every adjacent pair across all words, weighted by how often
            # each word occurs.
            var pair_freqs = _compute_pair_freqs(splits, word_freqs)
            # No pairs means every word is already a single token.
            if len(pair_freqs) == 0:
                break
            # Find the most frequent pair. The comparison is strict (<), so on a
            # tie the pair seen first in iteration order stays the winner.
            # max_freq == -1 only matters on the first pair: it accepts it.
            var best_pair: Tuple[String, String] = ("", "")
            var max_freq = -1
            for pair_freq in pair_freqs.items():
                var pair = pair_freq.key
                var freq = pair_freq.value
                if max_freq == -1 or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            # Rewrite every word so the winning pair becomes one token.
            _merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            # Record the new token. Appending to vocab gives it the next free
            # ID, and inserting into merges keeps the rules in learned order.
            var joined = best_pair[0].copy() + best_pair[1].copy()
            self.merges[best_pair] = joined
            self.vocab.append(joined)
            self.stoi[joined] = len(self.vocab) - 1

    def _tokenize(self, text: String) raises -> List[String]:
        """Turn text into a flat list of tokens (strings, not IDs).

        Replays the learned merge rules, in the order they were learned, on
        the characters of each word.

        The order matters. A later rule often joins a token that an earlier
        rule created: with the rules ("l", "o") then ("lo", "w"), the word
        "low" becomes "lo" + "w" and then "low". Applying them in the other
        order would find no ("lo", "w") pair at all.

        Cost: every rule is checked against every word, so the work grows as
        (number of rules) x (number of characters). That is easy to follow
        and slow on large inputs.
        """
        if text.byte_length() == 0:
            return List[String]()
        # Same word split as in training. The words must be cut the same way,
        # or the learned rules would not line up with the pieces.
        var words = PreTokenizer.tokenize(text)
        # One list of single-character tokens per word.
        var splits = [
            [chr(Int(cp)) for cp in word.codepoints()]
            for word in words
        ]
        # merges.items() yields rules in the order they were inserted, which
        # is the order they were learned.
        for pair_merge in self.merges.items():
            ref pair = pair_merge.key
            ref merge = pair_merge.value
            for idx, split in enumerate(splits):
                var i = 0
                var split_copied = split.copy()
                # Walk the word left to right looking for the rule's pair.
                while i < len(split_copied) - 1:
                    if split_copied[i] == pair[0] and split_copied[i + 1] == pair[1]:
                        # Replace the two tokens with the joined token: keep
                        # everything before i, insert the joined token, keep
                        # everything after i + 1. i is not advanced; the
                        # joined token sits at i and is compared with its new
                        # right neighbour on the next pass.
                        split_copied = (
                            [e for e in split_copied[:i]]
                            + [merge]
                            + [e for e in split_copied[i + 2 :]]
                        )
                    else:
                        i += 1
                splits[idx] = split_copied^
        # Join the per-word lists into one flat list of tokens.
        return [item for sublist in splits for item in sublist]

    def encode(self, text: String) raises -> List[Int]:
        """Convert text to a list of token IDs.

        Any token that is not in the vocabulary maps to ID 0 ("<UNK>"). In
        this tokenizer that happens for a character the training corpus never
        contained, and the original character cannot be recovered from the ID.
        """
        var tokens = self._tokenize(text)
        var ids = List[Int](capacity=len(tokens))
        for token in tokens:
            # get(token, 0) returns 0 when the token is missing.
            ids.append(self.stoi.get(token, 0))
        return ids^

    def decode(self, ids: List[Int]) raises -> String:
        """Convert token IDs back to text.

        Looks up each ID, joins the tokens with nothing between them, and turns
        every "Ġ" back into a space. An ID of 0 comes back as the literal text
        "<UNK>". An ID outside the vocabulary is an error.
        """
        if len(ids) == 0:
            return String("")
        # Concatenate the token for every ID.
        var raw = StringSlice("").join([self.vocab[i] for i in ids])
        # Undo the pre-tokenizer's space marker.
        return String(StringSlice(raw).replace("Ġ", " "))

    def __len__(self) -> Int:
        """The vocabulary size, including "<UNK>" and the single characters."""
        return len(self.vocab)

    def save(self, path: String) raises:
        """Write the tokenizer to a JSON file.

        The file holds two things:
            "vocab":  the token list; the position of each token is its ID.
            "merges": a list of [left, right, joined] triples in learned order.
        That is enough to rebuild all three data structures, so stoi is not
        stored. The JSON writing is done by Python's json module.
        """
        var json = Python.import_module("json")
        var data = Python.dict()

        # Convert the vocabulary to a Python list of strings.
        var py_vocab = Python.list()
        for token in self.vocab:
            py_vocab.append(Python.str(String(token)))
        data["vocab"] = py_vocab

        # Convert each rule to a [left, right, joined] list. The list keeps
        # the learned order, which load() relies on.
        var py_merges = Python.list()
        for merge in self.merges.items():
            var entry = Python.list()
            entry.append(Python.str(String(merge.key[0])))
            entry.append(Python.str(String(merge.key[1])))
            entry.append(Python.str(String(merge.value)))
            py_merges.append(entry)
        data["merges"] = py_merges

        Path(path).write_text(String(json.dumps(data)))

    @staticmethod
    def load(path: String) raises -> Self:
        """Read a tokenizer written by save() and return it.

        Rebuilds vocab and stoi from the saved token list (the list position
        is the ID) and rebuilds merges from the saved triples. Because the
        triples are inserted in file order, the rules keep their learned order.
        """
        var json = Python.import_module("json")
        var data = json.loads(Path(path).read_text())

        var tok = Self()

        # Rebuild vocab and its reverse table together.
        var py_vocab = data["vocab"]
        tok.vocab = List[String](capacity=len(py_vocab))
        for i in range(len(py_vocab)):
            var token = String(py_vocab[i])
            tok.vocab.append(token)
            tok.stoi[token] = i

        # Rebuild the merge rules in file order.
        var py_merges = data["merges"]
        tok.merges = Dict[Tuple[String, String], String]()
        for i in range(len(py_merges)):
            var entry = py_merges[i]
            tok.merges[(String(entry[0]), String(entry[1]))] = String(entry[2])

        return tok^


def _compute_pair_freqs(
    splits: Dict[String, List[String]], word_freqs: Dict[String, Int]
) raises -> Dict[Tuple[String, String], Int]:
    """Count every adjacent pair of tokens across all words.

    Each occurrence of a word contributes to its pairs, so a pair inside a
    word that appears 3 times is counted 3 times. For example, if "Ġlow"
    occurs 3 times and is currently split as [Ġ, l, o, w], then (Ġ, l),
    (l, o) and (o, w) each gain 3.

    Args:
        splits: Each distinct word mapped to its current list of tokens.
        word_freqs: Each distinct word mapped to its number of occurrences.

    Returns:
        A table from (left, right) to its total count.
    """
    var pair_freqs = Dict[Tuple[String, String], Int]()
    for word_freq in word_freqs.items():
        var word = word_freq.key
        var freq = word_freq.value
        ref split = splits[word]
        # A word reduced to one token has no pairs left.
        if len(split) == 1:
            continue
        # Pair each token with the one to its right.
        for i in range(len(split) - 1):
            var pair = (split[i], split[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs^


def _merge_pair(
    a: String,
    b: String,
    mut splits: Dict[String, List[String]],
    word_freqs: Dict[String, Int],
) raises:
    """Replace every adjacent (a, b) in every word with the single token a + b.

    Scans each word left to right, so matches do not overlap. In [a, a, a]
    with the pair (a, a), the first two tokens merge and the third stays.

    Args:
        a: The left token of the pair.
        b: The right token of the pair.
        splits: Each word's token list. Edited in place.
        word_freqs: Used only for the list of distinct words to visit.
    """
    for word in word_freqs:
        ref split = splits[word]
        if len(split) == 1:
            continue
        var i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Rebuild the list: everything before i, the joined token,
                # everything after i + 1. The merged token is at i, so i stays
                # put. The joined token never equals a, so it cannot match
                # again at the same position, and the next pass moves on.
                split = (
                    [e for e in split[:i]]
                    + [a + b]
                    + [e for e in split[i + 2 :]]
                )
            else:
                i += 1
        # Store the updated list back under the word.
        splits[word] = split.copy()
