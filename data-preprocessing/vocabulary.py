"""Vocabulary management — standalone copy of dataset/Dictionary.py logic.

Used by the preprocessing pipeline so it can run independently of the
main training code (e.g. on a CPU-only node without the full project on
PYTHONPATH).
"""
import pickle


class Dictionary:
    """Bidirectional word ↔ index vocabulary.

    Pre-populated special tokens:
        [MASK]     = 0   (padding / ignored tokens)
        [UNKNOWN]  = 1   (OOD events never seen in training)
        [START]    = 2   (beginning of every request sequence)
        [END]      = 3   (end of every request sequence)
        [TRUNCATE] = 4   (last token when a sequence is cut at max_token)
    """

    def __init__(self, file_path: str = None):
        if file_path is None:
            self.word2idx: dict = {
                "[MASK]": 0,
                "[UNKNOWN]": 1,
                "[START]": 2,
                "[END]": 3,
                "[TRUNCATE]": 4,
            }
            self.idx2word: list = [
                "[MASK]",
                "[UNKNOWN]",
                "[START]",
                "[END]",
                "[TRUNCATE]",
            ]
        else:
            with open(file_path, "rb") as fh:
                self.word2idx, self.idx2word = pickle.load(fh)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_idx(self, word: str) -> int:
        """Return the index for *word*, or 1 ([UNKNOWN]) if not in vocab."""
        return self.word2idx.get(word, 1)

    def add_word(self, word: str) -> int:
        """Add *word* to the vocabulary (no-op if already present).

        Returns the index of *word*.
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """Pickle the vocabulary to *file_path*."""
        with open(file_path, "wb") as fh:
            pickle.dump((self.word2idx, self.idx2word), fh)

    def __len__(self) -> int:
        return len(self.idx2word)

    def __repr__(self) -> str:
        return f"Dictionary(size={len(self)})"
