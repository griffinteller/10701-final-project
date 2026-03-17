import pickle
import numpy as np
from typing import Iterable


class VocabTokenizer:
    def __init__(self, vocab: Iterable[str]):
        self._ids = dict()
        self._words = dict()

        i = 0
        for word in vocab:
            self._ids[word.lower()] = i
            self._words[i] = word.lower()
            i += 1

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "VocabTokenizer":
        with open(path, "rb") as f:
            return pickle.load(f)
        
    def to_ids(self, text: str) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        words = text.split(" ")
        ids = np.empty(shape=(len(words),), dtype=np.int32)
        for i, word in enumerate(words):
            ids[i] = self._ids[word.lower()]
        return ids
    
    def from_ids(self, ids: Iterable[int]) -> str:
        words = []
        for i in ids:
            words.append(self._words[i])
        return " ".join(words)