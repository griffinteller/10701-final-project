import pytest
import sentencepiece as sp

def test_tokenizer():
    sentence = "The quick brown fox jumped over the lazy dogs."

    proc = sp.SentencePieceProcessor()
    proc.load("vocab/en.model")

    ids = proc.Encode(sentence, add_eos=True)
    toks = proc.Decode(ids)

    assert True
