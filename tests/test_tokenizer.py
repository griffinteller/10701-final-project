import pytest

from src.tokenizer import VocabTokenizer

def test_tokenizer():
    vocab = ["The", "quick", "brown", "fox"]
    tokenizer = VocabTokenizer(vocab)

    S = "The quick fox brown"
    ids = tokenizer.to_ids(S)
    assert tokenizer.from_ids(ids) == S.lower()
    
    with pytest.raises(Exception):
        tokenizer.to_ids("The quick brown fox jumped")
