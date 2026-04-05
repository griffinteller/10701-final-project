import pandas as pd
import sentencepiece as sp
import argparse
import os
import io


if __name__ == "__main__":
    num_sentences = 2_000_000

    print("Loading training data...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/train.csv"), engine="pyarrow")

    print("Shuffling data...")
    df = df.sample(n=num_sentences, random_state=42)

    def reader(lang):
        for row in df.itertuples():
            if not isinstance(row.en, str) or not isinstance(row.fr, str):
                continue
            
            if lang == "en":
                yield row.en
            elif lang == "fr":
                yield row.fr

    sp.SentencePieceTrainer.Train(
        sentence_iterator=reader("en"),
        model_prefix="en",
        vocab_size=32_000,  # inline with bert, mamba2 defaults
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        character_coverage=1.0,
    )

    sp.SentencePieceTrainer.Train(
        sentence_iterator=reader("fr"),
        model_prefix="fr",
        vocab_size=32_000,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        character_coverage=1.0,
    )

