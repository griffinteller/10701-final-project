import pandas as pd
import sentencepiece as sp
import argparse
import os
import io


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "../data/en-fr.csv")) as f:
        block_size = 1024
        num_sentences = 1_000_000
        
        def lazy_reader(lang):
            for i in range((num_sentences // block_size - 1) + 1):
                lines = []
                for j in range(block_size):
                    line = f.readline()
                    if line == "": break
                    lines.append(line.strip())

                df = pd.read_csv(io.StringIO("\n".join(lines)))
                for row in df.itertuples(index=True, name=None):
                    if lang == "en":
                        sen = row[1]
                    elif lang == "fr":
                        sen = row[2]
                    else:
                        raise RuntimeError(f"Invalid language {lang}")

                    if not isinstance(sen, str):
                        continue

                    yield sen

                if len(lines) < block_size:
                    break

        sp.SentencePieceTrainer.Train(
            sentence_iterator=lazy_reader("en"),
            model_prefix="en",
            vocab_size=20_000,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )

        sp.SentencePieceTrainer.Train(
            sentence_iterator=lazy_reader("fr"),
            model_prefix="fr",
            vocab_size=20_000,
            model_type="bpe",
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3
        )

