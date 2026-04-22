import torch
import numpy as np
import tqdm
import argparse
import pandas as pd
import os
import sentencepiece as sp
import tqdm
import yaml
import wandb
import shutil

from torch import nn
from dataclasses import dataclass
from typing import Callable, Literal
import sacrebleu
from ssm import SSMTranslator, SSMTranslatorConfig
from lstm import LSTMTranslator, LSTMTranslatorConfig
from transformer import TransformerTranslator, TransformerTranslatorConfig


@dataclass
class TrainResult:
    train_loss: np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]]
    val_loss: np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]] | None

@dataclass
class TrainConfig:
    lr: float | dict
    num_epochs: int
    verbose: bool
    train_val_split: float
    batch_size: int
    seed: int
    eval_steps: int
    data_nrows: int | None = None

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
    wandb_run: wandb.Run,
    example_fn: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    start_epoch: int = 0,
    start_step: int = 0,
):
    """
    Parameters
    ----------
    model : Module
        Model to train. Should take (input_sequences, target_sequences) and 
        return mean loss over batch.
    train_dl : DataLoader
        DataLoader for training data. Should return batches input sequences
        and batches of output sequences, both of token _ids_.
    val_dl : DataLoader
        DataLoader for validation. Validation metrics are collected at the end of 
        each epoch.
    config : TrainConfig
        Training config.
    wandb_run : wandb.Run
        Wandb run to log metrics and artifacts to.
    example_fn: Callable[[torch.Tensor, torch.Tensor], None], optional
        Function called at the beginning of each epoch with a set of input and target ids
        (a batch of length 1). This can be used to print an example input, target, and output
        sentence.
    """

    model.train()

    best_val_loss = None
    step = wandb_run.step

    for epoch in range(start_epoch,config.num_epochs):
        print(f"------------ EPOCH {epoch} ------------")

        # make this deterministic upon resuming from a checkpoint
        torch.manual_seed(config.seed + epoch + 1)

        # example train and val output
        if config.verbose and example_fn is not None:
            model.eval()

            with torch.no_grad():
                for inp, target in train_dl:
                    example_fn(inp[:1].to(device), target[:1].to(device))
                    break

                for inp, target in val_dl:
                    example_fn(inp[:1].to(device), target[:1].to(device))
                    break
            
            model.train()

        train_iter = iter(train_dl)
        if start_step > 0:
            print(f"Skipping {start_step} steps to resume from checkpoint...")
            for i in tqdm.tqdm(range(start_step)):
                next(train_iter)

        for i, (inp, target) in enumerate(tqdm.tqdm(train_iter, initial=start_step, total=len(train_dl))):
            step += 1

            try:
                inp = inp.to(device)
                target = target.to(device)

                if config.verbose:
                    print(f"====== Batch {i} =======")

                optimizer.zero_grad()

                loss = model(inp, target)

                loss.backward()
                optimizer.step()

                if scheduler is not None \
                    and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            except Exception as e:
                print(f"Error training batch {i}: {e}")
                print("Input batch shape:", inp.shape)
                print("Skipping this batch...")
                continue

            loss = loss.item()
            wandb_run.log({"train_loss": loss}, step=step)
            if scheduler is not None:
                wandb_run.log({"lr": scheduler.get_last_lr()[0]}, step=step)

            if config.verbose:
                print(f"Loss: {loss:.4f}")

            if step % config.eval_steps == 0:
                print(f"======== EVAL =======")

                model.eval()

                # example train and val output
                try:
                    if example_fn is not None:
                        with torch.no_grad():
                            for inp, target in train_dl:
                                example_fn(inp[:1].to(device), target[:1].to(device))
                                break

                            for inp, target in val_dl:
                                example_fn(inp[:1].to(device), target[:1].to(device))
                                break
                except Exception as e:
                    print(f"Error in example_fn: {e}")
                    print("Skipping example_fn for this evaluation step...")

                with torch.no_grad():
                    avg_loss = 0.0
                    for i, (inp, target) in enumerate(tqdm.tqdm(val_dl)):
                        try:
                            inp = inp.to(device)
                            target = target.to(device)
                            loss = model(inp, target).item()
                            avg_loss += (loss - avg_loss) / (i + 1)
                        except Exception as e:
                            print(f"Error evaluating batch {i}: {e}")
                            print("Skipping this batch...")

                wandb_run.log({"val_loss": avg_loss}, step=step)

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)

                if config.verbose:
                    print(f"Validation loss: {avg_loss:.4f}")

                if best_val_loss is None or avg_loss < best_val_loss:
                    print("Saving new best model...")

                    best_val_loss = avg_loss

                    os.makedirs("temp/", exist_ok=True)
                    torch.save(model.state_dict(), f"temp/best_model.pt")
                    torch.save(optimizer.state_dict(), f"temp/best_optimizer.pt")

                    artifact = wandb.Artifact(artifact_name(wandb_run), type="model")
                    artifact.add_file("temp/best_model.pt")
                    artifact.add_file("temp/best_optimizer.pt")
                    artifact.metadata = {
                        "epoch": epoch,
                        "step": step,
                        "wandb_step": wandb_run.step,
                    }

                    wandb_run.log_artifact(artifact)
                
                model.train()

def artifact_name(wandb_run: wandb.Run) -> str:
    return f"{wandb_run.id}_best.pt"


class EnFrTokenizedDataset(torch.utils.data.Dataset):
    """
    EN -> FR translation dataset. Yields a tensor of input ids (en) and target ids (fr).
    Both EN and FR id tensors begin with the `bos_id` of the tokenizer, and end with `sos_id`.
    """

    def __init__(self, df: pd.DataFrame, tok_en: sp.SentencePieceProcessor, tok_fr: sp.SentencePieceProcessor, max_toks: int = 256):
        self.df = df

        self.proc_en = tok_en
        self.proc_fr = tok_fr
        self.max_toks = max_toks

        self.en_pad_id = tok_en.pad_id()
        self.fr_pad_id = tok_fr.pad_id()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text_en = row.at["en"]
        text_fr = row.at["fr"]

        if not isinstance(text_en, str) or not isinstance(text_fr, str):
            return \
                torch.tensor([self.proc_en.bos_id(), self.proc_en.eos_id()]), \
                torch.tensor([self.proc_fr.bos_id(), self.proc_fr.eos_id()])

        ids_en = self.proc_en.Encode(text_en, add_bos=True, add_eos=True)
        ids_fr = self.proc_fr.Encode(text_fr, add_bos=True, add_eos=True)

        return torch.tensor(ids_en), torch.tensor(ids_fr)
    
    def collate(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of token ids into a padded tensor. Left justifies, 
        and appends `pad_id` to the end of shorter token lists.

        Parameters
        ------
        batch: list[tuple[torch.Tensor, torch.Tensor]]
            list of (en, fr) token id tensors

        Returns
        -------
        torch.Tensor, torch.Tensor
        """
        en_ = [x for x, _ in batch]
        fr_ = [y for _, y in batch]

        en = [x for i, x in enumerate(en_) if x.shape[0] + fr_[i].shape[0] <= self.max_toks]
        fr = [y for i, y in enumerate(fr_) if en_[i].shape[0] + y.shape[0] <= self.max_toks]

        if len(en) == 0:
            print("Warning: all sequences in batch are too long. Returning truncated singleton batch...")
            en = [en_[0][:self.max_toks // 2]]
            fr = [fr_[0][:self.max_toks - en_[0].shape[0]]]

        en = nn.utils.rnn.pad_sequence(en, batch_first=True, padding_value=self.en_pad_id)
        fr = nn.utils.rnn.pad_sequence(fr, batch_first=True, padding_value=self.fr_pad_id)

        return en, fr
    

class SSMTranslatorTrainer(nn.Module):
    """
    Wrapper around SSMTranslator for training. Uses teacher forcing to decode
    FR sequence, and returns mean cross entropy loss across batches.
    """

    def __init__(self, ssm_translator: SSMTranslator):
        super().__init__()

        self.module = ssm_translator
    
    def forward(
        self,
        inp: torch.Tensor, 
        target: torch.Tensor,
        decode_method: Literal["ag", "forced"] = "forced",
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        logits = self.module(
            inp_ids=inp,
            decode_method=decode_method,
            forcing_ids=target[:, :-1],
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id
        )

        return nn.functional.cross_entropy(
            torch.flatten(logits, end_dim=-2), 
            torch.flatten(target[:, 1:]),
            ignore_index=pad_id,
            reduction="mean"
        )
    
class LSTMTranslatorTrainer(nn.Module):
    """
    Wrapper around LSTMTranslator for training. Uses teacher forcing to decode
    FR sequence, and returns mean cross entropy loss across batches.
    """

    def __init__(self, lstm_translator: LSTMTranslator):
        super().__init__()
        self.module = lstm_translator
    
    def forward(
        self,
        inp: torch.Tensor, 
        target: torch.Tensor,
        decode_method: Literal["ag", "forced"] = "forced",
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        logits = self.module(
            inp_ids=inp,
            decode_method=decode_method,
            forcing_ids=target[:, :-1],
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id
        )

        return nn.functional.cross_entropy(
            torch.flatten(logits, end_dim=-2), 
            torch.flatten(target[:, 1:]),
            ignore_index=pad_id,
            reduction="mean"
        )
    
class TransformerTranslatorTrainer(nn.Module):
    """
    Wrapper around TransformerTranslator for training. Uses teacher forcing to decode
    FR sequence, and returns mean cross entropy loss across batches.
    """

    def __init__(self, transformer_translator: TransformerTranslator):
        super().__init__()
        self.module = transformer_translator
    
    def forward(
        self,
        inp: torch.Tensor, 
        target: torch.Tensor,
        decode_method: Literal["ag", "forced"] = "forced",
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> torch.Tensor:
        logits = self.module(
            inp_ids=inp,
            decode_method=decode_method,
            forcing_ids=target[:, :-1],
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id
        )

        return nn.functional.cross_entropy(
            torch.flatten(logits, end_dim=-2), 
            torch.flatten(target[:, 1:]),
            ignore_index=pad_id,
            reduction="mean"
        )

def preprocess(args):
    """
    Preprocess data. Currently, just shuffles (deterministically) and splits
    data into train and test sets, then saves to data/train.csv and data/test.csv.
    """

    import os
    import sentencepiece as sp

    project_dir = os.path.join(os.path.dirname(__file__), "..")

    print("Reading csv...")
    df = pd.read_csv(os.path.join(project_dir, "data/en-fr.csv"), engine="pyarrow")
    nrows = len(df)
    print(f"nrows: {nrows}")

    print("Shuffling...")
    df = df.sample(frac=1, random_state=1)

    print("Splitting...")
    train, test = df.iloc[:int(nrows * 0.9)], df.iloc[int(nrows * 0.9):]

    train.to_csv(os.path.join(project_dir, "data/train.csv"), index=False)
    test.to_csv(os.path.join(project_dir, "data/test.csv"), index=False)

def strip_special(ids, pad_id=0, bos_id=1, eos_id=2):
    out = []
    for t in ids:
        if t == bos_id or t == pad_id:
            continue
        if t == eos_id:
            break
        out.append(t)
    return out

if __name__ == "__main__":
    # ensure wandb data doesn't fill up /home device
    os.environ["WANDB_DATA_DIR"] = "./.wandb_data"
    os.environ["WANDB_CACHE_DIR"] = "./.wandb_cache"

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    preprocess_parser = subparser.add_parser("preprocess")

    train_parser = subparser.add_parser("train")
    train_parser.add_argument("--train_config", type=str)
    train_parser.add_argument("--model_config", type=str)
    train_parser.add_argument("--model", type=str)
    train_parser.add_argument("--run_id", type=str)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--start_from", type=str, default=None)

    eval_parser = subparser.add_parser("evaluate")
    eval_parser.add_argument("--model_config", type=str)
    eval_parser.add_argument("--model", type=str)
    eval_parser.add_argument("--run_id", type=str)
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--data_nrows", type=int, default=None)
    eval_parser.add_argument("--num_examples", type=int, default=1)

    args = parser.parse_args()


    if args.command == "preprocess":
        preprocess(args)


    elif args.command == "train":
        print("Loading training config...")
        with open(args.train_config) as f:
            train_config_dict = yaml.safe_load(f)

        train_config = TrainConfig(**train_config_dict)
        print(train_config)

        torch.manual_seed(train_config.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        print("Loading model config...")
        with open(args.model_config) as f:
            model_config_dict = yaml.safe_load(f)


        print("Creating tokenizers...")
        tok_en = sp.SentencePieceProcessor()
        tok_en.Load(os.path.join(os.path.dirname(__file__), "../vocab/en.model"))

        tok_fr = sp.SentencePieceProcessor()
        tok_fr.Load(os.path.join(os.path.dirname(__file__), "../vocab/fr.model"))


        print("Creating model and optimizer...")
        if args.model == "ssm":
            model_config = SSMTranslatorConfig(**model_config_dict)
            ssm_model = SSMTranslator(model_config)
            model = SSMTranslatorTrainer(ssm_model).to(device)

        elif args.model == "lstm":
            model_config = LSTMTranslatorConfig(**model_config_dict)
            lstm_model = LSTMTranslator(model_config)
            model = LSTMTranslatorTrainer(lstm_model).to(device)


        elif args.model == "transformer":
            model_config = TransformerTranslatorConfig(**model_config_dict)
            transformer_model = TransformerTranslator(model_config)
            model = TransformerTranslatorTrainer(transformer_model).to(device)
                
        else:
            raise RuntimeError(f"Unknown model type {args.model}")

        total_parameters = sum(param.numel() for param in model.parameters())
        print(f"Number of parameters: {total_parameters}")

        if isinstance(train_config.lr, float):
            scheduler = None
            optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
        else:
            if train_config.lr["type"] == "cosine":
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr["max_lr"])
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    eta_min=train_config.lr["min_lr"],
                    T_max=train_config.lr["t_max"]
                )

            elif train_config.lr["type"] == "exponential":
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr["max_lr"])
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=train_config.lr["gamma"]
                )

            elif train_config.lr["type"] == "step":
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr["max_lr"])
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=train_config.lr["step_size"],
                    gamma=train_config.lr["gamma"]
                )

            elif train_config.lr["type"] == "adaptive":
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr["max_lr"])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=train_config.lr["factor"],
                    patience=train_config.lr["patience"],
                )

            else:
                raise RuntimeError(f"Unsupported lr scheduler type {train_config.lr['type']}")


        def example_fn(inp, target):
            logits = model.module(
                inp_ids=inp,
                decode_method="forced",
                forcing_ids=target[:, :-1],
            )

            input_string = tok_en.Decode(inp.tolist())
            target_string = tok_fr.Decode(target.tolist())
            output_string = tok_fr.Decode(logits.argmax(dim=-1).tolist())

            print(f"Input string: {input_string}")
            print(f"Target string: {target_string}")
            print(f"Output string: {output_string}")

        print("Setting up wandb run...")
        wandb_run = wandb.init(
            entity="gteller-cmu",
            project="en-fr-10701",
            id=args.run_id,
            name=args.run_id,
            resume="must" if args.resume else "never",
            config={
                "model": args.model,
                "num_parameters": total_parameters,
                "train_config": train_config_dict,
                "model_config": model_config_dict,
            }
        )

        assert not (args.resume and args.start_from is not None), \
            "Cannot specify both --resume and --start_from"

        start_epoch = 0
        start_step = 0
        start_wandb_step = 0
        if wandb_run.resumed:
            try:
                artifact = wandb_run.use_artifact(artifact_name(wandb_run) + ":latest")
                if os.path.exists("temp/"):
                    shutil.rmtree("temp/")

                path = artifact.download("temp/")
                model.load_state_dict(torch.load("temp/best_model.pt", map_location=device))
                optimizer.load_state_dict(torch.load("temp/best_optimizer.pt", map_location=device))

                start_epoch = artifact.metadata["epoch"]
                start_step = artifact.metadata["step"]
                start_wandb_step = artifact.metadata["wandb_step"]
            
            except Exception as e:
                print(f"Error loading artifact: {e}")
                print("Starting run from beginning...")

            print(f"Resumed run from step {wandb_run.step} (epoch {start_epoch}, step {start_step} in epoch)")

        if args.start_from is not None:
            print(f"Loading artifact from {args.start_from}...")

            artifact = wandb_run.use_artifact(args.start_from)
            if os.path.exists("temp/"):
                shutil.rmtree("temp/")

            path = artifact.download("temp/")
            model.load_state_dict(torch.load("temp/best_model.pt", map_location=device))
            optimizer.load_state_dict(torch.load("temp/best_optimizer.pt", map_location=device))

            start_epoch = artifact.metadata["epoch"]
            start_step = artifact.metadata["step"]
            start_wandb_step = artifact.metadata["wandb_step"]

        print("Reading data...")
        data_path = os.path.join(os.path.dirname(__file__), "../data/train.csv")
        if train_config.data_nrows is None:
            df = pd.read_csv(data_path, engine="pyarrow")
        else:
            df = pd.read_csv(data_path, nrows=train_config.data_nrows)


        print("Creating datasets...")
        split_idx = int(train_config.train_val_split * len(df))
        train_dataset = EnFrTokenizedDataset(df.iloc[:split_idx], tok_en, tok_fr)
        val_dataset = EnFrTokenizedDataset(df.iloc[split_idx:], tok_en, tok_fr)


        print("Creating dataloaders...")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_config.batch_size,
            shuffle=True,
            # num_workers=2,
            collate_fn=train_dataset.collate,
            pin_memory=True if device.type == "cuda" else False
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=train_config.batch_size,
            shuffle=True,
            # num_workers=2,
            collate_fn=val_dataset.collate,
            pin_memory=True if device.type == "cuda" else False
        )

        train(
            model, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            val_dataloader, 
            device, 
            train_config, 
            wandb_run, 
            example_fn=example_fn,
            start_epoch=start_epoch,
            start_step=start_step
        )

    elif args.command == "evaluate":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        print("Loading model config...")
        with open(args.model_config) as f:
            model_config_dict = yaml.safe_load(f)

        print("Creating tokenizers...")
        tok_en = sp.SentencePieceProcessor()
        tok_en.Load(os.path.join(os.path.dirname(__file__), "../vocab/en.model"))

        tok_fr = sp.SentencePieceProcessor()
        tok_fr.Load(os.path.join(os.path.dirname(__file__), "../vocab/fr.model"))

        print("Creating model...")
        if args.model == "ssm":
            model_config = SSMTranslatorConfig(**model_config_dict)
            base_model = SSMTranslator(model_config)
            model = SSMTranslatorTrainer(base_model).to(device)

        elif args.model == "lstm":
            model_config = LSTMTranslatorConfig(**model_config_dict)
            base_model = LSTMTranslator(model_config)
            model = LSTMTranslatorTrainer(base_model).to(device)

        elif args.model == "transformer":
            model_config = TransformerTranslatorConfig(**model_config_dict)
            base_model = TransformerTranslator(model_config)
            model = TransformerTranslatorTrainer(base_model).to(device)

        else:
            raise RuntimeError(f"Unknown model type {args.model}")

        print("Loading artifact...")
        wandb_run = wandb.init(
            entity="gteller-cmu",
            project="en-fr-10701",
            id=args.run_id,
            name=args.run_id,
            resume="must",
        )
        artifact = wandb_run.use_artifact(artifact_name(wandb_run) + ":latest")

        if os.path.exists("temp/"):
            shutil.rmtree("temp/")
        os.makedirs("temp/", exist_ok=True)

        artifact.download("temp/")
        model.load_state_dict(torch.load("temp/best_model.pt", map_location=device))
        model.eval()

        print("Reading data...")
        train_path = os.path.join(os.path.dirname(__file__), "../data/train.csv")
        test_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")

        if args.data_nrows is None:
            train_df = pd.read_csv(train_path, engine="pyarrow")
            test_df = pd.read_csv(test_path, engine="pyarrow")
        else:
            train_df = pd.read_csv(train_path, nrows=args.data_nrows)
            test_df = pd.read_csv(test_path, nrows=args.data_nrows)

        print("Creating datasets...")
        train_dataset = EnFrTokenizedDataset(train_df, tok_en, tok_fr)
        test_dataset = EnFrTokenizedDataset(test_df, tok_en, tok_fr)

        print("Creating dataloaders...")
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=train_dataset.collate,
            pin_memory=True if device.type == "cuda" else False
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate,
            pin_memory=True if device.type == "cuda" else False
        )

        model.eval()
        results = {}

        example_idx = 0

        with torch.inference_mode():
            inp_ex, target_ex = test_dataset[example_idx]
            inp_ex = inp_ex.unsqueeze(0).to(device)
            target_ex = target_ex.unsqueeze(0).to(device)

            logits_ex = model.module(
                inp_ids=inp_ex,
                decode_method="ag",
                max_output_len=min(target_ex.shape[1], 256),
                pad_id=tok_fr.pad_id(),
                bos_id=tok_fr.bos_id(),
                eos_id=tok_fr.eos_id(),
            )

            pred_ids_ex = logits_ex.argmax(dim=-1)

            inp_ids = strip_special(
                inp_ex[0].tolist(),
                pad_id=tok_en.pad_id(),
                bos_id=tok_en.bos_id(),
                eos_id=tok_en.eos_id(),
            )
            target_ids = strip_special(
                target_ex[0].tolist(),
                pad_id=tok_fr.pad_id(),
                bos_id=tok_fr.bos_id(),
                eos_id=tok_fr.eos_id(),
            )
            pred_ids_0 = strip_special(
                pred_ids_ex[0].tolist(),
                pad_id=tok_fr.pad_id(),
                bos_id=tok_fr.bos_id(),
                eos_id=tok_fr.eos_id(),
            )

            inp_str = tok_en.Decode(inp_ids)
            target_str = tok_fr.Decode(target_ids)
            pred_str = tok_fr.Decode(pred_ids_0)

            print(f"\nExample index: {example_idx}")
            print(f"Input:  {inp_str}")
            print(f"Target: {target_str}")
            print(f"Pred:   {pred_str}")

        for split, dl in [("test", test_dataloader)]:
            references = []  
            candidates = []  

            print(f"\n{split} BLEU")

            with torch.inference_mode():
                for inp, target in tqdm.tqdm(dl):
                    inp = inp.to(device)
                    target = target.to(device)

                    max_output_len = min(target.shape[1], 256)

                    logits = model.module(
                        inp_ids=inp,
                        decode_method="ag",
                        max_output_len=max_output_len,
                        pad_id=tok_fr.pad_id(),
                        bos_id=tok_fr.bos_id(),
                        eos_id=tok_fr.eos_id(),
                    )

                    pred_ids = logits.argmax(dim=-1)

                    for ref, cand in zip(target.tolist(), pred_ids.tolist()):
                        ref = strip_special(
                            ref,
                            pad_id=tok_fr.pad_id(),
                            bos_id=tok_fr.bos_id(),
                            eos_id=tok_fr.eos_id(),
                        )
                        cand = strip_special(
                            cand,
                            pad_id=tok_fr.pad_id(),
                            bos_id=tok_fr.bos_id(),
                            eos_id=tok_fr.eos_id(),
                        )

                        ref_txt = tok_fr.Decode(ref)
                        cand_txt = tok_fr.Decode(cand)

                        references.append(ref_txt)   
                        candidates.append(cand_txt)

                    del logits, pred_ids, inp, target
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            score = sacrebleu.corpus_bleu(candidates, [references]).score
            print(f"\n{split} BLEU: {score:.2f}")
            results[split] = score
            wandb_run.summary[f"{split}_bleu"] = score

        # print("\nEval Results:")
        # for split, score in results.items():
        #     print(f"\n{split} BLEU {score:.2f}")

        wandb_run.finish()