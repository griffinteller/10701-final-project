import torch
import numpy as np
import tqdm

from torch import nn
from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainResult:
    train_loss: np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]]
    val_loss: np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]] | None


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader | None = None,
    verbose: bool = False,
    train_loss_reporting_beta: float = 0.99
) -> TrainResult:
    """
    Parameters
    ----------
    model : Module
        Model to train. Should take (input_sequences, target_sequences) and 
        return vector of loss over batch.
    optimizer : Optimizer
        Optimizer to use
    train_dl : DataLoader
        DataLoader for training data. Should return batches input sequences
        and batches of output sequences, both of token _ids_
    val_dl : DataLoader, optional
        DataLoader for validation. Validation metrics are collected at the end of 
        each epoch if provided. Default None.
    verbose: bool, optional
        Enable verbose logging. Default False.

    Returns
    -------
    TrainResult | None
        training results, or None if metric_steps = None.
    """

    train_losses = []
    val_losses = [] if val_dl is not None else None 

    for epoch in range(num_epochs):
        print(f"------------ EPOCH {epoch} ------------")

        train_loss_mva = 0

        for i, (inp, target) in enumerate(tqdm.tqdm(train_dl)):
            if verbose:
                print(f"====== Batch {i} =======")

            optimizer.zero_grad()

            loss = model(inp, target).mean()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            train_losses.append(loss)

            train_loss_mva = \
                (train_loss_reporting_beta * train_loss_mva \
                + (1 - train_loss_reporting_beta) * loss) \
                / (1 - train_loss_reporting_beta ** (i + 1))
            
            if verbose:
                print(f"Loss: {loss:.4f}")
                print(f"Loss MVA: {train_loss_mva:.4f}")

        if val_dl is not None:
            print(f"======== EVAL =======")

            with torch.no_grad():
                avg_loss = 0.0
                for i, (inp, target) in enumerate(tqdm.tqdm(val_dl)):
                    loss = model(inp, target).mean().item()
                    avg_loss += (loss - avg_loss) / (i + 1)

            val_losses.append(avg_loss) # type: ignore

            print(f"Train Loss MVA: {train_loss_mva:.4f}")
            print(f"Val Loss: {avg_loss:.4f}")

    return TrainResult(
        train_loss=np.array(train_losses),
        val_loss=np.array(val_losses)
    )
        