#!/usr/bin/env python
import os
import string
import random
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn.functional as F
import nltk
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn, optim
from collections import Counter
from typing import Dict, List, Optional
from pathlib import Path

nltk.download('punkt')
nltk.download('punkt_tab')

MAX_LINE_LEN = 128

TRAIN_INPUTS_PATH = Path(__file__).parent.parent / "data/train/train_inputs.txt"
TRAIN_LABELS_PATH = Path(__file__).parent.parent / "data/train/train_labels.txt"
VAL_INPUTS_PATH = Path(__file__).parent.parent / "data/val/val_inputs.txt"
VAL_LABELS_PATH = Path(__file__).parent.parent / "data/val/val_labels.txt"
TEST_INPUTS_PATH = Path(__file__).parent.parent / "data/test/test_inputs.txt"
TEST_LABELS_PATH = Path(__file__).parent.parent / "data/test/test_labels.txt"

CHAR_TO_INDEX_FILE = "char_to_index.json"
INDEX_TO_CHAR_FILE = "index_to_char.json"

class CharLSTM(nn.Module):

    # input_dim: size of vocab
    # embedding_dim: representation for each char
    # hidden_dim: hidden state vector
    # output_dim: num prediction classes (output chars)
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, dropout=0.2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

def load_dataset(inputs_path, labels_path, char2idx):
    inputs = []
    labels = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            inputs.append(line[:-1])
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line[:-1])
    # convert strings to fixed length numeric vectors
    # TODO: re-normalize text. this was already done when making the dataset
    # but perhaps would be good to do it again here to be safe.
    X_list = [encode_text(text, char2idx, width=MAX_LINE_LEN) for text in inputs]
    X = torch.stack(X_list)
    y = torch.tensor([char2idx.get(c, 0) for c in labels])
    return TensorDataset(X, y)

def load_training_data():
    # get char vocab from training data, use this to encode all dataset splits
    char2idx, idx2char = fit_char_vocab(TRAIN_INPUTS_PATH)
    train_dataset = load_dataset(TRAIN_INPUTS_PATH, TRAIN_LABELS_PATH, char2idx)
    val_dataset = load_dataset(VAL_INPUTS_PATH, VAL_LABELS_PATH, char2idx)
    test_dataset = load_dataset(TEST_INPUTS_PATH, TEST_LABELS_PATH, char2idx)
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

def load_test_data(fname):
    # your code here
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            inp = line[:-1]  # the last character is a newline
            data.append(inp)
    return data

def write_pred(preds, fname):
    with open(fname, 'wt', encoding="utf-8") as f:
        for p in preds:
            f.write('{}\n'.format(p))

def run_pred(data, char2idx, idx2char):
    preds = []
    for inp in data:
        inp = clean_text(inp)
        x = encode_text(inp, char2idx, width=MAX_LINE_LEN)
        y_hat = model(x.unsqueeze(0))
        top_guesses = decode_vec(y_hat, idx2char)
        preds.append(''.join(top_guesses))
    return preds

def save(model, work_dir):
    ckpt_path = os.path.join(work_dir, 'model.checkpoint')
    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)

def load(model, work_dir):
    ckpt_path = os.path.join(work_dir, 'model.checkpoint')
    model.load_state_dict(torch.load(ckpt_path))
    return model

def run_train(model, train_dataset, val_dataset):
    EPOCHS = 50
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    train_results = train(
        train_loader,
        model,
        criterion,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        early_stop=True
    )

def encode_text(text: str, char2idx, width=None) -> torch.Tensor:
    if width is not None and len(text) > width:
        text = text[len(text) - width:]
    vec = torch.tensor([char2idx.get(c, 0) for c in text])
    if width is not None:
        return F.pad(vec, (width - len(vec), 0), value=0)
    return vec

def decode_vec(y_hat, idx2char):
    top3_chars = ""
    labels = torch.topk(y_hat, k=3, dim=1)
    for label in labels.indices[0]:
        next_char = idx2char[label.item()]
        top3_chars += next_char
    return top3_chars

def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    val_loader: Optional[DataLoader] = None,
    lr: float = 0.001,
    epochs: int = 100,
    early_stop = True,
    verbose: bool = True
) -> Dict[str, List[float]]:
    losses = {
        'train': [],
        'val': []
    }
    best_val_loss = float('inf')
    no_improvement = 0
    grace_period = 5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs):
        train_loss = 0
        val_loss = 0
        for x_train,y_train in train_loader:
            optimizer.zero_grad()
            y_hat = model(x_train)
            batch_loss = criterion(y_hat, y_train)
            train_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        losses['train'].append(train_loss / len(train_loader))

        if val_loader is None:
            continue
        with torch.no_grad():
            for x_val,y_val in val_loader:
                y_hat = model(x_val)
                batch_loss = criterion(y_hat, y_val)
                val_loss += batch_loss.item()
        losses['val'].append(val_loss / len(val_loader))
        if verbose:
            print(f"EPOCH {i}/{epochs}: Train loss: {losses['train'][-1]:.4f}, Val loss: {losses['val'][-1]:.4f}")
        if losses['val'][-1] < best_val_loss:
            best_val_loss = losses['val'][-1]
        else:
            no_improvement += 1
        if no_improvement > grace_period:
            print(f"Stoping early, detected {grace_period} epochs with no val loss improvement.")
            break

    return losses


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn = nn.CrossEntropyLoss(),
    device: str = "cpu",
) -> Dict[str, float]:
    def get_accuracy(y_pred, y_true):
        y_pred_ = y_pred.squeeze(-1)
        y_true_ = y_true.squeeze(-1)
        n_correct = (y_pred_ == y_true_).sum().item()
        return n_correct / y_true_.shape[0]

    def get_precision(y_pred, y_true, epsilon=1e-8):
        psum = 0
        num_classes = y_true.max() + 1
        for i in range(num_classes):
            tp = ((y_pred == i) & (y_true == i)).sum().item()
            fp = ((y_pred == i) & (y_true != i)).sum().item()
            precision = tp / (tp + fp + epsilon)
            psum += precision
        return (psum / num_classes).item()

    def get_recall(y_pred, y_true, epsilon=1e-8):
        rsum = 0
        num_classes = y_true.max() + 1
        for i in range(num_classes):
            tp = ((y_pred == i) & (y_true == i)).sum().item()
            fn = ((y_pred != i) & (y_true == i)).sum().item()
            recall = tp / (tp + fn + epsilon)
            rsum += recall
        return (rsum / num_classes).item()

    def get_f1_score(y_pred, y_true, epsilon=1e-8):
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        return f1

    model.eval()
    model.to(device)
    val_loss = 0.0
    y_val_list = []
    y_pred_list = []

    with torch.no_grad():
      for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        batch_loss = loss_fn(logits, y_batch)
        batch_preds = torch.argmax(logits, dim=1)
        y_pred_list.extend(batch_preds.cpu())
        y_val_list.extend(y_batch)
        val_loss += batch_loss.item()
    val_loss /= len(data_loader)
    y_pred = torch.stack(y_pred_list)
    y_val = torch.stack(y_val_list)

    accuracy = get_accuracy(y_pred, y_val)
    precision = get_precision(y_pred, y_val)
    recall = get_recall(y_pred, y_val)
    f1 = get_f1_score(y_pred, y_val)

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# helper functions for text processing
def fit_char_vocab(text_file_path: str):
    """
    Gets char-to-index and index-to-char mappings for all characters in the given text file.
    Writes the resulting indices to disk so that it can be used at inference time.
    """
    with open(text_file_path, "r", encoding="utf-8") as f:
        all_text = f.read()
    all_text = clean_text(all_text)
    char_set = sorted(list(set(all_text)))
    char2idx = {ch:i for i,ch in enumerate(char_set)}
    idx2char = {i:ch for i,ch in enumerate(char_set)}
    with open(os.path.join(args.work_dir, CHAR_TO_INDEX_FILE), "w", encoding="utf-8") as fp:
        json.dump(char2idx, fp)
    with open(os.path.join(args.work_dir, INDEX_TO_CHAR_FILE), "w", encoding="utf-8") as fp:
        json.dump(idx2char, fp)
    return char2idx, idx2char

def load_char_indices():
    char2idx_path = os.path.join(args.work_dir, CHAR_TO_INDEX_FILE)
    idx2char_path = os.path.join(args.work_dir, INDEX_TO_CHAR_FILE)
    if not (os.path.exists(char2idx_path) and os.path.exists(idx2char_path)):
        raise FileNotFoundError("Character index has not been loaded yet. Please run fit_char_vocab.")
    # Load character indices from disk
    with open(char2idx_path, "r", encoding="utf-8") as fp:
        char2idx = json.load(fp)
    with open(idx2char_path, "r", encoding="utf-8") as fp:
        idx2char = json.load(fp)
        idx2char = {int(k):v for k,v in idx2char.items()}
    return char2idx, idx2char

def clean_text(text):
    text = text.lower()
    punct_removal = str.maketrans({c:"" for c in set(string.punctuation) if c != "."})
    text = text.translate(punct_removal)
    text = text.replace('\n', ' ').replace('\r','').replace('\t', '')
    text = text.strip()
    text = " ".join(text.split())
    return text

# main
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    EMBED_DIM = 64
    HIDDEN_DIM = 32
    DROPOUT = 0.2

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Loading training data')
        dataset_dict = load_training_data()
        char2idx, idx2char = load_char_indices()
        print('Instatiating model')
        model = CharLSTM(input_dim=len(char2idx),
                        output_dim=len(char2idx),
                        embedding_dim=EMBED_DIM,
                        hidden_dim=HIDDEN_DIM,
                        dropout=DROPOUT)
        print('Training')
        run_train(model, dataset_dict["train"], dataset_dict["val"])
        print('Saving model')
        save(model, args.work_dir)
    elif args.mode == 'test':
        print('Loading char vocab')
        char2idx, idx2char = load_char_indices()
        print('Loading model')
        model = CharLSTM(input_dim=len(char2idx),
                        output_dim=len(char2idx),
                        embedding_dim=EMBED_DIM,
                        hidden_dim=HIDDEN_DIM,
                        dropout=DROPOUT)
        load(model, args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = load_test_data(args.test_data)
        print('Making predictions')
        pred = run_pred(test_data, char2idx=char2idx, idx2char=idx2char)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
