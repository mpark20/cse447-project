#!/usr/bin/env python
import os
import string
import random
import json
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch import nn, optim
from typing import Dict, List, Optional
from pathlib import Path

# we only consider the previous 10 characters when training and making predictions,
MAX_LINE_LEN = 10

DATA_DIR = "data_prev_20"
TRAIN_INPUTS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/train/train_inputs.txt"
TRAIN_LABELS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/train/train_labels.txt"
VAL_INPUTS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/val/val_inputs.txt"
VAL_LABELS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/val/val_labels.txt"
TEST_INPUTS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/test/test_inputs.txt"
TEST_LABELS_PATH = Path(__file__).parent.parent / f"{DATA_DIR}/test/test_labels.txt"

CHAR_TO_INDEX_FILE = "char_to_index.json"
INDEX_TO_CHAR_FILE = "index_to_char.json"

PAD_TOKEN = 0

# results of hyperparameter search
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.005
EMBED_DIM = 32
HIDDEN_DIM = 64
DROPOUT = 0.2

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

def load_dataset(inputs_path, labels_path, char2idx, max_length=MAX_LINE_LEN):
    inputs = []
    labels = []
    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            inputs.append(line[:-1])
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line[:-1])
    # convert strings to fixed length numeric vectors
    X_list = [encode_text(text, char2idx, width=max_length) for text in inputs]
    X = torch.stack(X_list)
    y = torch.tensor([char2idx.get(c, PAD_TOKEN) for c in labels])
    return TensorDataset(X, y)

def load_training_data(max_length=MAX_LINE_LEN):
    # get char vocab from training data, use this to encode all dataset splits
    char2idx, idx2char = fit_char_vocab(TRAIN_INPUTS_PATH)
    train_dataset = load_dataset(TRAIN_INPUTS_PATH, TRAIN_LABELS_PATH, char2idx, max_length=max_length)
    val_dataset = load_dataset(VAL_INPUTS_PATH, VAL_LABELS_PATH, char2idx, max_length=max_length)
    test_dataset = load_dataset(TEST_INPUTS_PATH, TEST_LABELS_PATH, char2idx, max_length=max_length)
    
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

def run_pred(data, char2idx, idx2char, max_length=MAX_LINE_LEN):
    preds = []
    for inp in data:
        inp = clean_text(inp)
        x = encode_text(inp, char2idx, width=max_length)
        y_hat = model(x.unsqueeze(0))
        top_guesses = decode_vec(y_hat, idx2char)
        preds.append(''.join(top_guesses))
    return preds

def save(model, work_dir, filename='model.checkpoint'):
    ckpt_path = os.path.join(work_dir, filename)
    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)

def load(model, work_dir, filename='model.checkpoint'):
    # note to self: since this only saves the weights, we need to be
    # careful that we initialize the model with the same architecture
    # as the saved model.
    ckpt_path = os.path.join(work_dir, filename)
    model.load_state_dict(torch.load(ckpt_path))
    return model

def run_train(model, train_dataset, val_dataset, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    train_results = train(
        train_loader,
        model,
        criterion,
        val_loader=val_loader,
        lr=lr,
        epochs=epochs,
        early_stop=True
    )

def hparam_search(train_dataset, val_dataset, char2idx, n_iter=10):
    epochs = 5
    bsizes = [32, 64, 128, 256]
    lrs = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    embed_dims = [16, 32, 64, 128]
    hidden_dims = [16, 32, 64, 128]
    dropouts = [0, 0.2, 0.4, 0.6, 0.8]
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    for _ in range(n_iter):
        results_dict = {
            "lr": random.choice(lrs),
            "batch_size": random.choice(bsizes),
            "embed_dim": random.choice(embed_dims),
            "hidden_dim": random.choice(hidden_dims),
            "dropout": random.choice(dropouts),
        }
        model = CharLSTM(
            input_dim=len(char2idx),
            output_dim=len(char2idx),
            embedding_dim=results_dict['embed_dim'],
            hidden_dim=results_dict['hidden_dim'],
            dropout=results_dict['dropout']
        )
        train_loader = DataLoader(train_dataset, batch_size=results_dict['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=results_dict['batch_size'], shuffle=True)
        train_results = train(
            train_loader,
            model,
            criterion,
            val_loader=val_loader,
            lr=results_dict['lr'],
            epochs=epochs,
            early_stop=True
        )
        curr_train_loss = min(train_results["train"])
        curr_val_loss = min(train_results["val"])
        results_dict["train_loss"] = curr_train_loss
        results_dict["val_loss"] = curr_val_loss
        print(json.dumps(results_dict, indent=2))
        print("-----")

        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            best_hparams = results_dict
    return best_hparams

def encode_text(text: str, char2idx, width=None) -> torch.Tensor:
    if width is not None and len(text) > width:
        text = text[len(text) - width:]
    vec = torch.tensor([char2idx.get(c, PAD_TOKEN) for c in text])
    if width is not None:
        return F.pad(vec, (width - len(vec), PAD_TOKEN), value=0)
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
    lr: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    early_stop = True,
    verbose: bool = True
) -> Dict[str, List[float]]:
    start_time = time.time()
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
        if early_stop and no_improvement > grace_period:
            print(f"Stoping early, detected {grace_period} epochs with no val loss improvement.")
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time}")
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
def fit_char_vocab(text_file_path: str, remove_punct=True):
    """
    Gets char-to-index and index-to-char mappings for all characters in the given text file.
    Writes the resulting indices to disk so that it can be used at inference time.
    """
    with open(text_file_path, "r", encoding="utf-8") as f:
        all_text = f.read()
    all_text = clean_text(all_text, remove_punct=remove_punct)
    char_set = sorted(list(set(all_text)))
    char2idx = {ch:i for i,ch in enumerate(char_set)}
    idx2char = {i:ch for i,ch in enumerate(char_set)}
    with open(os.path.join(args.work_dir, CHAR_TO_INDEX_FILE), "w", encoding="utf-8") as fp:
        json.dump(char2idx, fp)
    with open(os.path.join(args.work_dir, INDEX_TO_CHAR_FILE), "w", encoding="utf-8") as fp:
        json.dump(idx2char, fp)
    return char2idx, idx2char

def load_char_indices():
    """
    Load character indices from disk
    """
    char2idx_path = os.path.join(args.work_dir, CHAR_TO_INDEX_FILE)
    idx2char_path = os.path.join(args.work_dir, INDEX_TO_CHAR_FILE)
    if not (os.path.exists(char2idx_path) and os.path.exists(idx2char_path)):
        raise FileNotFoundError("Character index has not been loaded yet. Please run fit_char_vocab.")
    with open(char2idx_path, "r", encoding="utf-8") as fp:
        char2idx = json.load(fp)
    with open(idx2char_path, "r", encoding="utf-8") as fp:
        idx2char = json.load(fp)
        idx2char = {int(k):v for k,v in idx2char.items()}
    return char2idx, idx2char

def clean_text(text, lowercase=True, remove_punct=True):
    """
    Remove escape characters and additional white spaces.
    Optionally, make all characters lowercase and/or remove punctuation.
    """
    if lowercase:
        text = text.lower()
    if remove_punct:
        punct_removal = str.maketrans({c:"" for c in set(string.punctuation) if c != "."})
        text = text.translate(punct_removal)
    text = text.replace('\n', ' ').replace('\r','').replace('\t', '')
    text = text.strip()
    text = " ".join(text.split())
    return text

# main
if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'hparam_search'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

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
        start_time = time.time()
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
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time}")
    elif args.mode == 'hparam_search':
        print('Loading training data')
        dataset_dict = load_training_data()
        char2idx, idx2char = load_char_indices()
        print('Starting hyperparameter search...')
        best_hparams = hparam_search(dataset_dict["train"], dataset_dict["val"], char2idx, n_iter=10)
        print("Best hyperparams:")
        print(json.dumps(best_hparams, indent=2))
        print("TODO: Update model architecture and retrain for more epochs!")
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
