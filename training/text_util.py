import torch
import random
import torch.nn as nn
from torchtext import data


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, model_name):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.model_name = model_name
        if model_name == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout
            )
        elif model_name == 'gru':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout
            )
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embedded = self.dropout(self.embedding(text))
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        if self.model_name == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        elif self.model_name == 'gru':
            packed_output, hidden = self.rnn(packed_embedded)
            
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # hidden = [num layers * num directions, batch size, hid dim]
        hidden = self.dropout(
            torch.cat((
                hidden[-2, :, :], hidden[-1, :, :]),
                dim=1)
        )
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


def create_text_and_label(SEED, ratio):
    TEXT = data.Field(tokenize='spacy', include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('text', TEXT), ('label', LABEL)]
    train_data = data.TabularDataset.splits(
                                            path='',
                                            train='/content/dataset.csv',
                                            format='csv',
                                            fields=fields,
                                            skip_header=True
    )
    train_data = train_data[0]
    if ratio == 8:
        ratio = 0.8
    else:
        ratio = 0.7
    train_data, test_data = train_data.split(split_ratio=ratio, random_state=random.seed(5))
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(
        train_data,
        max_size=MAX_VOCAB_SIZE,
        vectors="glove.6B.50d",
        unk_init=torch.Tensor.normal_
    )

    LABEL.build_vocab(train_data)
    return TEXT, LABEL, train_data, valid_data, test_data


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    # convert into float for division 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths.cpu()).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths.cpu()).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
