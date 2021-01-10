import torch
import torch.nn as nn


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
