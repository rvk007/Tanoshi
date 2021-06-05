import os
import torch
import pickle
from torchtext import data as torchtext_data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .util import RNN
from .util import create_text_and_label, train, evaluate, save_model_cpu, plot_accuracy


DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data'
)


def train_sentiment_analysis(
    username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr, optimizer,
        batch_size, learning_rate, epochs, dataset_filename
):
    SEED = 1
    BATCH_SIZE = batch_size
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = f'./data/{username}.csv'
    TEXT, LABEL, train_data, valid_data = create_text_and_label(SEED, ratio, filename)

    train_iterator, valid_iterator = torchtext_data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)
        
    print('Text iterator created')
    with open(f'./data/{username}_tokenizer.pkl', 'wb') as tokens:
        pickle.dump(TEXT.vocab.stoi, tokens)
    
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    INPUT_DIM = len(TEXT.vocab)
    OUTPUT_DIM = len(LABEL.vocab)
    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX,
                model_name)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    print('Getting ready for training')

    if is_reducelrscheduler == 'on':
        scheduler = ReduceLROnPlateau(
            optimizer, factor=factor, patience=patience, verbose=False, min_lr=min_lr
        )

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = epochs

    best_valid_loss = float('inf')
    valid_acc = 0
    train_accuracy = []
    valid_accuracy = []

    print('Training started ')
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)

        if is_reducelrscheduler == 'on':
            scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{DATA}/checkpoints/{username}_best.pt')

        print(f'Epoch:{epoch}')
        print(f' Train Loss: {train_loss} Train accuracy: {train_acc}')
        print(f' Validation Loss: {valid_loss} Validation accuracy: {valid_acc}')

    save_model_cpu(model, username)
    plot_accuracy(username, train_accuracy, valid_accuracy)
    print('Trained model and images saved')

    classes = LABEL.vocab.stoi
    classify = {}
    for (k, v) in classes.items():
        classify[v] = k

    stoi = {}
    for k, v in TEXT.vocab.stoi.items():
        if (k != TEXT.unk_token and v != UNK_IDX) or (k == TEXT.unk_token):
            stoi[k] = v

    inference = {
        'task_type': 'text',
        'accuracy': float(valid_acc),
        'input_stoi': stoi,
        'label_itos': LABEL.vocab.stoi,
        'unk_idx': UNK_IDX,
        'model_parametes': {
            'model_name': model_name,
            'input_dim': INPUT_DIM,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'number_of_layers': N_LAYERS,
            'bidirectional': BIDIRECTIONAL,
            'dropout': DROPOUT,
            'pad_index': PAD_IDX,
        },
        'plot_path': f'{username}_accuracy_change.jpg',
        'model_path': f'{username}_model.pt',
        'tokenizer_path': f'{username}_tokenizer.pkl',
        'classes': classify
    }

    print(" Returning from text classification")

    return inference
    