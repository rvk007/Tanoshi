import torch
import pickle
from torchtext import data
import torch.nn as nn
import torch.optim as optim

from util import RNN
from util import create_text_and_label, train, evaluate


def get_config_data(userdata_filename):
    with open(userdata_filename, 'rb') as f:
        data = f.read()

    config_data = pickle.loads(data)
    username = config_data["user_name"]
    model_name = config_data["modelname"]
    ratio = config_data["ratio"]
    loss_function = config_data["lossfunction"]
    optimizer = config_data["optimizer"]
    batch_size = config_data["batch_size"]
    learning_rate = config_data["learning_rate"]
    epochs = config_data["epoch"]
    dataset_filename = config_data['dataset_file']

    return username, model_name, ratio, loss_function, optimizer, batch_size, learning_rate, epochs, dataset_filename


def train_sentiment_analysis(optimizer, loss_function, model_name, learning_rate, epochs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    INPUT_DIM = len(TEXT.vocab)
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = epochs

    best_valid_loss = float('inf')
    valid_acc = 0
    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{username}_text.pt')
    return f'{valid_acc*100:.2f}%'

(username, model_name, ratio, loss_function, optimizer, batch_size, learning_rate, epochs,
    dataset_filename) = get_config_data(userdata_filename)

SEED = 1
BATCH_SIZE = batch_size
EMBEDDING_DIM = 50
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT, LABEL, train_data, valid_data, test_data = create_text_and_label(SEED, ratio)
train_sentiment_analysis(optimizer, loss_function, model_name, learning_rate, epochs)
