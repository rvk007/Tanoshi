import os
import pickle
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim

from tensornet.data.dataset import Dataset
from tensornet.models import mobilenet_v2, resnet_34
from tensornet.models.optimizer import sgd
from tensornet.data.cleaning import clean_data, split_data

from tensornet.engine.ops import ModelCheckpoint
from tensornet.engine.ops.lr_scheduler import reduce_lr_on_plateau
from tensornet.utils import initialize_cuda, get_predictions, save_and_show_result


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


def create_image_class(filename):
    image_classes = {}
    index = 0
    for f in os.listdir(filename):
        image_classes[index] = f
        index += 1
    with open('image_class.py', 'w+') as f:
        f.write(f'image_classes={image_classes}')
        f.close()
    return index+1


def model_train(cuda, device, optimizer, batch_size, learning_rate, epochs, classes):
    dataset = Dataset(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        cuda=cuda,
        num_workers=16,
        path='data',
        random_resize_crop=(224, 224),
        scale=(0.4, 1.0),
        horizontal_flip_prob=0.5
    )
    dataset.create_class_imbalance_sampler()

    # Change last model layer to contain 4 classes
    if model_name == 'resnet34':
        model = resnet_34(pretrained=True)
        model.fc.out_features = classes
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(1280, classes)

    model = model.to(device)

    # Create train data loader
    train_loader = dataset.loader(train=True)

    # Create val data loader
    val_loader = dataset.loader(train=False)

    if loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Setup Reduce LR on Plateau
    scheduler = reduce_lr_on_plateau(
        optimizer, patience=2, verbose=True, min_lr=1e-6
    )

    # Setup model checkpointing
    checkpoint = ModelCheckpoint('checkpoints', monitor='val_accuracy', verbose=1)

    model.fit(
        train_loader, optimizer, criterion,
        device=device, epochs=epochs, val_loader=val_loader,
        callbacks=[scheduler, checkpoint], metrics=['accuracy']
    )

    return model, dataset, val_loader


def model_save(model):
    model = torch.load('checkpoints/model.pt')

    # Save the model to CPU for deployment
    model_deploy = model.to('cpu')
    model_deploy = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    model_deploy.save('mobilenetv2.pt')


def model_result(model, val_loader, device, dataset):
    correct_predictions, incorrect_predictions = get_predictions(
        model, val_loader, device, sample_count=10
    )
    # Convert predictions to numpy array
    for idx in range(len(incorrect_predictions)):
        incorrect_predictions[idx]['image'] = dataset.unnormalize(
            incorrect_predictions[idx]['image'], transpose=True
        )
        correct_predictions[idx]['image'] = dataset.unnormalize(
            correct_predictions[idx]['image'], transpose=True
        )

        # Display predictions
    save_and_show_result(
        dataset.classes,
        correct_pred=correct_predictions,
        incorrect_pred=incorrect_predictions,
        path='.'
    )


def train_image_classification(ratio, batch_size, learning_rate, epochs, dataset_filename):
    with zipfile.ZipFile(dataset_filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    filename = 'dataset'
    classes = create_image_class(filename)

    if ratio == 8:
        ratio = 0.8
    else:
        ratio = 0.7
    split_data('dataset', 'data', split_value=ratio)
    # Initialize CUDA and set random seed
    cuda, device = initialize_cuda(1)
    # Train
    model, dataset, val_loader = model_train(cuda, device, optimizer, batch_size, learning_rate, epochs, classes)
    # Save trained model
    model_save(model)
    # Save results
    model_result(model, val_loader, device, dataset)


(username, model_name, ratio, loss_function, optimizer, batch_size, learning_rate, epochs,
    dataset_filename) = get_config_data(userdata_filename)
train_image_classification(batch_size, learning_rate, epochs)
