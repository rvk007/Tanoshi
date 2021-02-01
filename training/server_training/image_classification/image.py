import os
import pickle
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim

from .tensornet.data.dataset import Dataset
from .tensornet.models import mobilenet_v2, resnet_34
from .tensornet.models.optimizer import sgd
from .tensornet.data.cleaning import clean_data, split_data

from .tensornet.engine.ops import ModelCheckpoint
from .tensornet.engine.ops.lr_scheduler import reduce_lr_on_plateau
from .tensornet.utils import initialize_cuda, get_predictions, save_and_show_result, plot_metric


DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data'
)


def create_image_class(username, filename):
    image_classes = {}
    index = 0
    for f in os.listdir(filename):
        image_classes[index] = f
        index += 1
    return index+1, image_classes


def model_train(
        cuda, device, optimizer, batch_size, learning_rate, epochs, classes, model_name,
        is_reducelrscheduler, patience, factor, min_lr
):
    dataset = Dataset(
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        cuda=cuda,
        num_workers=16,
        path=os.path.join(DATA, 'split_data'),
        random_resize_crop=(224, 224),
        scale=(0.4, 1.0),
        horizontal_flip_prob=0.5
    )

    # Change last model layer to contain 4 classes
    if model_name == 'resnet34':
        model = resnet_34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, classes)
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(1280, classes)

    model = model.to(device)

    # Create train data loader
    train_loader = dataset.loader(train=True)

    # Create val data loader
    val_loader = dataset.loader(train=False)

    criterion = nn.CrossEntropyLoss()

    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Setup model checkpointing
    checkpoint = ModelCheckpoint('checkpoints', monitor='val_accuracy', verbose=1)

    # Setup Reduce LR on Plateau
    if is_reducelrscheduler == 'on':
        scheduler = reduce_lr_on_plateau(
            optimizer, patience=patience, verbose=True, min_lr=min_lr, factor=factor
        )
        model.fit(
            train_loader, optimizer, criterion,
            device=device, epochs=epochs, val_loader=val_loader,
            callbacks=[scheduler, checkpoint], metrics=['accuracy']
        )
    else:
        model.fit(
            train_loader, optimizer, criterion,
            device=device, epochs=epochs, val_loader=val_loader,
            callbacks=[checkpoint], metrics=['accuracy']
        )
    return model, dataset, val_loader


def model_save(model, username):
    model = torch.load(f'{DATA}/checkpoints/model.pt')

    # Save the model to CPU for deployment
    model_deploy = model.to('cpu')
    model_deploy = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    model_deploy.save(f'{DATA}/checkpoints/{username}_model.pt')
    os.remove(f'{DATA}/checkpoints/model.pt')


def model_result(model, username, val_loader, device, dataset):
    correct_predictions, incorrect_predictions = get_predictions(
        model, val_loader, device, sample_count=15
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
        path=DATA,
        username=username
    )
    accuracy_data = {
        'train': model.learner.train_metrics[0]['accuracy'],
        'validation': model.learner.val_metrics[0]['accuracy']
    }
    plot_metric(accuracy_data, 'Accuracy', DATA, username)
    return model.learner.val_metrics[0]['accuracy'][-1]


def train_image_classification(
    username, model_name, ratio, is_reducelrscheduler, patience, factor, min_lr, optimizer,
    batch_size, learning_rate, epochs, dataset_filename
):

    inference = {
        'task_type': 'image',
        'plot_path': f'{username}_accuracy_change.jpg',
        'correct_prediction': f'{username}_correct_predictions.jpg',
        'incorrect_prediction': f'{username}_incorrect_predictions.jpg',
        'model_path': f'{username}_model.pt'
    }
    dataset_filename = os.path.join(DATA, dataset_filename)
    with zipfile.ZipFile(dataset_filename, 'r') as zip_ref:
        zip_ref.extractall('./data/')

    dataset = ''
    possible_dir = ['__MACOSX', 'checkpoints']
    for f in os.listdir('./data/'):
        if f not in possible_dir and f.split('.')[0] != username:
            dataset = f

    os.rename(os.path.join(DATA, dataset), os.path.join(DATA, username))
    filename = os.path.join(DATA, username)
    number_of_classes, classes = create_image_class(username, filename)
    inference['classes'] = classes
    clean_data(filename)

    if ratio == 8:
        ratio = 0.8
    else:
        ratio = 0.7
    split_data(filename, os.path.join(DATA, 'split_data'), split_value=ratio)

    # Initialize CUDA and set random seed
    cuda, device = initialize_cuda(1)

    # Train
    model, dataset, val_loader = model_train(
        cuda, device, optimizer, batch_size, learning_rate, epochs, number_of_classes, model_name,
        is_reducelrscheduler, patience, factor, min_lr
    )

    # Save trained model
    model_save(model, username)

    # Save results
    validation_accuracy = model_result(model, username, val_loader, device, dataset)
    inference['accuracy'] = validation_accuracy
    return inference
