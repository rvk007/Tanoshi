import io
import torch
from torchvision.models import resnet


from util.s3_helper import read_bucket, read_from_s3, get_from_s3

config_filename = 'config.pkl'

def in_bucket(filename):
    list_of_files = read_bucket()
    print(list_of_files)
    if filename in list_of_files:
        return True
    else:
        return False


def username_found(username):
    config_data = read_from_s3(config_filename)
    if username in config_data['list_of_users']['image'].keys():
        return 'image'
    elif username in config_data['list_of_users']['text'].keys():
        return 'text'
    else:
        return False


def get_model(username):
    try:
        model_path = f'{username}.pt'
        get_from_s3(username)
        model = torch.jit.load(model_path)
        return [True, model]
    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return [False, e]

