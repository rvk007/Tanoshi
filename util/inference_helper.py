from util.s3_helper import read_bucket, read_from_s3

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
