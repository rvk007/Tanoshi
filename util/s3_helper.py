import os
import json
import pickle
import boto3
from decouple import config


PREFIX = 'training/'
s3 = boto3.client(
    "s3",
    aws_access_key_id=config('AWS_ACCESS_KEY'),
    aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
)


def store_to_s3(filename, data):
    """Update config"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()
    s3.upload_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config.pkl'
    )
    # os.remove(filename)


def read_from_s3(filename):
    """Read config"""
    print('filename', filename)
    s3.download_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config.pkl'
    )
    with open(filename, 'rb') as f:
        data = f.read()

    config_data = pickle.loads(data)
    # os.remove(filename)
    return config_data


def read_bucket():
    """Read contents of a s3 bucket"""
    content = s3.list_objects(Bucket=config("AWS_BUCKET_NAME", Prefix=PREFIX))
    for item in content.get('Contents', []):
        yield item.get('Key')


def put_on_s3(filename):
    """Put a file in a folder(PREFIX) present in s3 bucket on AWS"""
    try:
        s3.put_object(Bucket=config("AWS_BUCKET_NAME"), Key=PREFIX + filename)
        return [True, '']

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return [False, e]


def get_from_s3(username):
    """Get a file in a folder(PREFIX) present in s3 bucket on AWS"""
    try:
        model_object = s3.get_object(Bucket=config("AWS_BUCKET_NAME"), Key=f'{PREFIX}{username}.pt')
        return [True, model_object]

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return [False, e]
