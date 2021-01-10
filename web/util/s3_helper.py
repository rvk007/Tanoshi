import os
import json
import pickle
import boto3
import credentials
from decouple import config

# Names
BUCKET_NAME = credentials.BUCKET_NAME

# S3 Connection
S3_CLIENT = boto3.client('s3')
S3_RESOURCE = boto3.resource('s3')
BUCKET = S3_RESOURCE.Bucket(BUCKET_NAME)

PREFIX = 'training/'
bucket = (
        os.environ['AWS_BUCKET_NAME']
        if 'AWS_BUCKET_NAME' in os.environ else config('AWS_BUCKET_NAME')
    )
s3 = boto3.client(
    's3',
    aws_access_key_id=(
        os.environ['AWS_ACCESS_KEY']
        if 'AWS_ACCESS_KEY' in os.environ else config('AWS_ACCESS_KEY')
    ),
    aws_secret_access_key=(
        os.environ['AWS_SECRET_ACCESS_KEY']
        if 'AWS_SECRET_ACCESS_KEY' in os.environ else config('AWS_SECRET_ACCESS_KEY')
    )
)


def fetch_json(path):
    obj = S3_CLIENT.get_object(
        Bucket=BUCKET_NAME, Key=path
    )
    return json.loads(obj['Body'].read())


def put_object(path, data):
    BUCKET.put_object(
        ContentType='application/json',
        Key=path,
        Body=json.dumps(data, indent=2).encode('utf-8'),
    )


def download_file(path, target_path):
    S3_CLIENT.download_file(BUCKET_NAME, path, target_path)


def upload_file(source_path, target_path):
    try:
        S3_CLIENT.upload_file(source_path, BUCKET_NAME, target_path)
        return [True, '']

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        return [False, e]


def delete_object(path):
    S3_RESOURCE.Object(BUCKET_NAME, path).delete()

    
def store_to_s3(filename, data):
    """Update config"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()
    s3.upload_file(
        Bucket=bucket,
        Filename=filename,
        Key='config.pkl'
    )
    # os.remove(filename)


def read_from_s3(filename):
    """Read config"""
    s3.download_file(
        Bucket=bucket,
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
        s3.put_object(Bucket=bucket, Key=PREFIX + filename)
        return [True, '']

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        return [False, e]


def get_from_s3(filename):
    """Get a file in a folder(PREFIX) present in s3 bucket on AWS"""
    try:
        model_object = s3.get_object(Bucket=bucket, Key=PREFIX+filename)
        return [True, model_object]

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        return [False, e]


def download_from_s3(filename):
    """Download files from s3"""
    s3.download_file(
        Bucket=bucket,
        Key=PREFIX+filename,
        Filename=filename
    )
