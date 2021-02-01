import os
import json
import boto3
import credentials

# Names
BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME', credentials.AWS_BUCKET_NAME)

# S3 Connection
S3_CLIENT = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY', credentials.AWS_ACCESS_KEY),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', credentials.AWS_SECRET_ACCESS_KEY)
)
S3_RESOURCE = boto3.resource(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY', credentials.AWS_ACCESS_KEY),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', credentials.AWS_SECRET_ACCESS_KEY)
)
BUCKET = S3_RESOURCE.Bucket(BUCKET_NAME)


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
