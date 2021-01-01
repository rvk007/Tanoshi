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


def upload_file_to_s3(file, acl="public-read"):
    """Upload dataset"""
    try:
        s3.upload_fileobj(
            file,
            config("AWS_BUCKET_NAME"),
            file.filename,
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )
        return [True, '']

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return [False, e]


def upload_localfile_to_s3(filename):
    """Upload userdata"""
    try:
        s3.upload_file(
            Bucket=config("AWS_BUCKET_NAME"),
            Filename=PREFIX+filename,
            Key=filename
        )
        return True
    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return False


def store_to_s3(filename, data):
    """Update config"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()
    s3.upload_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config'
    )
    # os.remove(filename)


def read_from_s3(filename):
    """Read config"""
    s3.download_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config'
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

def put_on_s3():
    s3.put_object(Bucket=config("AWS_BUCKET_NAME"),)