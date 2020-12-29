import os
import json
import pickle
from io import BytesIO
import gzip
import boto3, botocore
from werkzeug.utils import secure_filename
from decouple import config

filename = 'config.pkl'
s3 = boto3.client(
    "s3",
    aws_access_key_id=config('AWS_ACCESS_KEY'),
    aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY')
)


def upload_file_to_s3(file, acl="public-read"):
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
        return [True,'']

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return [False, e]

def upload_localfile_to_s3(filename):
    try:
        s3.upload_file(
        Bucket = config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key=filename
        )
        return True

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Error occured: ", e)
        return False

def store_to_s3(data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        f.close()

    s3.upload_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config'
    )
    os.remove(filename)

def read_from_s3():
    s3.download_file(
        Bucket=config("AWS_BUCKET_NAME"),
        Filename=filename,
        Key='config'
    )
    
    with open(filename, 'rb') as f:
        data = f.read()

    config_data = json.loads(data)
    os.remove(filename)
    return config_data



