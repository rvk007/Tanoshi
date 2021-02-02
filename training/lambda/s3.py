import os
import json
import boto3


# Names
BUCKET_NAME = os.environ.get('S3_BUCKET')
STATUS_FILE = 'status.json'

# S3 Connection
S3_CLIENT = boto3.client('s3')
S3_RESOURCE = boto3.resource('s3')
BUCKET = S3_RESOURCE.Bucket(BUCKET_NAME)


def fetch_status():
    print('Connecting to S3...')
    obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=STATUS_FILE)
    return json.loads(obj['Body'].read())


def change_server_status(new_status, dev_mode, username=None):
    print('Changing training server status')
    BUCKET.put_object(
        ContentType='application/json',
        Key=STATUS_FILE,
        Body=json.dumps({
            'status': new_status,
            'username': '' if username is None else username,
            'dev_mode': dev_mode,
        }, indent=2).encode('utf-8'),
    )
