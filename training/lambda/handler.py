try:
    import unzip_requirements
except ImportError:
    pass

import os
import boto3
import json

from s3 import (
    fetch_status,
    change_server_status,
)


INSTANCE_ID = os.environ.get('INSTANCE_ID')
REGION = os.environ.get('REGION')

EC2_RESOURCE = boto3.resource('ec2', region_name=REGION)


def create_response(body, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Credentials': True
        },
        'body': json.dumps(body)
    }


def server_start(event, context):
    message = 'Status not active. Server not turned on.'

    server_status = fetch_status()
    if server_status['dev_mode']:
        message = 'Dev mode is on.'
    elif server_status['status'] == 'active':
        ec2_client = boto3.client('ec2', region_name=REGION)
        ec2_client.start_instances(InstanceIds=[
            INSTANCE_ID
        ])
        message = 'Instance started.'

    print(message)
    return create_response({
        'message': message
    })


def server_stop(event, context):
    server_status = fetch_status()
    if server_status['dev_mode']:
        message = 'Dev mode is on.'
    else:
        # Stop instance
        ec2_client = boto3.client('ec2', region_name=REGION)
        ec2_client.stop_instances(InstanceIds=[
            INSTANCE_ID
        ])
        message = 'Instance stopped.'

        # Change server status
        change_server_status('sleeping', server_status['dev_mode'])

    print(message)
    return create_response({
        'message': message
    })
