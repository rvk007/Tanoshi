# AWS Lambda

This code is deployed in AWS S3 bucket.

## Lambda Functions

There are two purpose of the lambda functions:

- As soon as a json file is added in the s3 bucket the lambda function will start the ec2 instance
- As soon as a json file is deleted from the s3 bucket the lambda function will stop the ec2 instance

## Code Structure

Steps to install serverless can be found [here](https://www.serverless.com/framework/docs/providers/aws/guide/installation/).

- [credentials-sample.yml](credentials-sample.yml): Rename this file to credentials and provide your aws details for the AWS ID, name of the region in which your ec2 instance is placed, name of your s3 bucket and id of the ec2 instance.

- [handler.py](handler.py): Provides the functionality to start and stop the ec2 instance

- [s3.py](s3.py): When the ec2 instance is stopped, update the "status" value to **sleeping** in status.json and upload it back to s3 bucket.

Go [here](../../data_json/README.md) to know about json files.
