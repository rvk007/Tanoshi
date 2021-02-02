# Server Training

A t3.2xlarge CPU instance was used to train the models.b The code in this folder is placed on the previously mentioned CPU instance.

## Code Structure

### image_classification

- **[tensornet](image_classification/tensornet)**: A pytorch library for computer vision applications
- **[image.py](image_classification/image.py)**: Provides functionality to train image classification model using custom hyperparameters, provided by the user. Save model checkpoints and training results in data/ folder.

### Sentiment Analysis

- **[text.py](sentiment_analysis/text.py)**: Provides functionality to train sentiment analysis model using custom hyperparameters, provided by the user. Save model checkpoints and training results in data/ folder.
- **[util.py](sentiment_analysis/util.py)**:
  - Defines RNN class for sentiment analysis
  - A helper file for text.py.
- It also contains pretrained word vector **glove.6B.100d.txt** file for word embeddings (not uploaded on github, added in .gitignore). This file can be dowloaded from [here](https://nlp.stanford.edu/projects/glove/).

### Others

- **[main.py](main.py)**:

  - Create data/ and data/checkpoints folder
  - Fetch training parameters from json file present in s3 bucket
  - Download dataset from s3 bucket and place in 'data' folder
  - Start training
  - Upload checkpoints,training results and inference.json file on s3 bucket
  - Delete data/ folder

- **[s3.py](s3.py)**

  - Update a json and place in a s3 bucket,
  - Download and upload a file to s3 bucket.
  - Delete a file from s3 bucket

- **[credentials-sample.py](credentials-sample.py)**: Rename this file to credentials and provide your aws details for the bucket name.

Go [here](../../data_json/README.md) to know about json files.
