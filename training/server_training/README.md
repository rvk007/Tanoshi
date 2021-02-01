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

- **[s3](S3.py)**

  - Update a json and place in a s3 bucket,
  - Download and upload a file to s3 bucket.
  - Delete a file from s3 bucket

- **[credentials-sample.py](credentials-sample.py)**: Rename this file to credentials and provide your aws details for the bucket name.

### Json File

**inference.json**: A sample inference.json file is placed [here](Tanoshi/temp/inference.json) for reference.

**Image Clasiification**

```yaml
  'tanoshi-image-171':
    {
      'task_type': 'image',
      'plot_path': 'tanoshi-image-171_accuracy_change.jpg',
      'correct_prediction': 'tanoshi-image-171_correct_predictions.jpg',
      'incorrect_prediction': 'tanoshi-image-171_incorrect_predictions.jpg',
      'model_path': 'tanoshi-image-171_model.pt',
      'classes': { '0': 'jellyfish', '1': 'duck' },
      'accuracy': 83.33,
      'created': '14-01-21 09:05',
    },
```

- **tanoshi-image-171**: username
- **task_type**: Name of the task, wether it is **text** or **image**
- **plot_path**: Name of the image which contains plot of accuracy change
- **correct_prediction**: Name of the image which contains 5x5 images **correctly** predicted
- **incorrect_prediction**: Name of the image which contains 5x5 images **incorrectly** predicted
- **model_path**: Name of the model checkpoint file
- **classes**: Dictionary which contains name of each class given in the dataset
- **accuracy**: Accuracy achieved by the model
- **created**: Date and time once the training is finished

**Sentiment Analysis**

```yaml
'tanoshi-text-433':
  {
    'task_type': 'text',
    'accuracy': 78.33,
    'model_parametes':
      {
        'model_name': 'lstm',
        'input_dim': 56,
        'embedding_dim': 100,
        'hidden_dim': 256,
        'output_dim': 2,
        'number_of_layers': 2,
        'bidirectional': true,
        'dropout': 0.5,
        'pad_index': 1,
      },
    'plot_path': 'tanoshi-text-433_accuracy_change.jpg',
    'model_path': 'tanoshi-text-433_model.pt',
    'tokenizer_path': 'tanoshi-text-433_tokenizer.pkl',
    'classes': { 'pos': 0, 'neg': 1 },
    'created': '10-01-21 13:49',
  }
```

- **tanoshi-text-433**: username
- **task_type**: Name of the task, wether it is **text** or **image**
- **accuracy**: Accuracy achieved by the model
- **model_parametes**: Dictionary containing training hyperparameters
- **model_name**: Name of the task, wether it is **lstm** or **gru**
- **input_dim**: Input dimension for the model
- **embedding_dim**: Embedding dimension of the word embedding
- **hidden_dim**: Hidden Dimension of the model
- **output_dim**: Output Dimension of the model, number of classes
- **number_of_layers**: Number of layers used in forming the model
- **bidirectional**: Wether model is bidirectional or not, **true** or **false**
- **dropout**: Value of Dropout regularization
- **pad_index**: Word embedding index of pad
- **plot_path**: Name of the image which contains plot of accuracy change
- **model_path**: Name of the model checkpoint file
- **tokenizer_path**: Name of the file containg word tokenizer used in inference
- **classes**: Dictionary which contains name of each class given in the dataset
- **created**: Date and time once the training is finished
