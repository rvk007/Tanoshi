# Web

The application is built on Flask.

## Code Structure

- Static
- Templates
- Util

### Static

This folder contains css and js files and stores all the images shown in the project.

### Templates

This folder contains all the html files.

### Util

This folder contains python files which provides the following functionality:

- **[s3_helper](util/s3_helper.py)**:
  - Update a json and place in a s3 bucket,
  - Download and upload a file to s3 bucket.
- **[train_helper](util/train_helper.py)**:
  - Check if training is happening on the ec2 instance or not.
  - Performs validation on the inputs provided by the user, if the input is not acceptable it raises an alert else update the main json file that training has to be started.
  - Creates a new json file for the user which will contain the details of the training parameters provided by the user and upload the json and dataset file on the s3 bucket.
- **[inference_helper](util/inference_helper.py)**:
  - Checks if the username provided by the user is correct or not
  - Downloads model and images for the inference from s3 bucket.
- **[text_model](util/text_model.py)**:
  - Defines RNN class for sentiment analysis

### Others

- **[app](app.py)**: This files helps application to route between html pages, checks the size of the dataset provided and saves inference information.

- **[credentials-sample](credentials-sample.py)**: Rename this file to credentials and provide your aws details for the bucket name, access key and secret access key.

- **[Procfile](Procfile)**: Heroku was for deploying the code present in the web folder. 'app.py' is the file name mentioned in the Procfile from which UI of the application starts.

In heroku as well we can store the values we have listed in credentials.py. To access those value use os.environ[<KEY_NAME>].

Go [here](../data_json/README.md) to know about json files.
