<div align="center">
  <img src="web/static/information/logo.png" height="100px" />
  <h1 class="custom-inline">Tanoshi</h1>
</div>

![progress badge](https://img.shields.io/badge/status-version%201.0-blue)
[![Website](https://img.shields.io/badge/Website-orange.svg)](https://tanoshi.herokuapp.com/)

This is an end-to-end platform where you can upload your own custom dataset, set model parameters and train your own deep learning model without writing any line of code.

Tanoshi currently provides Image classification and Sentiment Analysis machine learning models.
It has a feature to create a custom model by setting model parameters such as:

- Batch size
- Optimizer
- Learning rate
- Number of epochs
- Training-Validation ratio

After validating the model parameters an input dataset file a user token is created and training starts, which takes around five-ten minutes. Once training is completed the user can use that token to test their model.

There are three major components of this project:

- [Heroku](web)
- [EC2 instance](training/server_training)
- [AWS Lambda](training/lambda)

[Go to the above links to know more about this project in detail.]

The below image explains the work of each component and how they are related to each other:

<div align="center">
  <img src="images/flowchart.png" height="350px" />
</div>

## Image Classification

Image classification, as the name suggests, ia an algorithm which predicts the content of an image. This project provides two different models, **Resnet34** and **MobileNetV2** which are pretrained on Imagenet dataset, to classify an image. Use the below format while creating the dataset and make sure to **zip** it before uploading else it won't be accepted.

<div align="center">
  <img src="images/image_dataset.png" height="220px" />
</div>

## Sentiment Analysis

Sentiment Analysis is a type of text classification, where a text is classified as **Positive** and **Negative**. Although, in this project Sentiment analysis is trained from scratch so it can be used for any kind text classification. Again, use the below specified format to create your dataset. The file should be a **csv** file.

<div align="center">
  <img src="images/text_dataset.png" height="200px" />
</div>

