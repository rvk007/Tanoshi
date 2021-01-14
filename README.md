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

## Image Classification

In this process an algorithm takes an image as input and tells you what is there in that image, much like what is shown below.

<div align="center">
  <img src="web/static/information/image_classification.gif" height="180px" />
</div>

It provides two models **Resnet34** and **MobileNetV2** which are pretrained on Imagenet dataset you can use either of them. Create a custom dataset which follows the below mentioned description and you can start training.

<div align="center">
  <img src="images/image_dataset.png" height="150px" />
</div>

## Sentiment Analysis

<div align="center">
  <img src="web/static/information/sentiment_analysis.gif" height="180px" />
</div>

You can create a custom dataset and train using either **LSTM** or **GRU** from scratch. Allowed dataset format is shown below.

<div align="center">
  <img src="images/text_dataset.png" height="150px" />
</div>

## Code Structure

There are three major components of this project:

- Heroku
- EC2 instance
- AWS Lambda

<div align="center">
  <img src="images/flowchart.png" height="270px" />
</div>

You can go to the above links to know more about this project in detail.
