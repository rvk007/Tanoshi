import os
from flask import Flask, request, render_template, redirect, url_for, flash

from util.train_helper import training
from util.inference_helper import username_found
from util.s3_helper import store_to_s3


username = ''
PREFIX = '_tanoshi'
task = 'image'
app = Flask(__name__)
app.secret_key = 'super secret key'
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

config_filename = 'config.pkl'


@app.route('/')
def home():
    # data = {'status': 'sleeping',
    #         'user_name': 'none',
    #         'list_of_users': {'image': {},
    #                           'text': {}
    #                           }
    #         }
    # store_to_s3(config_filename, data)
    return render_template('home.html')


@app.route('/image_training.html', methods=['GET', 'POST'])
def train_image():
    train_file = 'image_training'
    task = 'image'
    if request.method == 'POST':
        return training(request=request, train_file=train_file, task='image')
    else:
        return render_template(f'{train_file}.html')


@app.route('/text_training.html')
def train_text():
    train_file = 'text_training'
    task = 'text'
    if request.method == 'POST':
        return training(request=request, train_file=train_file, task='test')
    else:
        return render_template(f'{train_file}.html')


@app.route('/inference.html', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        username = request.form['username']
        print(username)
        task = username_found(username)
        if task:
            print("task ", task)
            return redirect(url_for(f'{task}_inference'))
        else:
            flash(' Username is incorrect. Please enter a valid username.')
            return render_template('inference.html')
    else:
        return render_template('inference.html')


@app.route('/tour.html')
def tour():
    return render_template('tour.html')


@app.route('/popup_upload.html')
def popup_upload():
    return render_template('popup_upload.html')


@app.route('/popup_training.html')
def popup_training():
    return render_template('popup_training.html')


@app.route('/popup_no_model.html')
def popup_no_model():
    return render_template('popup_no_model.html')


@app.route('/image_inference.html', methods=['GET', 'POST'])
def image_inference(filename=None):
    print("in  image")
    return render_template('image_inference.html')


@app.route('/text_inference.html', methods=['GET', 'POST'])
def text_inference():
    if request.method == 'POST':
        inputSentence = request.form['inputSentence']
        output = 'Positive'
        return render_template('text_inference.html', input_Sentence=inputSentence, prediction=output)
    else:
        return render_template('text_inference.html')


@app.errorhandler(413)
def request_entity_too_large(error):
    print("Task ", task)
    flash('The file is too large. The maximum allowed size is 200 MB.')
    return redirect(url_for(f'train_{task}'))


if __name__ == '__main__':
    app.run(debug=True)  # developer mode
