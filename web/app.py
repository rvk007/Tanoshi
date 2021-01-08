import os
from PIL import Image
from torchvision import transforms
from flask import Flask, request, render_template, redirect, url_for, flash

from image_class import image_classes
from util.train_helper import training
from util.inference_helper import username_found, get_image_model, get_text_model, inference_cache
from util.s3_helper import store_to_s3


username = ''
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
    #store_to_s3(config_filename, data)
    return render_template('home.html')


@app.route('/image_training', methods=['GET', 'POST'])
def train_image():
    train_file = 'image_training'
    task = 'image'
    if request.method == 'POST':
        return training(request=request, train_file=train_file, task=task)
    else:
        return render_template(f'{train_file}.html')


@app.route('/text_training', methods=['GET', 'POST'])
def train_text():
    train_file = 'text_training'
    task = 'text'
    if request.method == 'POST':
        return training(request=request, train_file=train_file, task=task)
    else:
        return render_template(f'{train_file}.html')


@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        username = request.form['username']
        task = username_found(username)
        if task:
            return redirect(url_for(f'{task}_inference', user_name=username))
        else:
            flash(' Username is incorrect. Please enter a valid username.')
            return render_template('inference.html')
    else:
        return render_template('inference.html')


@app.route('/tour')
def tour():
    return render_template('tour.html')

@app.route('/image_inference/<user_name>', methods=['GET', 'POST'])
def image_inference(user_name):
    if request.method == 'POST':
        image_file = request.files['input_image']
        destination = os.path.join(app_root, 'static/images')
        print(destination)
        for files in os.listdir(destination):
            os.remove(os.path.join(destination, files))

        destination = '/'.join([destination, image_file.filename])
        image_file.save(destination)
        image_path = '/'.join(['/static/images', image_file.filename])
        inf_exist = inference_cache(user_name, app_root)
        model = get_image_model(user_name, inf_exist)
        if model[0]:
            transformations = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transformations(Image.open(destination)).unsqueeze(0)
            output = image_classes[model[1](image_tensor).argmax().item()]
            print("image_path ", image_path)
            return render_template('image_inference.html', file_name=image_path, prediction=output)
        else:
            flash(f'An error has occured: {model[1]}')
            return render_template('image_inference.html')
    else:
        return render_template('image_inference.html')


@app.route('/text_inference/<user_name>', methods=['GET', 'POST'])
def text_inference(user_name):
    if request.method == 'POST':
        input_sentence = request.form['inputSentence']
        inf_exist = inference_cache(user_name, app_root)
        output = get_text_model(
            input_sentence,
            'upgraded_sentiment_analysis.pt',
            'upgraded_sentiment_analysis_metadata.pkl',
            inf_exist
        )
        if output[0]:
            pred = ''
            if output[1] == 'neg':
                pred = 'Negative'
            elif output[1] == 'pos':
                pred = 'Positive'
            return render_template('text_inference.html', input_Sentence=input_sentence, prediction=pred)
        else:
            flash(f'An error has occured: {output[1]}')
            return render_template('text_inference.html')
    else:
        return render_template('text_inference.html')


@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large. The maximum allowed size is 200 MB.')
    return redirect(url_for(f'train_{task}'))


if __name__ == '__main__':
    app.run(debug=True)  # developer mode
