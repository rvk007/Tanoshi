from flask import Flask, redirect, url_for, request, render_template
from torchvision.models import resnet
from torchvision import transforms
import os 
from PIL import Image
from image_class import image_classes

username = ''

app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/image_training.html')
def train_image():
    return render_template("image_training.html")


@app.route('/text_training.html')
def train_text():
    return render_template("text_training.html")


@app.route('/inference.html')
def inference():
    if request.method == 'POST':
        username = request.form["username"]
        
    popup = False
    
    if popup:
        return render_template("inference.html", popup = popup)
    else:
        return render_template("inference.html")


@app.route('/tour.html')
def tour():
    return render_template("tour.html")


@app.route('/popup.html')
def popup():
    return render_template("popup.html")


@app.route('/no_model_popup.html')
def no_model_popup():
    return render_template("no_model_popup.html")


@app.route('/image_inference.html', methods=["GET","POST"])
def image_inference(filename=None):
    if request.method == 'POST':
        target = os.path.join(app_root, 'static/images')
        f = request.files['image']
        destination = '/'.join([target, f.filename])
        f.save(destination)
        img_path = '/'.join(["static/images", f.filename])

        model = resnet.resnet34(pretrained=True)
        model.eval()

        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transformations(Image.open(img_path)).unsqueeze(0)
        output  = image_classes[model(img_tensor).argmax().item()]
        return render_template("image_inference.html", fname = img_path, prediction = output)
    else:
        return render_template("image_inference.html")


@app.route('/text_inference.html', methods=["GET","POST"])
def text_inference():
    if request.method == 'POST':
        inputSentence = request.form["inputSentence"]
        output  = 'Positive'
        return render_template("text_inference.html", input_Sentence = inputSentence, prediction = output)
    else:
        return render_template("text_inference.html")

if __name__=="__main__":
    app.run(debug=True)  #developer mode



