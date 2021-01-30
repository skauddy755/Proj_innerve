from flask import Flask, render_template, redirect, request
import os
import subprocess, base64
import cv2
from PIL import Image
from utils import img2heatmap
from alzheimer.classification.classification import predict

app = Flask(__name__)

app.config["ALZHEIMER_IMAGE_UPLOADS"] = os.getcwd()+"/alzheimer/input_imgs/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["ALZHEIMER_IMAGE_HEATMAP"] = os.getcwd()+"/alzheimer/heat_map/"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/aboutus')
def about():
    return render_template('about_us.html')

@app.route('/contactus')
def contact():
    return render_template('contact_us.html')

@app.route('/services')
def service():
    return render_template('services.html')

@app.route('/services/alzheimer', methods=['GET', 'POST'])
def alzheimersdetection():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename =image.filename 
            name, ext = filename.split(".")
            
            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                image.save(os.path.join(app.config["ALZHEIMER_IMAGE_UPLOADS"], filename))
                subprocess.call(f'python3 alzheimer/ml_backend/api.py --file {filename}', shell=True)
                img = img2heatmap(f'alzheimer/input_imgs/{name}.{ext}', f'alzheimer/output_imgs/{name}_predicted.{ext}')
                print(os.getcwd()+f'heatmap/{name}_map.{ext}')
                cv2.imwrite(os.path.join(app.config["ALZHEIMER_IMAGE_HEATMAP"], f'{name}_map.{ext}'), img)
                with open(os.getcwd()+f'/alzheimer/heat_map/{name}_map.{ext}', 'rb') as out_raw:
                    out_img64 = base64.b64encode(out_raw.read())
                out_img64 = out_img64.decode("utf-8")
                prediction = predict(f'alzheimer/input_imgs/{filename}', 'alzheimer/classification/saved_weight/current_checkpoint.pt')
                return render_template('alzheimer.html', predicted = True, imgData = out_img64, supplied_text = f'{prediction}')
            else:
                return "NON SUPPORTED FILE TYPE"
    return render_template('alzheimer.html', predicted = False)

@app.route('/services/tuberculosis')
def tuberculosis():
    return "<h1>Working On it</h1>"

@app.route('/services/breast')
def breast():
    return "<h1>Working On it</h1>"

@app.route('/services/melanoma')
def melanoma():
    return "<h1>Working On it</h1>"


@app.route('/blog')
def blog():
    return render_template('blog.html')


# @app.route('/<link>')
# def e404():
#     r

app.run(debug=True)