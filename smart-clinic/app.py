from flask import Flask, render_template, redirect, request
import os
import subprocess, base64
import cv2
from PIL import Image
from utils import img2heatmap
from alzheimer.classification.classification import predict
from aptos.inference_aptos import predict_aptos
from melanoma.inference_melanoma import predict_melanoma
from pneumonia.inference_pneumonia import predict_pneumonia
from breast_cancer.inference_brest import predict_breast

app = Flask(__name__)

app.config["PNEUMONIA_UPLOADS"] = os.getcwd()+"/pneumonia/input_imgs"
app.config["BREAST_CANCER_UPLOADS"] = os.getcwd()+"/breast_cancer/input_imgs"
app.config["MELANOMA_UPLOADS"] = os.getcwd()+"/melanoma/input_img"
app.config["APTOS_UPLOADS"] = os.getcwd()+"/aptos/input_imgs"
app.config["ALZHEIMER_IMAGE_UPLOADS"] = os.getcwd()+"/alzheimer/input_imgs/"
app.config["ALZHEIMER_IMAGE_HEATMAP"] = os.getcwd()+"/alzheimer/heat_map/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]



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


@app.route('/services/breast', methods=['POST', 'GET'])
def breast():
    if request.method == 'POST':
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename = image.filename 
            name, ext = filename.split('.')

            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                image.save(os.path.join(app.config["BREAST_CANCER_UPLOADS"], filename))
                result = predict_breast(os.path.join(app.config["BREAST_CANCER_UPLOADS"], filename))
                return render_template('breast.html', predicted=True, supplied_text = result)
    return render_template('breast.html', predicted=False)

@app.route('/services/melanoma', methods=['POST', 'GET'])
def melanoma():
    if request.method == 'POST':
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename = image.filename 
            name, ext = filename.split('.')

            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                image.save(os.path.join(app.config["MELANOMA_UPLOADS"], filename))
                result = predict_melanoma(os.path.join(app.config["MELANOMA_UPLOADS"], filename))
                return render_template('melanoma.html', predicted=True, supplied_text = result)
    return render_template("melanoma.html", predicted=False)

@app.route('/services/pneumonia', methods=['POST', 'GET'])
def pneumonia():
    if request.method == 'POST':
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename = image.filename 
            name, ext = filename.split('.')

            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                image.save(os.path.join(app.config["PNEUMONIA_UPLOADS"], filename))
                result = predict_pneumonia(os.path.join(app.config["PNEUMONIA_UPLOADS"], filename))
                return render_template('pneumonia.html', predicted=True, supplied_text = result)
    return render_template('pneumonia.html', predicted=False)

@app.route('/services/aptos', methods=['POST', 'GET'])
def aptos():
    if request.method == 'POST':
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename = image.filename 
            name, ext = filename.split('.')

            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                image.save(os.path.join(app.config["APTOS_UPLOADS"], filename))
                result = predict_aptos(os.path.join(app.config["APTOS_UPLOADS"], filename))
                return render_template('aptos.html', predicted=True, supplied_text = result)
    return render_template('aptos.html', predicted = False)

@app.route('/<link>')
def page_not_found(link):
    return render_template('404.html')

app.run(debug=True)