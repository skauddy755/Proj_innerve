from flask import Flask, render_template, redirect
import os
import subprocess, base64
import cv2
from PIL import Image
# from alzheimer.classification import predict

app = Flask(__name__)

app.config["ALZHEIMER_IMAGE_UPLOADS"] = os.getcwd()+"/smart-clinic/alzheimer/input_imgs/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["ALZHEIMER_IMAGE_HEATMAP"] = os.getcwd()+"smart-clinic/alzheimer/heat_map/"


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

@app.route('/services/tuberculosis')
def tuberculosis():
    return render_template('tuberculosis.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


app.run(debug=True)