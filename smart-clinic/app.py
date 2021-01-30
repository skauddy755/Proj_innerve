from flask import Flask, render_template, redirect

app = Flask(__name__)

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

@app.route('/blog')
def blog():
    return render_template('blog.html')


app.run(debug=True)