from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload')
def upload():
    # Python logic for upload page
    return render_template('upload.html')


@app.route('/contact')
def contact():
    # Python logic for contact page
    return render_template('contact.html')


@app.route('/about')
def about():
    # Python logic for about page
    return render_template('about.html')


@app.route('/campredict')
def campredict():
    # Python logic for campredict page
    return render_template('campredict.html')


if __name__ == '__main__':
    app.run(debug=True)
