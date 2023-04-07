from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Logic to handle form submission
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)
