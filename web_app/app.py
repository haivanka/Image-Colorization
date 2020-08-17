from flask import Flask, render_template, url_for, request, jsonify, send_from_directory, send_file
from flask_bootstrap import Bootstrap
import os
import io
import base64
from PIL import Image
import numpy as np
import re
from io import StringIO
from web_app.colorization.colorization import Colorization


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "Uploads"
colorization = Colorization()
Bootstrap(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['image']
    filepath = os.path.join('uploads', imagefile.filename)
    imagefile.save(filepath)

    pixels = request.form['pixels']
    image_data = re.sub('^data:image/.+;base64,', '', pixels)
    hints_filename = os.path.join('uploads', imagefile.filename + '_hints.png')

    with open(hints_filename, "wb") as fh:
        fh.write(base64.urlsafe_b64decode(image_data))

    rgb_img = colorization.colorize(filepath, hints_filename)

    colorized_filepath = os.path.join('colorized', imagefile.filename + '.png')
    rgb_img.save(colorized_filepath)

    encoded = base64.b64encode(open(colorized_filepath, "rb").read())
    return encoded


if __name__ == '__main__':
    app.run(debug=True)