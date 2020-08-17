from flask import Flask, render_template, url_for, request, jsonify, send_file
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

@app.route('/upload', methods=['POST'])
def upload():
    imagefile = request.files['image']


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

    colorized_filepath = os.path.join('colorized', imagefile.filename)
    rgb_img.save(colorized_filepath)

    file_object = io.BytesIO()
    rgb_img.save(file_object, 'PNG')
    file_object.seek(0)

    return send_file(file_object, mimetype='image/PNG')

    # image = Image.open(filepath).convert('RGB')
    # return jsonify(prediction=classify_image(loaded_model, image))


    #return jsonify(path="../" + colorized_filepath)

    #return flask.send_file(colorized_filepath, mimetype='image/gif')
    #return render_template("index.html", user_image=colorized_filepath)


if __name__ == '__main__':
    app.run(debug=True)