# ===================
# Simple image classifier website by Debjyoti Chowdhury
# ===================

import os
import datetime
from flask import Flask, render_template, request, session
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing import image
index = 0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

app = Flask(__name__)
app.secret_key = os.urandom(24)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = 'static/uploaded_images'
UPLOAD_FOLDER = os.path.join(APP_ROOT, UPLOAD_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def write_to_csv(num, clf, index_val):
    values = {'Image': [num], 'classified': [clf]}
    df_w = pd.DataFrame(values, columns=['Image', 'classified'])
    df_w.to_csv("results.csv", index=index_val, header=True)


@app.route('/')
def upload_file():
    return render_template('index.html', uploaded_image='default-image.png', classification="unclassified")


@app.route('/uploader', methods=['POST'])
def image_upload():
    current_dt = datetime.datetime.now()
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            session['logged_in'] = True
            filename = 'uploaded_{date_0}_{time_0}.jpg'.format(date_0=current_dt.strftime('%Y-%m-%d'),
                                                               time_0=current_dt.strftime('%H-%M-%S-%f'))
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(path)
            session['path'] = filename
            return render_template('index.html', uploaded_image=filename, classification="unclassified")
        else:
            return 'Unsupported file type. Try [png, jpg or jpeg] <a href="/">Go back?</a>'


@app.route('/classify', methods=['POST'])
def image_classify():
    if request.method == 'POST':
        from_path = session['path']
        # load_model = tf.keras.models.load_model('saved_model/my_model')
        load_model = tf.lite.Interpreter(
            model_path="saved_model/tflite_model/converted_model.tflite")
        load_model.allocate_tensors()
        input_details = load_model.get_input_details()
        output_details = load_model.get_output_details()
        img = image.load_img('static/uploaded_images/' + from_path,
                             target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        # classes = load_model.predict(images, batch_size=10)
        load_model.set_tensor(input_details[0]['index'], images)
        load_model.invoke()
        classes = load_model.get_tensor(output_details[0]['index'])
        if classes[0] > 0.5:
            write_to_csv(str(from_path), 1, index+1)
            print(str(from_path))
            return render_template('index.html', uploaded_image=from_path,
                                   classification="uploaded image is of a human")
        else:
            write_to_csv(str(from_path), 0, index+1)
            print(str(from_path))
            return render_template('index.html', uploaded_image=from_path,
                                   classification="uploaded image is of a horse")


if __name__ == '__main__':
    app.run()
