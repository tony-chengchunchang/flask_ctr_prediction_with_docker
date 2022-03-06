import os
import datetime
from flask import Blueprint, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from common.image_handler import is_allowed_file

predict_route = Blueprint('predict', __name__)
UPLOAD_FOLDER = 'static/uploads'
ONEHOT_ENCODER_FILE_PATH = 'ctr_model/onehot_encoder.pkl'
MODEL_PATH = 'ctr_model/all_industries_model_checkpoint/'
INDUSTRY_TYPES = [
    'Select industry', 'OTHERS', 'H&B', 'ENT', 'A&A', 'F&B', 'EDU', 'TECH',
    'PHARMA', 'RETAIL', 'FIN', 'HOUSE', 'EC', 'AUTO', 'GOV', 'GAME', 'TRAVEL'
]

model = tf.keras.models.load_model(MODEL_PATH)

with open(ONEHOT_ENCODER_FILE_PATH, 'rb') as f:
    ohe = pickle.load(f)

def predict(img_path, age, gender, season, flight, industry):
    img_input = np.array([np.array(Image.open(img_path).resize((128,128)))])
    brf_input = ohe.transform([[age, gender, season]])
    brf_input = tf.expand_dims(np.append(brf_input, int(flight)), axis=0)
    ind_input = np.array([industry])

    pred = model.predict([img_input, brf_input, ind_input])

    return pred[0][0]

@predict_route.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    result = 'Result Here'
    data = {'age': '', 'gender': '', 'season': '',
        'flight': '', 'industry': ''}
    if request.method == 'POST':
        img = request.files.get('image')
        data = request.form
        if img and is_allowed_file(img.filename):
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = secure_filename(img.filename)
            formatted_file_name = '{}_{}'.format(now, file_name)
            img_path = os.path.join(
                UPLOAD_FOLDER, formatted_file_name)
            img.save(img_path)

            result = predict(img_path, **data)
        else:
            result = 'Invalid file. Please try again.'

    return render_template(
        'upload.html', result=result,
        industry_types=INDUSTRY_TYPES,
        **data
    )
