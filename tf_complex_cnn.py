import os
import argparse
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

tf.random.set_seed(1)

def preprocessing(base_dir):
    df = pd.read_csv(os.path.join(base_dir, 'agg_revised.csv'))
    
    ages = sorted(df['age'].unique())
    genders = sorted(df['gender'].unique())
    seasons = sorted(df['season'].unique())
    ohe = OneHotEncoder(categories=[ages, genders, seasons], sparse=False, drop='first')
    brf_df = pd.DataFrame(ohe.fit_transform(df[['age', 'gender', 'season']]))
    brf_df['n'] = df['flight'].values
    
    img_path_arr = df['url_encoded'].values
    brief_arr = brf_df.values
    label_arr = df['ctr'].values
    
    return img_path_arr, brief_arr, label_arr

def make_img_generator(img_path_array, img_size, batch_size):
    while True:
        for i in range(0, len(img_path_array), batch_size):
            imgs = np.array([np.array(Image.open(p).resize(img_size)) for p in img_path_array[i:i+batch_size]])
            yield imgs
            
def make_brief_generator(brief_array, batch_size):
    while True:
        for i in range(0, len(brief_array), batch_size):
            yield brief_array[i:i+batch_size]
            
def make_label_generator(label_array, batch_size):
    while True:
        for i in range(0, len(label_array), batch_size):
            yield label_array[i:i+batch_size]
        
def make_input_generator(img_gen, brief_gen, label_gen, batch_size):
    while True:
        x1 = next(img_gen)
        x2 = next(brief_gen)
        x3 = next(label_gen)
        yield [x1, x2], x3
        
def process_input(img_path_arr, brief_arr, label_arr, img_size=(128, 128), batch_size=32):
    split_at = round(len(img_path_arr) * 0.8)

    train_img_path_array = img_path_arr[:split_at]
    train_img_gen = make_img_generator(train_img_path_array, img_size, batch_size)
    test_img_path_array = img_path_arr[split_at:]
    test_img_gen = make_img_generator(test_img_path_array, img_size, batch_size)
    print(len(train_img_path_array), len(test_img_path_array))

    train_brief_array = brief_arr[:split_at]
    train_brf_gen = make_brief_generator(train_brief_array, batch_size)
    test_brief_array = brief_arr[split_at:]
    test_brf_gen = make_brief_generator(test_brief_array, batch_size)
    print(len(train_brief_array), len(test_brief_array))

    train_label_array = label_arr[:split_at]
    train_lable_gen = make_label_generator(train_label_array, batch_size)
    test_label_array = label_arr[split_at:]
    test_label_gen = make_label_generator(test_label_array, batch_size)
    print(len(train_label_array), len(test_label_array))
    
    train_gen = make_input_generator(train_img_gen, train_brf_gen, train_lable_gen, batch_size)
    test_gen = make_input_generator(test_img_gen, test_brf_gen, test_label_gen, batch_size)
    
    train_step = round(len(train_img_path_array) / batch_size)
    test_step = round(len(test_img_path_array) / batch_size)
    
    return train_gen, test_gen, train_step, test_step

def create_model(train_gen, test_gen, train_step, test_step):
    early_stopping =  EarlyStopping(patience=2)
    model_ckpt = ModelCheckpoint('/opt/ml/checkpoints/best_model', save_best_only=True)
    
    img_inputs = layers.Input((128, 128, 3))
    img_x = layers.experimental.preprocessing.Rescaling(1/255)(img_inputs)
    img_x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(img_x)
    img_x = layers.MaxPooling2D()(img_x)
    img_x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(img_x)
    img_x = layers.MaxPooling2D()(img_x)
    img_x = layers.Flatten()(img_x)

    brf_inputs = layers.Input((7,))
    brf_x = layers.Dense(7, activation='relu')(brf_inputs)

    x = layers.Concatenate()([img_x, brf_x])
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model([img_inputs, brf_inputs], outputs)

    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    
    model.fit(train_gen, steps_per_epoch=train_step, epochs=2, 
              validation_data=test_gen, validation_steps=test_step, 
              callbacks=[early_stopping, model_ckpt])
    
    return model

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--unique_images', type=str, default=os.environ.get('SM_CHANNEL_UNIQUE_IMAGES'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == "__main__":
    args, unknown = _parse_args()
    
    img_path_arr, brief_arr, label_arr = preprocessing(args.train)
    img_path_arr = [os.path.join(args.unique_images, path) for path in img_path_arr]
    train_gen, test_gen, train_step, test_step = process_input(img_path_arr, brief_arr, label_arr)
    model = create_model(train_gen, test_gen, train_step, test_step)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
#         model.save(os.path.join(args.sm_model_dir, '000000001'))
        best_model = tf.keras.models.load_model('/opt/ml/checkpoints/best_model')
        best_model.save(os.path.join(args.sm_model_dir, 'best_model'))