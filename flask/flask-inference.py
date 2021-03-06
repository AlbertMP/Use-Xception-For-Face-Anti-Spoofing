import numpy as np
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.models import load_model

# flask configuration
from flask import request, Flask, jsonify, render_template
from matplotlib import pyplot as plt

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# optimize the utilization of 3090 GPU accelerator
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


import PIL.Image as Image
import cv2


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/xception')
def xception():
    return render_template('xception.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/inference/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 使用request上传文件，其中‘file'表示前端表单的Key值；也可以使用request.files['file']
        f = request.files.get('file')

        # print(request.files)
        # print(f)
        # print(type(f))

        # 判断是否上传成功
        if f is None:
            return jsonify({"Code": "401", "Info": "no file uploaded"})
        # 检查文件后缀名是否是图片文件
        if not allow_file(f.filename):
            return jsonify({"Code": "402", "Info": "file not supported, support jpg, jpeg, png"})
        # read img file as numpy array, remember to
        np_img = plt.imread(f) * 255
        # delete alpha channel of image array
        np_img_del = np_img[:, :, :3]
        x = image.img_to_array(np_img_del)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # load model to predict & load class names
        with open("../classes.txt", 'r') as f:
            classes = list(map(lambda x: x.strip(), f.readlines()))
        model = load_model("../result-balanced-4w_10_100_180_16_1e-3_1e-4_2/model_fine_final.h5")
        pred = model.predict(x)[0]
        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])
        for i in range(2):
            (class_name, prob) = result[i]
            print("Top %d ====================" % (i + 1))
            print("Class name: %s" % (class_name))
            print("Probability: %.2f%%" % (prob))
            # prob = '%.2f'% prob
            # print(prob)
        result = dict(result)
        print(result)
    return jsonify({"Code": "200", "Info": result})


# 检查文件后缀名是否是图片文件
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
