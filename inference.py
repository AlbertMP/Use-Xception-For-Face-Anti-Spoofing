import argparse
import numpy as np
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('classes')
parser.add_argument('image')

# optimize the utilization of 3090 GPU accelerator
import tensorflow as tf
config = tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def main(args):

    # create model
    model = load_model(args.model)

    # load class names
    classes = []
    with open(args.classes, 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))
    print(classes)

    # load an input image
    img = image.load_img(args.image, target_size=(299, 299))
    x = image.img_to_array(img)
    # print(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1])
    for i in range(2):
        (class_name, prob) = result[i]
        print("Top %d ====================" % (i + 1))
        print("Class name: %s" % (class_name))
        print("Probability: %.2f%%" % (prob))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
