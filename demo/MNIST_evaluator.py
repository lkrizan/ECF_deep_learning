from model_loader import *
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    DATA_DIR = '../MNIST'
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels
    # divide with 255 to get the same normalization effect used in training
    train_x /= 255.0
    valid_x /= 255.0
    test_x /= 255.0
    # load model
    loader = ModelLoader("./MNIST/")
    feed_dict = loader.feed_dict
    tf.import_graph_def(loader.graph_def, name="")
    with tf.Session() as sess:
        feed_dict["inputs:0"] = test_x
        output = sess.run(["FC3_out:0"], feed_dict=feed_dict)
        class_output = np.argmax(output[0], 1)
        result = class_output == np.argmax(test_y, 1)
        print("Test accuracy: ", np.mean(result))
