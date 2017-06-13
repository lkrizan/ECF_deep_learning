from model_loader import *
import pickle
import os
import numpy as np
import tensorflow as tf
import pdb

def one_hot_encode(values):
    """
    :param values: array like
    :return: ndarray len(values) x (max(values) + 1)
    """
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

if __name__ == "__main__":
    # read and preprocess the dataset
    DATA_DIR = "../CIFAR10/cifar-10-batches-py"
    img_height = 32
    img_width = 32
    num_channels = 3
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, img_height, img_width, num_channels))
    train_y = np.array(train_y, dtype=np.int32)
    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, img_height, img_width, num_channels)).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)
    data_mean = train_x.mean((0,1,2))
    data_std = train_x.std((0,1,2))
    train_x = (train_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std
    train_y_ = one_hot_encode(train_y)
    test_y_ = one_hot_encode(test_y)
    # load trained model
    loader = ModelLoader("./CIFAR10/")
    feed_dict = loader.feed_dict
    tf.import_graph_def(loader.graph_def, name="")
    with tf.Session() as sess:
        feed_dict["inputs:0"] = test_x
        output = sess.run(["FC3_out:0"], feed_dict=feed_dict)
        class_output = np.argmax(output[0], 1)
        result = class_output == np.argmax(test_y_, 1)
        print("Test accuracy: ", np.mean(result))
