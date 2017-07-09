from model_loader import *
import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":
    DATA_DIR = "../example_datasets/regression_f5/"
    FILE_NAME = "test_dataset_f5.txt"
    # read inputs and outputs
    test_x = []
    test_y = []
    fp = open(os.path.join(DATA_DIR, FILE_NAME), "r")
    for line in fp:
        line = line.split()
        test_x.append([float(x) for x in line[:2]])
        test_y.append([float(y) for y in line[2:]])
    fp.close()
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # load trained model
    loader = ModelLoader("./regression/22-06-2017-12-28-48/generation_1000/")
    feed_dict = loader.feed_dict
    tf.import_graph_def(loader.graph_def, name="")
    with tf.Session() as sess:
        feed_dict["inputs:0"] = test_x
        output = sess.run(["FC2_out:0"], feed_dict=feed_dict)
        squared_error = (output[0] - test_y) ** 2
        print("Test mse: ", np.mean(squared_error))
