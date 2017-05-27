from model_loader import *
import os
import numpy as np
import tensorflow as tf
import pdb
from sklearn.metrics import f1_score

if __name__ == "__main__":
    CL_THRESHOLD = 0.25
    # read and preprocess the dataset
    DATA_DIR = "../example_datasets/one_class/"
    FILE_NAME = "oneclass_realistic_small_test_all_numeric.txt"
    test_x = []
    test_y = []
    fp = open(os.path.join(DATA_DIR, FILE_NAME), "r")
    # first line contains data about shapes
    line = fp.readline()
    # should be delimited by ','
    line = line.split(',')
    inputs_shape = int(line[0])
    outputs_shape = int(line[1])
    for line in fp:
        line = line.split(',')
        test_x.append([float(x) for x in line[:inputs_shape]])
        test_y.append([int(x) for x in line[inputs_shape:]])
    fp.close()
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    # load avg train error
    fp = open("./OneClass/info.dat")
    assert fp.readline() == "Avg train error\n"
    avg_error = float(fp.readline())
    fp.close()
    # load trained model
    loader = ModelLoader("./OneClass/")
    feed_dict = loader.feed_dict
    tf.import_graph_def(loader.graph_def, name="")
    with tf.Session() as sess:
        output = []
        for i in range(len(test_x)):
            feed_dict["inputs:0"] = [test_x[i]]
            feed_dict["outputs:0"] = [test_x[i]]
            output += sess.run(["loss:0"], feed_dict=feed_dict)
        class_output = np.array(output) > CL_THRESHOLD * avg_error
        result = class_output == test_y
        print("Test accuracy: ", np.mean(result))
        print("F1 score: ", f1_score(test_y, class_output))
