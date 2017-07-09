from model_loader import *
import os
import numpy as np
import tensorflow as tf
import pdb
from sklearn.metrics import f1_score

if __name__ == "__main__":
    DATA_DIR = "../example_datasets/PUF_4x64/"
    INPUTS_FILE = "f_4x64_100000.txt"
    LABELS_FILE = "r_4x64_100000.txt"
    # DATA_DIR = "../example_datasets/bkp/"
    # INPUTS_FILE = "f_4x64_16384.txt"
    # LABELS_FILE = "r_4x64_16384.txt"
    # DATA_DIR = "../example_datasets/PUF_4x64/alt/"
    # INPUTS_FILE = "f_4x64_50000_test.txt"
    # LABELS_FILE = "r_4x64_50000_test.txt"
    # read inputs
    test_x = []
    fp = open(os.path.join(DATA_DIR, INPUTS_FILE), "r")
    for line in fp:
        line = line.split()
        test_x.append([int(x) for x in line])
    test_x = np.array(test_x)
    fp.close()
    # read labels
    test_y = []
    fp = open(os.path.join(DATA_DIR, LABELS_FILE), "r")
    for line in fp:
        line = line.split()
        # change -1 into zeros for easier evaluation
        test_y.append([int(x) > 0 for x in line])
    fp.close()
    test_y = np.array(test_y)
    # load trained model
    loader = ModelLoader("./PUF/24-06-2017-20-11-21/generation_5000/")
    feed_dict = loader.feed_dict
    tf.import_graph_def(loader.graph_def, name="")
    with tf.Session() as sess:
        feed_dict["inputs:0"] = test_x
        output = sess.run(["TanhActivation1_out:0"], feed_dict=feed_dict)
        class_output = output[0] > 0
        result = class_output == test_y
        print("Test accuracy: ", np.mean(result))
        print("F1 score: ", f1_score(test_y, class_output))
