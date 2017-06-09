from model_step_loader import *
import os
import numpy as np
import tensorflow as tf
import pdb

def read_dataset():
    """
    :return: (test_x, test_y) - tuple of ndarray
    """
    # NOTE: adjust this function for every dataset
    DATA_DIR = "../../../example_datasets/regression/"
    FILE_NAME = "test_dataset.txt"
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
    return test_x, test_y

def calculate_error(outputs, test_y):
    """
    :return: error/precision, whatever, float
    """
    # NOTE: change this for different datasets
    return np.mean((outputs - test_y) ** 2)

if __name__ == "__main__":
    # NOTE: copy graph.pb to root folder
    # NOTE: change this into whatever you want to load from the graph
    OUTPUT_NODE = "FC2_out:0"
    # import graph definition
    model_loader = ModelLoader("./")
    tf.import_graph_def(model_loader.graph_def, name="")
    # open dataset
    test_x, test_y = read_dataset()
    # all folders with data from batch mode
    folders = [ name for name in os.listdir("./") if os.path.isdir(os.path.join("./", name)) ]
    if "__pycache__" in folders:
        folders.remove("__pycache__")
    export_data = []
    # create tensorflow session
    sess = tf.Session()
    for folder in folders:
        nested_folders = os.listdir("./" + folder)
        # nested folders should have names like "generation_xxx"
        nested_folders = sorted(nested_folders, key=lambda name: (int(name[11:])))
        export_data.append([])
        for nested_folder in nested_folders:
            model_loader.create_variable_data("./" + folder + "/" + nested_folder + "/")
            model_loader.create_feed_dict()
            feed_dict = model_loader.feed_dict
            feed_dict["inputs:0"] = test_x
            output = sess.run([OUTPUT_NODE], feed_dict=feed_dict)
            export_data[-1].append(calculate_error(output[0], test_y))
    sess.close()
    # export as .csv
    # columns - error per generation
    # rows - error per batch run
    fp = open("results.csv", "w")
    for batch in export_data:
        fp.write(",".join([str(x) for x in batch]))
        fp.write("\n")
    fp.close()
