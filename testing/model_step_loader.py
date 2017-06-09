import tensorflow as tf
import numpy as np

class VariableData:

    def __init__(self):
        self.name = None
        self.shape = None
        self.values = None


class ModelLoader:

    def __init__(self, folder_path):
        """
        :param folder_path:     string
        """
        self.folder_path = folder_path
        self._open_graph_def()

    def create_feed_dict(self):
        self.feed_dict = {}
        for var in self._variables:
            self.feed_dict[var.name+":0"] = var.values

    def _open_graph_def(self):
        self.graph_def = tf.GraphDef()
        with open(self.folder_path + "graph.pb", "rb") as f:
            self.graph_def.ParseFromString(f.read())

    def create_variable_data(self, folder_path=None):
        # variables - list of tuples (name, shape, values)
        self._variables = []
        if folder_path is None:
            folder_path = self.folder_path
        # states for reading file
        eStart, eName, eShape, eValues = range(0, 4)
        state = eName
        with open(folder_path + "variables.dat") as f:
            var_data = VariableData()
            for line in f:
                line = line.split()
                if state == eStart:
                    var_data = VariableData()
                    state = eName
                    continue
                elif state == eName:
                    var_data.name = line[0]
                    state = eShape
                    continue
                elif state == eShape:
                    var_data.shape = tuple([int(x) for x in line])
                    state = eValues
                    continue
                elif state == eValues:
                    var_data.values = np.array([float(x) for x in line])
                    var_data.values = var_data.values.reshape(var_data.shape)
                    self._variables.append(var_data)
                    state = eStart
                    continue
