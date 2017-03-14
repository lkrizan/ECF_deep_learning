import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, 2], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    w1 = tf.placeholder(tf.float32, [4, 2], name='w1')
    b1 = tf.placeholder(tf.float32, [4], name='b1')
    w2 = tf.placeholder(tf.float32, [1, 4], name='w2')
    b2 = tf.placeholder(tf.float32, [1], name='b2')
    first_layer = tf.add(tf.matmul(x, tf.transpose(w1)),b1)
    f1_output = tf.nn.sigmoid(first_layer)
    second_layer = tf.add(tf.matmul(f1_output, tf.transpose(w2)), b2)
    loss = tf.reduce_mean((second_layer - y) ** 2, name='loss')
    sess.run(tf.initialize_all_variables())
    tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb',
                        as_text=False)
