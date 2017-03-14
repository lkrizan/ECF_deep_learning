import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    w = tf.Variable(tf.constant([[1.0, 1, 1]]), name='W')
    b = tf.constant([[1.0, 2, 3]], name='b')
    result = tf.reduce_sum(tf.sub(w, b) ** 2, name='R')

    sess.run(tf.initialize_all_variables())
    vars_ = tf.trainable_variables()
    names = [v.name for v in vars_]
    print(w.eval())
    print(b.eval())
    print(result.eval())
    print(w.get_shape())
    print(names)
    tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb',
                        as_text=False)
