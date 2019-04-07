import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.Session()

saver = tf.train.import_meta_graph('./model/cnn_model.ckpt.meta')
saver.restore(sess, './model/cnn_model.ckpt')

gd = sess.graph.as_graph_def()

for node in gd.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
converted_graph_def = graph_util.convert_variables_to_constants(sess, gd, ['predict'])
tf.train.write_graph(converted_graph_def, './model/', 'solve_norm_model.pb', as_text=False)
