import tensorflow as tf
from utils import load_data
import numpy as np

GRAHP_PATH = './save_model'
TESTPATH = './datasets/CalliData/testData/'

testdata, testlabels = load_data(TESTPATH)
testdata = np.reshape(testdata, [800, 64, 64, 1]) / 127.5 - 1.0

x_test = testdata[0].reshape([1, 64, 64, 1])
y_test = testlabels[0].reshape([1, 4])

sess = tf.Session()

# load meta graph
meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                            GRAHP_PATH)

# get signature
signature_def = meta_graph_def.signature_def
signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# get input tensor
x_input_tensor = signature.inputs['x_input'].name
y_input_tensor = signature.inputs['y_input'].name
istrain = signature.inputs['istrain_input'].name
lr = signature.inputs['lr_input'].name

# get output tensor
y_predict_tensor = signature.outputs['y_predict'].name

# get loss func
loss_op = signature.outputs['loss_func'].name

prediction, loss = sess.run([y_predict_tensor, loss_op], {
    x_input_tensor: x_test,
    y_input_tensor: y_test,
    istrain: False,
    lr: 1e-3
})

frozen_graphdef = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                               ['HW_SE_NET/fully_connected/prediction'])
tflite_model = tf.lite.toco_convert(frozen_graphdef, x_test, y_test)
open('model.tflite', 'wb').write(tflite_model)

print("predict:", np.around(prediction, 5))
