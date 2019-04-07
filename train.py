import numpy as np
import tensorflow as tf
from utils import load_data, random_read_batch
from net import network1
import os
from tensorflow.python.framework import graph_util

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10
TRAINPATH = './datasets/CalliData/trainData/'
TESTPATH = './datasets/CalliData/testData/'
LOG_PATH = './logs/'
SAVE_PATH = './model/'


def train():
    inputs = tf.placeholder("float", shape=[None, 64, 64, 1], name="inputs")
    labels = tf.placeholder("float", shape=[None, 4], name="labels")
    is_training = tf.placeholder("bool", name="istrain")
    logits, prediction = network1(inputs, is_training)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope("loss"):
        loss = -tf.reduce_sum(labels * tf.log(prediction + EPSILON)) + tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * WEIGHT_DECAY
        # tf.summary.scalar('loss', loss)

    Opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 构建signature_def 对象
    # signature = tf.saved_model.signature_def_utils.build_signature_def(
    #     inputs={
    #         'x_input': tf.saved_model.utils.build_tensor_info(inputs),
    #         'y_input': tf.saved_model.utils.build_tensor_info(labels),
    #         'istrain_input': tf.saved_model.utils.build_tensor_info(is_training),
    #         'lr_input': tf.saved_model.utils.build_tensor_info(learning_rate)
    #     },
    #     outputs={
    #         'y_predict': tf.saved_model.utils.build_tensor_info(prediction),
    #         'loss_func': tf.saved_model.utils.build_tensor_info(loss)
    #     },
    #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    # )

    # merged = tf.summary.merge_all()

    # writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    traindata, trainlabels = load_data(TRAINPATH)
    testdata, testlabels = load_data(TESTPATH)
    traindata = np.reshape(traindata, [2400, 64, 64, 1]) / 127.5 - 1.0
    testdata = np.reshape(testdata, [800, 64, 64, 1]) / 127.5 - 1.0

    # print(np.around(traindata[342, 32:38, 32:38, :], 4))

    max_test_acc = 0
    loss_list = []
    acc_list = []
    for i in range(2):
        batch_data, batch_label = random_read_batch(traindata, trainlabels, BATCH_SIZE)
        # train Op
        sess.run(Opt, feed_dict={inputs: batch_data, labels: batch_label,
                                 is_training: True})
        if i % 1 == 0:
            [LOSS, TRAIN_ACCURACY, prob] = sess.run([loss, accuracy, prediction],
                                                    feed_dict={inputs: batch_data, labels: batch_label,
                                                               is_training: False})
            loss_list.append(LOSS)
            print(prob)
            # if os.path.exists(SAVE_PATH):
            #     shutil.rmtree(SAVE_PATH)

            # builder = tf.saved_model.builder.SavedModelBuilder(SAVE_PATH)
            # builder.add_meta_graph_and_variables(sess,
            #                                      [tf.saved_model.tag_constants.SERVING],
            #                                      {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            #                                           signature})
            # builder.save()
            # log_res = sess.run(merged, feed_dict={inputs: batch_data, labels: batch_label,
            #                                       is_training: False, learning_rate: LEARNING_RATE})
            # writer.add_summary(log_res, i)

            TEST_ACCURACY = 0
            # 测试时每次测试数量为50个
            for j in range(16):
                TEST_ACCURACY += sess.run(accuracy, feed_dict={inputs: testdata[j * 50: j * 50 + 50],
                                                               labels: testlabels[j * 50: j * 50 + 50],
                                                               is_training: False})
            # 计算测试平均acc
            TEST_ACCURACY /= 16
            acc_list.append(TEST_ACCURACY)
            if TEST_ACCURACY > max_test_acc:
                max_test_acc = TEST_ACCURACY
            print("step: %d, loss: %4g, train acc: %4g, test acc: %4g, max testacc: %4g"
                  % (i, LOSS, TRAIN_ACCURACY, TEST_ACCURACY, max_test_acc))

    saver.save(sess, os.path.join(SAVE_PATH, 'cnn_model.ckpt'))
    # 固化变量
    output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['prediction'])
    with tf.gfile.FastGFile(os.path.join(SAVE_PATH, 'mobile_model.pb'), mode='wb') as f:
        f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    train()
