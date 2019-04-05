import numpy as np
import tensorflow as tf
from utils import load_data, random_read_batch
from net import network
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
EPSILON = 1e-10
TRAINPATH = '../datasets/CalliData/trainData/'
TESTPATH = '../datasets/CalliData/testData/'


def train():
    learning_rate = tf.placeholder("float")
    inputs = tf.placeholder("float", shape=[None, 64, 64, 1])
    labels = tf.placeholder("float", shape=[None, 4])
    is_training = tf.placeholder("bool")
    logits, prediction = network(inputs, is_training)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    loss = -tf.reduce_sum(labels * tf.log(prediction + EPSILON)) + tf.add_n(
        [tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * WEIGHT_DECAY
    Opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    traindata, trainlabels = load_data(TRAINPATH)
    testdata, testlabels = load_data(TESTPATH)
    traindata = np.reshape(traindata, [2400, 64, 64, 1]) / 127.5 - 1.0
    testdata = np.reshape(testdata, [800, 64, 64, 1]) / 127.5 - 1.0

    max_test_acc = 0
    loss_list = []
    acc_list = []
    for i in range(11000):
        batch_data, batch_label = random_read_batch(traindata, trainlabels, BATCH_SIZE)
    # train Op
    sess.run(Opt, feed_dict={inputs: batch_data, labels: batch_label,
                             is_training: True, learning_rate: LEARNING_RATE})
    if i % 20 == 0:
        [LOSS, TRAIN_ACCURACY] = sess.run([loss, accuracy], feed_dict={inputs: batch_data, labels: batch_label,
                                                                       is_training: False,
                                                                       learning_rate: LEARNING_RATE})
        loss_list.append(LOSS)
        TEST_ACCURACY = 0
        # 测试时每次测试数量为50个
        for j in range(16):
            TEST_ACCURACY += sess.run(accuracy, feed_dict={inputs: testdata[j * 50: j * 50 + 50],
                                                           labels: testlabels[j * 50: j * 50 + 50],
                                                           is_training: False,
                                                           learning_rate: LEARNING_RATE})
        # 计算测试平均acc
        TEST_ACCURACY /= 16
        acc_list.append(TEST_ACCURACY)
        if TEST_ACCURACY > max_test_acc:
            max_test_acc = TEST_ACCURACY
        print("step: %d, loss: %4g, train acc: %4g, test acc: %4g, max testacc: %4g"
              % (i, LOSS, TRAIN_ACCURACY, TEST_ACCURACY, max_test_acc))


if __name__ == "__main__":
    train()
