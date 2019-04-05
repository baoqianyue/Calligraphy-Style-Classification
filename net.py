from ops import *


def network(inputs, train_phase):
    with tf.variable_scope("HW_SE_NET"):
        with tf.variable_scope("conv1"):
            inputs = conv(inputs, 32, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchNorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
        with tf.variable_scope("conv2"):
            inputs = conv(inputs, 32, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchNorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
        with tf.variable_scope("conv3"):
            inputs = conv(inputs, 64, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchNorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
            # 通过se_block处理，得到每个通道上的scale
            # 将每个通道上的scale与上层卷积层输出相乘
            # 然后返回考虑了通道相关性的featuremap，然后继续向前传递
            inputs = SE_Block(inputs)
        with tf.variable_scope("conv4"):
            inputs = conv(inputs, 128, 5, 1)
            inputs = max_pooling(inputs, 3, 2)
            inputs = batchNorm(inputs, train_phase, "BN")
            inputs = relu(inputs)
            inputs = SE_Block(inputs)
        # with tf.variable_scope("haar_wavelet"):
        #     inputs = haar_wavelet_block(inputs)
        with tf.variable_scope("fully_connected"):
            # 全联接计算每个类别的输出值
            logits = fully_conn(inputs, 4)
            prediction = tf.nn.softmax(logits)
        return logits, prediction
