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
            prediction = tf.nn.softmax(logits, name='prediction')
        return logits, prediction


def network1(inputs, train_phase):
    conv1 = conv(inputs, 32, 5, 1)
    pool1 = max_pooling(conv1, 3, 2)
    bn1 = batchNorm(pool1, train_phase, "BN")
    relu1 = relu(bn1)

    conv2 = conv(relu1, 32, 5, 1)
    pool2 = max_pooling(conv2, 3, 2)
    bn2 = batchNorm(pool2, train_phase, "BN")
    relu2 = relu(bn2)

    conv3 = conv(relu2, 64, 5, 1)
    pool3 = max_pooling(conv3, 3, 2)
    bn3 = batchNorm(pool3, train_phase, "BN")
    relu3 = relu(bn3)
    # 通过se_block处理，得到每个通道上的scale
    # 将每个通道上的scale与上层卷积层输出相乘
    # 然后返回考虑了通道相关性的featuremap，然后继续向前传递
    se1 = SE_Block(relu3)

    conv4 = conv(se1, 128, 5, 1)
    pool4 = max_pooling(conv4, 3, 2)
    bn4 = batchNorm(pool4, train_phase, "BN")
    relu4 = relu(bn4)
    se2 = SE_Block(relu4)
    # with tf.variable_scope("haar_wavelet"):
    #     inputs = haar_wavelet_block(inputs)

    # 全联接计算每个类别的输出值
    logits = fully_conn(se2, 4)
    prediction = tf.nn.softmax(logits, name='prediction')

    return logits, prediction
