import tensorflow as tf
import tensorflow.contrib as contrib


def conv(inputs, num_out, ksize, strides):
    # 输入通道数
    c = int(inputs.shape[-1])
    # xavier_initializer表示保证每层初始化的参数梯度大致相同
    W = tf.get_variable("W", shape=[ksize, ksize, c, num_out], initializer=contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[num_out], initializer=tf.constant_initializer([0.01]))
    return tf.nn.conv2d(inputs, W, strides=[1, strides, strides, 1], padding="SAME") + b


def max_pooling(inputs, ksize, strides):
    return tf.nn.max_pool(inputs, [1, ksize, ksize, 1], [1, strides, strides, 1], padding="SAME")


def relu(inputs):
    return tf.nn.relu(inputs)


def fully_conn(inputs, num_out):
    inputs = tf.layers.flatten(inputs)
    c = int(inputs.shape[-1])
    W = tf.get_variable("W", [c, num_out], initializer=contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [num_out], initializer=tf.constant_initializer([0.01]))
    return tf.matmul(inputs, W) + b


def global_avg_pooling(inputs):
    h = int(inputs.shape[1])
    w = int(inputs.shape[2])
    return tf.nn.avg_pool(inputs, [1, h, w, 1], [1, 1, 1, 1], padding='VALID')


def SE_Block(inputs):
    """squeeze and excitation block"""
    # 先获取卷积层输出的通道数
    c = int(inputs.shape[-1])
    # 压缩卷积层输出
    squeeze = tf.squeeze(global_avg_pooling(inputs), [1, 2])
    # 两个fc层
    with tf.variable_scope("FC1"):
        # 这里out_num等于c/16是压缩通道
        excitation = relu(fully_conn(squeeze, int(c / 16)))
    with tf.variable_scope("FC2"):
        # 第二个fc层处将通道恢复成c
        # 通过sigmoid计算得到各通道的权重系数
        excitation = tf.nn.sigmoid(fully_conn(excitation, c))
    # 将输出reshape成[-1, 1, 1, c],便于与之前的卷积层相乘
    excitation = tf.reshape(excitation, [-1, 1, 1, c])
    # scale代表对se_block后对卷积层每个通道上的调整参数
    #
    scale = inputs * excitation
    return scale


def batchNorm(x, train_phase, scope_bn):
    """batch normalization"""
    with tf.variable_scope(scope_bn):
        # beta和gamma是在bn计算后对每个输出神经元进行拟合的两个参数
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), trainable=True, name='beta')
        # gamma相当于偏置项
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), trainable=True, name='gamma')
        # 该函数返回两个张量，均值和方差，第二个参数代表在哪个维度上面求解
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        # 滑动更新参数，decay表示衰减速率，用于控制模型的更新速度
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                # identity 返回一个与输入张量形状和内容一致的张量
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 这里要区分是训练过程中的bn还是测试时的bn
        # 使用cond来进行判断，cond的第一个参数是一个逻辑表达式
        # 返回为true时，执行后面第一个函数，否则执行后面第二个函数
        # bn在train过程中，mean和var都是计算当前batch的
        # bn在test过程中，输入的是单个data，这时bn计算用到的mean和var可以由整个数据集的代替
        # 或者使用每次train中每个batch的均值来代替
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def haar_wavelet_block(x):
    # shape of x: Batch_size x feature_map_size
    x = tf.squeeze(global_avg_pooling(x), [1, 2])
    feature_map_size = x.shape[-1]

    length = feature_map_size // 2
    temp = tf.reshape(x, [-1, length, 2])
    a = (temp[:, :, 0] + temp[:, :, 1]) / 2
    detail = (temp[:, :, 0] - temp[:, :, 1]) / 2
    length = length // 2
    while length != 16:
        a = tf.reshape(a, [-1, length, 2])
        detail = tf.concat([(a[:, :, 0] - a[:, :, 1]) / 2, detail], axis=1)
        a = (a[:, :, 0] + a[:, :, 1]) / 2
        length = length // 2
    haar_info = tf.concat([a, detail], axis=1)
    return haar_info
