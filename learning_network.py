from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout, Lambda, Softmax
from tensorflow.keras.models import Model


# ---------- Models ----------
def dense_block(x=None, f=None, reg=None, use_bn=False):
    """
    A compact block for a dense layer.
    :param x: The input layer which should be a Keras Layer object.
    :param f: The number of neurons of this dense layer.
    :param reg: The regularization term for this layer, default=None.
    :param use_bn: Whether use batch-normalization, default=False.
    :return: A dense block (a callable layer object).
    """
    x = Dense(units=f, activation='linear', use_bias=not use_bn, kernel_regularizer=reg)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.02)(x)
    return x
    #返回一个可调用的深度层


def build_learning_net(method=None, input_shape=None, task=None, output_shape=None,
                       num_target_fea=None, reg=None, ln_dim=None):
    """
    Build the learning network.
    :param method: The name of the variable seleciton method.
    特征选择方法名: FM/BFM
    :param input_shape: The shape of the input data, not taking the batch size into consideration.
    :param task: The name of the learning task, e.g. regression or classification.
    任务名称: regression/ multihead/ classification
    :param output_shape: The number of output layer neurons.
    :param num_target_fea: The number of features to be selected, only necessary for the concrete autoencoder.
    :param reg: The regularizaiton term for the learning network, default=None.
    :param ln_dim: The number of the hidden neurons in the first hidden layer.
    :return: A Keras model for the learning network.
    """

    use_bn = False  # not use batch-normalization
    input_layer = Input(shape=input_shape)
    x = dense_block(x=input_layer, f=ln_dim, reg=reg, use_bn=use_bn)

    if task in ['regression']:
        x = dense_block(x=x, f=ln_dim//2, reg=reg, use_bn=use_bn)
        x = Dropout(0.3)(x)
        top_layer = Dense(units=output_shape, activation='linear', name='regression')(x)

    elif task in ['multihead']:
        x = dense_block(x=x, f=ln_dim//2, reg=reg, use_bn=use_bn)
        x = Dropout(0.3)(x)
        top_layers = []
        for i in range(output_shape):
            top_layer = Dense(units=1, activation='linear', name='top%d' % (i+1))(x)
            top_layers.append(top_layer)
        mlp = Model(inputs=input_layer, outputs=top_layers, name='multihead')

    elif task in ['classification']:
        x = dense_block(x=x, f=64, reg=reg, use_bn=use_bn)
        x = Dropout(0.3)(x)
        top_layer = Dense(units=output_shape, activation='softmax', name='classification')(x)

    else:
        raise ValueError('Please enter a valid learning task...')

    mlp = Model(inputs=input_layer, outputs=top_layer, name='learning_net')
    return mlp
    #返回一个用来学习网络的模型
