import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform, Constant, GlorotUniform
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class FM(Layer):
    def __init__(self, lat_dim=None, **kwargs):
        """
        A Feature Mask module (callable Keras layer obejct).
        :param lat_dim: The number of hidden neurons in the FM-module.
        :param regularizer: The regularization term of the FM-module (not necessary).
        :param kwargs: The name of this layer.
        """
        super(FM, self).__init__(**kwargs)
        self.lat_dim = lat_dim
        self.regularizer = None
        self.constraint = None
        self.num_fea = None
        self.w1, self.b1, self.w2, self.b2 = [None]*4

    def build(self, input_shape):
        super(FM, self).build(input_shape)
        self.num_fea = input_shape[1]

        self.w1 = self.add_weight(name='w1',
                                  shape=(self.num_fea, self.lat_dim),
                                  initializer=GlorotUniform(),
                                  regularizer=self.regularizer,
                                  constraint=self.constraint,
                                  trainable=True)
        self.b1 = self.add_weight(name='b1',
                                  shape=(self.lat_dim,),
                                  initializer=Constant(0),
                                  regularizer=self.regularizer,
                                  constraint=self.constraint,
                                  trainable=True)
        self.w2 = self.add_weight(name='w2',
                                  shape=(self.lat_dim, self.num_fea),
                                  initializer=RandomUniform(minval=0.9, maxval=1.1),
                                  regularizer=self.regularizer,
                                  constraint=self.constraint,
                                  trainable=True)
        self.b2 = self.add_weight(name='b2',
                                  shape=(self.num_fea,),
                                  initializer=Constant(0),
                                  regularizer=self.regularizer,
                                  constraint=self.constraint,
                                  trainable=True)

    def call(self, inputs):
        x_encoded = tf.nn.tanh(tf.matmul(inputs, self.w1) + self.b1)
        # Encoder: feature dimensions -> latent dimensions
        z = tf.matmul(x_encoded, self.w2) + self.b2
        # Decoder: latent dimensions -> feature dimensions
        # Nonlinear Transformation
        z_bar = tf.reduce_mean(z, axis=0)
        # Batchwise Attentnuation
        m = tf.nn.softmax(z_bar)
        # Feature Mask Normalization
        return tf.multiply(inputs, m)
        #返回importance scores与输入特征的点乘

    def get_support(self, inputs):
        x_encoded = tf.nn.tanh(tf.matmul(K.constant(inputs), self.w1) + self.b1)
        z = tf.matmul(x_encoded, self.w2) + self.b2
        z_bar = tf.reduce_mean(z, axis=0)
        m = tf.nn.softmax(z_bar)
        return K.get_value(m)
        #返回importance scores
