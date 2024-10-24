import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform, Constant, GlorotUniform
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class MCFM(Layer):
    def __init__(self, lat_dim=None, dropout_rate=0.2, **kwargs):
        super(MCFM, self).__init__(**kwargs)
        self.lat_dim = lat_dim
        self.regularizer = None
        self.constraint = None
        self.num_fea = None
        self.dropout_rate = dropout_rate
        self.w1, self.b1, self.w2, self.b2 = [None]*4

    def build(self, input_shape):
        super(MCFM, self).build(input_shape)
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

    def call(self, inputs, training=False):
        x_encoded = tf.nn.tanh(tf.matmul(inputs, self.w1) + self.b1)
        if training:
            x_encoded = tf.nn.dropout(x_encoded, rate=self.dropout_rate)
        z = tf.matmul(x_encoded, self.w2) + self.b2
        if training:
            z = tf.nn.dropout(z, rate=self.dropout_rate)
        z_bar = tf.reduce_mean(z, axis=0)
        m = tf.nn.softmax(z_bar)
        return tf.multiply(inputs, m)

    def get_support(self, inputs):
        x_encoded = tf.nn.tanh(tf.matmul(K.constant(inputs), self.w1) + self.b1)
        x_encoded = tf.nn.dropout(x_encoded, rate=self.dropout_rate)
        z = tf.matmul(x_encoded, self.w2) + self.b2
        z = tf.nn.dropout(z, rate=self.dropout_rate)
        z_bar = tf.reduce_mean(z, axis=0)
        m = tf.nn.softmax(z_bar)
        return K.get_value(m)
