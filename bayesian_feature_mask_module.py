import tensorflow as tf
from tensorflow.keras.layers import Lambda, Softmax
from tensorflow.keras.models import Model
import tensorflow_probability as tfp


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

# ---------- Different Variable Selectors ----------
class BFM(Model):
    def __init__(self, input_dim=None, lat_dim=None, kl_weight=None, **kwargs):
        super(BFM, self).__init__(**kwargs)
        self.encoder_layer = tfp.layers.DenseVariational(units=lat_dim,
                                                         make_prior_fn=prior,
                                                         make_posterior_fn=posterior,
                                                         kl_weight=kl_weight,
                                                         activation="tanh")
        self.decoder_layer = tfp.layers.DenseVariational(units=input_dim,
                                                         make_prior_fn=prior,
                                                         make_posterior_fn=posterior,
                                                         kl_weight=kl_weight,
                                                         activation="linear")

    def call(self, inputs):
        x_lat = self.encoder_layer(inputs)
        # Encoder: feature dimensions -> latent dimensions
        z_primitive = self.decoder_layer(x_lat)
        # Decoder: latent dimensions -> feature dimensions
        # Nonlinear Transformation
        z_bar = Lambda(lambda x: tf.reduce_mean(x, axis=0, keepdims=True))(z_primitive)
        # Batchwise Attentnuation
        m = Softmax()(z_bar)
        # Feature Mask Normalization
        weighted_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([inputs, m])
        return weighted_x
        #返回importance scores与输入特征的点乘

    def get_support(self, inputs):
        x_lat = self.encoder_layer(tf.constant(inputs))
        z_primitive = self.decoder_layer(x_lat)
        z_bar = Lambda(lambda x: tf.reduce_mean(x, axis=0))(z_primitive)
        m = Softmax()(z_bar)
        return m.numpy()
        #返回importance scores
