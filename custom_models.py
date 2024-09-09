import tensorflow as tf
from keras import layers, models, ops
import custom_layers


class Encoder(models.Model):
    def __init__(self, widths, kernel_size, activation, use_bn=True, depth=2, latent_filters=4, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.widths = widths
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.depth = depth
        self.latent_filters = latent_filters

    def build(self, input_shape):
        self.downblocks = []

        for filters in self.widths:
            downblock_i = custom_layers.DownBlock(filters=filters, kernel_size=self.kernel_size, activation=self.activation, use_bn=self.use_bn, depth=self.depth)
            self.downblocks.append(downblock_i)

        self.z_mean = layers.Conv2D(self.latent_filters, self.kernel_size, padding='same', activation='linear')
        self.z_log_var = layers.Conv2D(self.latent_filters, self.kernel_size, padding='same', activation='linear')

    def call(self, x):
        for downblock in self.downblocks:
            x = downblock(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        return z_mean, z_log_var
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update(
            {
                "widths": self.widths,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "use_bn": self.use_bn,
                "depth": self.depth,
                "latent_filters": self.latent_filters
            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


class Decoder(models.Model):
    def __init__(self, widths, kernel_size, activation, use_bn=True, depth=2, output_channels=3, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.widths = widths
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.depth = depth
        self.output_channels = output_channels

    def build(self, input_shape):
        self.upblocks = []

        for filters in self.widths:
            upblock_i = custom_layers.UpBlock(filters=filters, kernel_size=self.kernel_size, activation=self.activation, use_bn=self.use_bn, depth=self.depth)
            self.upblocks.append(upblock_i)

        self.outputs = layers.Conv2D(self.output_channels, self.kernel_size, padding='same', activation='sigmoid')

    def call(self, x):
        for upblock in self.upblocks:
            x = upblock(x)

        x = self.outputs(x)

        return x
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update(
            {
                "widths": self.widths,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "use_bn": self.use_bn,
                "depth": self.depth,
                "output_channels": self.output_channels
            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class VAE(models.Model):
    def __init__(self, encoder_widths, decoder_widths, kernel_size, activation, use_bn=True, depth=2, latent_filters=4, output_channels=3, kl_beta=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = Encoder(encoder_widths, kernel_size, activation, use_bn, depth, latent_filters)
        self.decoder = Decoder(decoder_widths, kernel_size, activation, use_bn, depth, output_channels)
        self.kl_beta = kl_beta

        if len(encoder_widths) != len(decoder_widths):
            raise ValueError(f"the number of widths in the encoder and decoder should be equal. found {len(encoder_widths)} encoder_widths and {len(decoder_widths)} decoder_widths")
        
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]
    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z_latent = self.sample(z_mean, z_log_var)
        return self.decoder(z_latent)
    
    
    def sample(self, z_mean, z_log_var, eps=None):
        if eps == None:
            eps = tf.random.normal(shape=ops.shape(z_mean))

        stddev = tf.math.exp(z_log_var / 2)

        return z_mean + eps * stddev
    
    def train_step(self, x):
        dims = tf.shape(x)[1] * tf.shape(x)[2] * tf.shape(x)[3]
        dims = tf.cast(dims, dtype=tf.float32)
        
        with tf.GradientTape() as tape:

            z_mean, z_log_var = self.encoder(x)
            z_latent = self.sample(z_mean, z_log_var)
            x_reconstructed = self.decoder(z_latent)

            reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_reconstructed) * dims
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=1
                )
            ) 

            total_loss = reconstruction_loss + self.kl_beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}
    
    def save_model(self, epoch=None, logs=None):
        self.encoder.save_weights("vae_encoder.weights.h5")
        self.decoder.save_weights("vae_decoder.weights.h5")
        print(f"\n Weights saved")
    

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update(
            {
                "encoder": self.encoder,
                "decoder": self.decoder,
                "kl_beta": self.kl_beta
            }
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)
