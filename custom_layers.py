from keras import layers


class ResnetBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation="linear", use_bn=True):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn

    def build(self, input_shape):

        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act = layers.Activation(activation=self.activation)

        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        if self.use_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            y = self.conv2(x)
            y = self.bn2(y)
            
            return x + y
        
        else:
            x = self.conv1(x)
            x = self.act(x)
            y = self.conv2(x)

            return x + y
        
    def get_config(self):
        config = super(ResnetBlock, self).get_config()
        config.update({
            "filters": self.filters, 
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_bn": self.use_bn
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

class DownBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation="linear", use_bn=True, depth=2):
        super(DownBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.depth = depth

    def build(self, input_shape):
        self.resnets = []

        for _ in range(self.depth):
            resnet_block_i = ResnetBlock(self.filters, self.kernel_size, self.activation, self.use_bn)
            self.resnets.append(resnet_block_i)

        self.pooling = layers.MaxPooling2D()

    def call(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        x = self.pooling(x)

        return x

    def get_config(self):
        config = super(DownBlock, self).get_config()
        config.update({
            "filters": self.filters, 
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_bn": self.use_bn,
            "depth": self.depth
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class UpBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation="linear", use_bn=True, depth=2):
        super(UpBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bn = use_bn
        self.depth = depth

    def build(self, input_shape):
        self.resnets = []

        for _ in range(self.depth):
            resnet_block_i = ResnetBlock(self.filters, self.kernel_size, self.activation, self.use_bn)
            self.resnets.append(resnet_block_i)

        self.upsampling = layers.UpSampling2D(size=(2, 2))

    def call(self, x):
        resnet = self.resnets[0]
        x = resnet(x)
        x = self.upsampling(x)
        for resnet in self.resnets[1:]:
            x = resnet(x)

        return x


    def get_config(self):
        config = super(UpBlock, self).get_config()
        config.update({
            "filters": self.filters, 
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_bn": self.use_bn,
            "depth": self.depth
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
