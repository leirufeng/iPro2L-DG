
from keras.layers import Dense, Activation, Conv2D, BatchNormalization, Reshape
from keras.layers import Layer

class GAMAttention( Layer ):
    def __init__(self, filters, rate=4, **kwargs):
        self.input_dims = int( filters )
        self.out_dims = int( filters )
        self.reduce_dims = int( filters / rate )
        self.rate = rate

        # channel attention
        self.channel_linear_reduce = Dense( self.reduce_dims )
        self.channel_activation = Activation( 'relu' )
        self.channel_linear = Dense( self.input_dims )

        # spatial attention
        self.spatial_con_reduce = Conv2D( filters=self.reduce_dims, kernel_size=7, padding='same', name='spatial_con_reduce' )
        self.spatial_bn_reduce = BatchNormalization()
        self.spatial_activation = Activation( 'relu' )
        self.spatial_con = Conv2D( filters=self.out_dims, kernel_size=7, padding='same', name='spatial_con' )
        self.spatial_bn = BatchNormalization()

        self.in_shape = None
        super(GAMAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dims": self.input_dims,
            "out_dims": self.out_dims,
            "reduce_dims": self.reduce_dims,
            "rate": self.rate,
            "in_shape": self.in_shape
        })
        return config

    def build(self, input_shape):
        assert input_shape[-1] == self.out_dims, 'input filters must equal to input of channel'
        self.in_shape = input_shape
        # channel attention
        self.channel_linear_reduce.build( (input_shape[0], input_shape[1] * input_shape[2], input_shape[3]) )
        self._trainable_weights += self.channel_linear_reduce.trainable_weights
        self.channel_linear.build( (input_shape[0], input_shape[1] * input_shape[2], self.reduce_dims) )
        self._trainable_weights += self.channel_linear.trainable_weights

        # spatial attention
        self.spatial_con_reduce.build( input_shape )
        self._trainable_weights += self.spatial_con_reduce.trainable_weights
        self.spatial_bn_reduce.build( (input_shape[0], input_shape[1], input_shape[2], self.reduce_dims) )
        self._trainable_weights += self.spatial_bn_reduce.trainable_weights
        self.spatial_con.build( (input_shape[0], input_shape[1], input_shape[2], self.reduce_dims) )
        self._trainable_weights += self.spatial_con.trainable_weights
        self.spatial_bn.build( input_shape )
        self._trainable_weights += self.spatial_bn.trainable_weights

        super( GAMAttention, self ).build( input_shape )  # Be sure to call this at the end

    def call(self, f1, **kwargs):
        # channel attention
        tmp = Reshape( (-1, self.input_dims) )( f1 )
        tmp = self.channel_linear_reduce( tmp )
        tmp = self.channel_activation( tmp )
        tmp = self.channel_linear( tmp )
        mc = Reshape( self.in_shape[1:] )( tmp )
        f2 = mc * f1

        # spatial attention
        tmp = self.spatial_con_reduce(f2)
        tmp = self.spatial_bn_reduce(tmp)
        tmp = self.spatial_activation(tmp)
        tmp = self.spatial_con(tmp)
        ms = self.spatial_bn(tmp)
        f3 = ms * f2

        return f3
