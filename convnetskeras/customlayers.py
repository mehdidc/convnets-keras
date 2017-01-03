from keras.layers.core import  Lambda, Merge
from keras.layers.convolutional import Convolution2D
from keras import backend as K

from keras.engine import Layer

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)

class SplitTensor(Layer):

    def __init__(self, axis=1, ratio_split=1, id_split=0, **kwargs):
        super(SplitTensor, self).__init__(**kwargs)
        self.axis = axis
        self.ratio_split = ratio_split
        self.id_split = id_split

    def call(self, X, mask=None):
        axis = self.axis
        ratio_split = self.ratio_split
        id_split = self.id_split

        div = X.shape[axis] // ratio_split
        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = output_shape[self.axis] // self.ratio_split
        return tuple(output_shape)

    def get_config(self):
        config = {'axis': self.axis, 'ratio_split': self.ratio_split, 'id_split': self.id_split}
        base_config = super(SplitTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

splittensor = SplitTensor

def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        return Merge([
            Convolution2D(nb_filter//n_group,nb_row,nb_col)(
                splittensor(axis=1,
                            ratio_split=n_group,
                            id_split=i)(input))
            for i in range(n_group)
        ],mode='concat',concat_axis=1)

    return f


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape

custom_objects = {
    'SplitTensor': SplitTensor,
    'Softmax4D': Softmax4D,
}
