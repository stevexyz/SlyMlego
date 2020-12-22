
# https://raw.githubusercontent.com/titu1994/keras-squeeze-excite-network/master/se.py

from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras import backend as K

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        - input: input tensor
        - ratio: ...
    Returns:
        - a keras tensor
    References:
        - [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    assert K.image_data_format() == "channels_last"
    filters = input._keras_shape[-1] # number of channels
    se = GlobalAveragePooling2D() \
                (input)
    se = Reshape((1, 1, filters)) \
                (se)
    se = Dense(
            filters//ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False) \
                (se)
    se = Dense(
            filters,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False)
                (se)
    se = multiply([input, se])
    return se

def spatial_squeeze_excite_block(input):
    ''' Create a spatial squeeze-excite block
    Args:
        - input: input tensor
    Returns:
        - a keras tensor
    References
        - [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''
    sse = Conv2D(
            1,
            (1, 1),
            activation='sigmoid',
            use_bias=False,
            kernel_initializer='he_normal') \
                (input)
    sse = multiply([input, sse])
    return sse

def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block
    Args:
        - input: input tensor
        - ratio: ...
    Returns:
        - a keras tensor
    References
        - [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
        - [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''
    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)
    csse = add([cse, sse])
    return csse

####################################################################################

# https://github.com/arthurdouillard/keras-squeeze_and_excitation_network

from keras.models import Model
from keras.layers import Conv2D, Input, Dense, Flatten, MaxPool2D
from keras.layers import Activation, BatchNormalization

from squeeze_excite import SqueezeExcite

def alexnet_block(x, filters, kernel_size, se):
    y = Conv2D(filters, kernel_size, padding='same')(x)
    if se:
        y = SqueezeExcite(y, ratio=16)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = MaxPool2D(pool_size=(2, 2), padding='same')(y)
    return y


def dense_block(x, size, bn=True):
    y = Dense(size)(x)
    if bn:
        y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y


def SeAlexNet(nb_class, input_shape=(227, 227, 3), include_top=True, weights=None,
              batch_norm=True, se=True):
    """AlexNet without the splitted stream."""

    img_input = Input(shape=input_shape)

    x = alexnet_block(img_input, 64, (11, 11), se=se)

    for i, (filters, kernel_size) in enumerate([(128, (7, 7)), (192, (3, 3))]):
        x = alexnet_block(x, filters, kernel_size, se=se)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = dense_block(x, 4096, bn=batch_norm)
        x = dense_block(x, 4096, bn=batch_norm)
        x = Dense(nb_class, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    if weights:
        print('Loading')
        model.load_weights(weights)

    return model

####################################################################################

# https://missinglink.ai/guides/keras/keras-resnet-building-training-scaling-residual-nets-keras/

def resnext_residual_block(y, nb_channels_in, nb_channels_out, cardinality=4):
    shortcut = y
    if cardinality == 1:
        y = Conv2D(
                nb_channels_in,
                kernel_size=(5, 5),
                strides=(1,1),
                padding='same',
                use_bias=False) \
                    (y)
    else:
        assert not nb_channels_in % cardinality
        groups = []
        for j in range(cardinality):
            group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(
                Conv2D(
                    nb_channels_in // cardinality, 
                    kernel_size=(5, 5), 
                    strides=(1,1), 
                    padding='same', 
                    use_bias=False) \
                        (group))
        y = concatenate(groups)
    y = BatchNormalization(axis=-1) (y)
    y = ELU() (y)
    y = add([shortcut, y])
    return y

####################################################################################

# https://arxiv.org/pdf/1611.09326v2.pdf
# https://github.com/asprenger/keras_fc_densenet/blob/master/keras_fc_densenet.py

"""
Build a FC-DenseNet model as described in https://arxiv.org/abs/1611.09326.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Reshape, Flatten, Embedding, Dropout, Input
from tensorflow.keras.layers import BatchNormalization, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, concatenate, Activation
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2


def _channel_dim(data_format):
    return 1 if data_format == 'channels_first' else -1

def _conv_block(x, nb_filter, bn_momentum, dropout_rate=None, block_prefix='ConvBlock', 
               data_format='channels_last'):
    """
    Adds a single layer (conv block) of a dense block. It is composed of a 
    batch normalization, a relu, a convolution and a dropout layer.
    
    Args
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers
        dropout_rate: dropout rate
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'
        
    Return:
        output tensor
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

        # FC-DenseNet paper does not say anything about stride in the conv block, assume default (1,1)
        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False,
                   data_format=data_format)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x



def _dense_block(x, nb_layers, nb_filter, growth_rate, bn_momentum, dropout_rate=None, grow_nb_filters=True, 
                  return_concat_list=False, block_prefix='DenseBlock', data_format='channels_last'):
    """
    Adds a dense block. The input and output of each conv block is 
    concatenated and used as input for the next conv block. The result
    is the concatenated outputs of all conv blocks. In addition
    the first element of the result is the input tensor `x`, this 
    works as a shortcut connection.
    
    The block leaves height and width of the input unchanged and
    adds `nb_layers` * `growth_rate` feature maps.
    
    Args:
        x: input tensor
        nb_layers: the number of conv_blocks in the dense block
        nb_filter: filter count that will be incremented for each conv block            
        growth_rate: growth rate of the dense block, this is the number
            of filters in each conv block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers    
        dropout_rate: dropout rate
        grow_nb_filters: flag if nb_filters should be updated
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'
    
    Return:
        x: tensor concatenated from [x, cb_1_out, ..., cb_n_out]
        nb_filter: updated nb_filters
        x_list: list [x, cb_1_out, ..., cb_n_out]
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x_list = [x]
        for i in range(nb_layers):
            cb = _conv_block(x, growth_rate, bn_momentum, dropout_rate, data_format=data_format,
                            block_prefix='ConvBlock_%i' % i)
            x_list.append(cb)
            x = concatenate([x, cb], axis=concat_axis)
            if grow_nb_filters:
                nb_filter += growth_rate

        return x, nb_filter, x_list


def _transition_down_block(x, nb_filter, bn_momentum, weight_decay=1e-4, transition_pooling='max', 
                          block_prefix='TransitionDown', data_format='channels_last'):
    """
    Adds a pointwise convolution layer (with batch normalization and relu),
    and a pooling layer. 
    
    The block cuts height and width of the input in half.
    
    Args:
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers    
        weight_decay: weight decay factor
        transition_pooling: aggregation type for pooling layer
        block_prefix: prefix for naming
        data_format: 'channels_first' or 'channels_last'

    Return:
        output tensor
    """
    with tf.name_scope(block_prefix):
        concat_axis = _channel_dim(data_format)
        x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False, 
                   kernel_regularizer=l2(weight_decay), data_format=data_format)(x)
        if transition_pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2), data_format=data_format)(x)
        elif transition_pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2), data_format=data_format)(x)
        return x        

def _transition_up_block(x, nb_filters, type='deconv', weight_decay=1e-4, 
                          block_prefix='TransitionUp', data_format='channels_last'):
    """
    Adds an upsampling block. 
    
    The block doubles height and width of the input.
    
    Args:
        x: input tensor
        nb_filter: number of convolution filters, this is also the number 
            of feature maps returned by the block
        type: type of upsampling operation: 'upsampling' or 'deconv'
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming
        data_format: 'channels_first' or 'channels_last'
    Returns:
        output tensor
    """
    with tf.name_scope(block_prefix):
        if type == 'upsampling':
            return UpSampling2D(data_format=data_format)(x)
        else:
            return Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
                                kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
                                data_format=data_format)(x)
        

def _create_fc_dense_net(img_input, 
                           nb_classes,
                           nb_dense_block=3,
                           nb_layers_per_block=4,
                           init_conv_filters=48, 
                           growth_rate=12, 
                           initial_kernel_size=(3, 3), 
                           transition_pooling='max',
                           upsampling_type='deconv',
                           bn_momentum=0.9,
                           weight_decay=1e-4, 
                           dropout_rate=0.2,  
                           final_softmax=False,
                           name_scope='DenseNetFCN',
                           data_format='channels_last'):
    """
    Create a fully convolutional DenseNet.
    
    Args:
        img_input: tuple of shape (batch_size, channels, height, width) or (batch_size, height, width, channels)
            depending on data_format
        nb_classes: number of classes
        nb_dense_block: number of dense blocks on the downsampling path, without the bottleneck dense block
        nb_layers_per_block: number of layers in dense blocks, can be an int if all dense blocks have the
            same number of layers or a list of ints with the number of layers in the dense block on the
            downsampling path and the bottleneck dense block
        init_conv_filters: number of filters in the initial concolution
        growth_rate: number of filters in each conv block
        dropout_rate: dropout rate
        initial_kernel_size: the kernel of the first convolution might vary in size based 
            on the application
        transition_pooling: aggregation type of pooling in the downsampling layers: 'avg' or 'max'            
        upsampling_type: type of upsampling operation used: 'upsampling' or 'deconv'
        bn_momentum: Momentum for moving mean and the moving variance in 
            BN layers
        weight_decay: weight decay
        dropout_rate: dropout rate
        final_softmax: if True a final softmax activation layers is added, otherwise the network 
            returns unnormalized log probabilities  
        data_format: 'channels_first' or 'channels_last'

    Returns:
        Tensor with shape: (height * width, nb_classes): class probabilities if final_softmax==True, 
        otherwiese the unnormalized output of the last layer
    """
    with tf.name_scope(name_scope):

        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError('Invalid data_format: %s. Must be one of [channels_first, channels_last]' % data_format)
        
        if data_format == 'channels_first':
            concat_axis = 1
            _, channel, row, col = img_input.shape
        else:
            concat_axis = -1
            _, row, col, channel = img_input.shape
        
        if channel not in [1,3]:
            raise ValueError('Invalid number of channels: %d. Must be one of [1,3]' % channel)

        upsampling_type = upsampling_type.lower()
        if upsampling_type not in ['upsampling', 'deconv']:
            raise ValueError('"upsampling_type" must be one of [upsampling, deconv]')

        # `nb_layers` is a list with the number of layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_layers_per_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        tf.logging.info('Layers in each dense block: %s' % nb_layers)

        # make sure we can concatenate the skip connections with the upsampled
        # images on the upsampling path
        img_downsize_factor = 2**nb_dense_block
        if row % img_downsize_factor != 0:
            raise ValueError('Invalid image height %d. Image height must be a multiple of '
                             '2^nb_dense_block=%d' % (row, img_downsize_factor))
        if col % img_downsize_factor != 0:
            raise ValueError('Invalid image width %d. Image width must be a multiple of '
                             '2^nb_dense_block=%d' % (col, img_downsize_factor))

        # Initial convolution
        with tf.name_scope('Initial'):
            x = Conv2D(init_conv_filters, initial_kernel_size, kernel_initializer='he_normal', padding='same', 
                       use_bias=False, kernel_regularizer=l2(weight_decay), data_format=data_format)(img_input)
            x = BatchNormalization(momentum=bn_momentum, axis=concat_axis, epsilon=1.1e-5)(x)
            x = Activation('relu')(x)

        # keeps track of the current number of feature maps
        nb_filter = init_conv_filters
        
        # collect skip connections on the downsampling path so that
        # they can be concatenated with outputs on the upsampling path
        skip_list = []
                            
        # Build the downsampling path by adding dense blocks and transition down blocks
        for block_idx in range(nb_dense_block):
            x, nb_filter, _ = _dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bn_momentum=bn_momentum,
                                           dropout_rate=dropout_rate, data_format=data_format, 
                                           block_prefix='DenseBlock_%i' % block_idx)

            skip_list.append(x)
            x = _transition_down_block(x, nb_filter, weight_decay=weight_decay, bn_momentum=bn_momentum,
                                       transition_pooling=transition_pooling, data_format=data_format,
                                       block_prefix='TransitionDown_%i' % block_idx)

        # Add the bottleneck dense block.
        _, nb_filter, concat_list = _dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate, bn_momentum=bn_momentum, 
                                                 dropout_rate=dropout_rate, data_format=data_format,
                                                 block_prefix='Bottleneck_DenseBlock_%i' % nb_dense_block)

        tf.logging.info('Number of skip connections: %d' %len(skip_list))

        # reverse the list of skip connections
        skip_list = skip_list[::-1]  
        
        # Build the upsampling path by adding dense blocks and transition up blocks
        for block_idx in range(nb_dense_block):
            n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]
                    
            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps
            l = concatenate(concat_list[1:], axis=concat_axis, name='Concat_DenseBlock_out_%d' % block_idx)
            
            t = _transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay,
                                      data_format=data_format, block_prefix='TransitionUp_%i' % block_idx)

            # concatenate the skip connection with the transition block output
            x = concatenate([t, skip_list[block_idx]], axis=concat_axis, name='Concat_SkipCon_%d' % block_idx)
        
            # Dont allow the feature map size to grow in upsampling dense blocks
            x_up, nb_filter, concat_list = _dense_block(x, nb_layers[nb_dense_block + block_idx + 1], nb_filter=growth_rate, 
                                                        growth_rate=growth_rate, bn_momentum=bn_momentum, dropout_rate=dropout_rate, 
                                                        grow_nb_filters=False, data_format=data_format,
                                                        block_prefix='DenseBlock_%d' % (nb_dense_block + 1 + block_idx))            
        
        # final convolution
        with tf.name_scope('Final'):
            l = concatenate(concat_list[1:], axis=concat_axis)
            x = Conv2D(nb_classes, (1, 1), activation='linear', padding='same', use_bias=False, data_format=data_format)(l)
            x = Reshape((row * col, nb_classes), name='logit')(x)
                
            if final_softmax:
                x = Activation('softmax', name='softmax')(x)

        return x


def build_FC_DenseNet56(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet56"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=4,
                               growth_rate=12,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet56',
                               data_format=data_format)
    return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet67(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet67"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=5,
                               growth_rate=16,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet67',
                               data_format=data_format)
    return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet103(nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    """Build FC-DenseNet103"""
    inputs = Input(shape=input_shape)
    logits = _create_fc_dense_net(inputs,
                               nb_classes=nb_classes,
                               nb_dense_block=5,
                               nb_layers_per_block=[4,5,7,10,12,15],
                               growth_rate=16,
                               init_conv_filters=48, 
                               dropout_rate=dropout_rate,
                               final_softmax=final_softmax,
                               name_scope='FCDenseNet103',
                               data_format=data_format)
    return Model(inputs=inputs, outputs=logits)


def build_FC_DenseNet(model_version, nb_classes, final_softmax, input_shape=(224, 224, 3), dropout_rate=0.2, data_format='channels_last'):
    if model_version == 'fcdn56':
        return build_FC_DenseNet56(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    elif model_version == 'fcdn67':
        return build_FC_DenseNet67(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    elif model_version == 'fcdn103':
        return build_FC_DenseNet103(nb_classes, final_softmax, input_shape, dropout_rate, data_format)
    else:
        raise ValueError('Invalid model_version: %s' % model_version)


###############################################################################
###############################################################################

# https://github.com/mad-Ye/FC-DenseNet-Keras

from keras.layers import Activation,Conv2D,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model														  
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils
from keras.layers import Activation,MaxPooling2D,UpSampling2D,Dense,BatchNormalization,Input,Reshape,multiply,add,Dropout,AveragePooling2D,GlobalAveragePooling2D,concatenate
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Model														  
import keras.backend as K
from keras.regularizers import l2
from keras.engine import Layer,InputSpec
from keras.utils import conv_utils

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)''' 

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
#    l = Reshape((-1, n_classes))(l)
    l = Activation('sigmoid')(l)#or softmax for multi-class
    return l
    
#------

def Tiramisu(
        input_shape=(None,None,3),
        n_classes = 1,
        n_filters_first_conv = 48,
        n_pool = 5,
        growth_rate = 16 ,
        n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4],
        dropout_p = 0.2
        ):
    if type(n_layers_per_block) == list:
            print(len(n_layers_per_block))
    elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError
        
#####################
# First Convolution #
#####################        
    inputs = Input(shape=input_shape)
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

#####################
# Downsampling path #
#####################     
    skip_connection_list = []
    
    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = concatenate([stack, l])
            n_filters += growth_rate
        
        skip_connection_list.append(stack)        
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]
    
#####################
#    Bottleneck     #
#####################     
    block_to_upsample=[]
    
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack,l])
    block_to_upsample = concatenate(block_to_upsample)
  
#####################
#  Upsampling path  #
#####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i ]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)
        
        block_to_upsample = []
        for j in range(n_layers_per_block[ n_pool + i + 1 ]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)

#####################
#  Softmax          #
#####################
    output = SoftmaxLayer(stack, n_classes)            
    model=Model(inputs = inputs, outputs = output)    
    model.summary()
    
    return model

###############################################################################
###############################################################################

# https://nbviewer.jupyter.org/github/Calysto/conx-notebooks/blob/master/work-in-progress/AlphaZero.ipynb

## Building the network, layer blocks:

def add_conv_block(net, input_layer):
    cname = net.add(cx.Conv2DLayer("conv2d-%d",
                    filters=75,
                    kernel_size=(4,4),
                    padding='same',
                    use_bias=False,
                    activation='linear',
                    kernel_regularizer=regularizers.l2(0.0001)))
    bname = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    lname = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    net.connect(input_layer, cname)
    net.connect(cname, bname)
    net.connect(bname, lname)
    return lname

def add_residual_block(net, input_layer):
    prev_layer = add_conv_block(net, input_layer)
    cname = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=75,
        kernel_size=(4,4),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    bname = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    aname = net.add(cx.AdditionLayer("add-%d"))
    lname = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    net.connect(prev_layer, cname)
    net.connect(cname, bname)
    net.connect(input_layer, aname)
    net.connect(bname, aname)
    net.connect(aname, lname)
    return lname

def add_value_block(net, input_layer):
    l1 = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=1,
        kernel_size=(1,1),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    l2 = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    l3 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l4 = net.add(cx.FlattenLayer("flatten-%d"))
    l5 = net.add(cx.Layer("dense-%d",
        20,
        use_bias=False,
        activation='linear',
        kernel_regularizer=regularizers.l2(0.0001)))
    l6 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l7 = net.add(cx.Layer('value_head',
        1,
        use_bias=False,
        activation='tanh',
        kernel_regularizer=regularizers.l2(0.0001)))
    net.connect(input_layer, l1)
    net.connect(l1, l2)
    net.connect(l2, l3)
    net.connect(l3, l4)
    net.connect(l4, l5)
    net.connect(l5, l6)
    net.connect(l6, l7)
    return l7

def add_policy_block(net, input_layer):
    l1 = net.add(cx.Conv2DLayer("conv2d-%d",
        filters=2,
        kernel_size=(1,1),
        padding='same',
        use_bias=False,
        activation='linear',
        kernel_regularizer = regularizers.l2(0.0001)))
    l2 = net.add(cx.BatchNormalizationLayer("batch-norm-%d", axis=1))
    l3 = net.add(cx.LeakyReLULayer("leaky-relu-%d"))
    l4 = net.add(cx.FlattenLayer("flatten-%d"))
    l5 = net.add(cx.Layer('policy_head',
                          42,
                          vshape=(6,7),
                          use_bias=False,
                          activation='linear',
                          kernel_regularizer=regularizers.l2(0.0001)))
    net.connect(input_layer, l1)
    net.connect(l1, l2)
    net.connect(l2, l3)
    net.connect(l3, l4)
    net.connect(l4, l5)
    return l5

def make_network(game, config, residuals=5, name="Residual CNN"):
    """
    Make a full network.

    Game is passed in to get the columns and rows.
    """
    net = cx.Network(name)
    net.add(cx.Layer("main_input", (game.v, game.h, 2),
                     colormap="Greys", minmax=(0,1)))
    out_layer = add_conv_block(net, "main_input")
    for i in range(residuals):
        out_layer = add_residual_block(net, out_layer)
    add_policy_block(net, out_layer)
    add_value_block(net, out_layer)
    net.compile(loss={'value_head': 'mean_squared_error',
                      'policy_head': softmax_cross_entropy_with_logits},
                optimizer="sgd",
                lr=config.LEARNING_RATE,
                momentum=config.LEARNING_RATE,
                loss_weights={'value_head': 0.5,
                              'policy_head': 0.5})
    for layer in net.layers:
        if layer.kind() == "hidden":
            layer.visible = False
    return net

def LocalMain():
    NetConfig = namedtuple("NetConfig", [
        "LEARNING_RATE",
        "MOMENTUM",
    ])
    netconfig = NetConfig(
        LEARNING_RATE=0.1, # SGD Learning rate
        MOMENTUM=0.9, # SGD Momentum
    )
    net = make_network(game, config=netconfig)



