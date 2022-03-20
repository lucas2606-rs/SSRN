import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation,\
    Dropout, GlobalAveragePooling3D, Dense, Flatten


class ConvBN(Model):
    def __init__(self, kernel_size, filters, strides, padding, is_bn=True, is_dropout=True):
        super(ConvBN, self).__init__()
        self.model = Sequential()
        self.model.add(Conv3D(kernel_size=kernel_size, filters=filters, strides=strides, padding=padding, use_bias=True))
        if is_bn:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        if is_dropout:
            self.model.add(Dropout(0.5))

    def call(self, x):
        y = self.model(x, training=False)
        return y


class SSResBlk(Model):
    def __init__(self, kernel_size, filters, strides, is_bn=True, is_dropout=True):
        super(SSResBlk, self).__init__()
        self.c1 = ConvBN(kernel_size=kernel_size, filters=filters,
                         strides=strides, padding='same', is_bn=is_bn, is_dropout=is_dropout)
        self.c2 = ConvBN(kernel_size=kernel_size, filters=filters,
                         strides=strides, padding='same', is_bn=is_bn, is_dropout=is_dropout)

    def call(self, x):
        residual = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        out = x2 + residual
        return out


class SSResNet(Model):
    def __init__(self, nc, filters, is_bn=True, is_dropout=True):
        super(SSResNet, self).__init__()
        # spectral residual network
        self.c1 = Conv3D(filters=24, kernel_size=(1,1,7), strides=(1,1,2))
        self.spectral_res_blk1 = SSResBlk(kernel_size=(1,1,7), filters=filters,
                                          strides=(1,1,1), is_bn=is_bn, is_dropout=is_dropout)
        self.spectral_res_blk2 = SSResBlk(kernel_size=(1,1,7), filters=filters,
                                          strides=(1,1,1), is_bn=is_bn, is_dropout=is_dropout)
        self.c2 = Conv3D(filters=128, kernel_size=(1,1,97), strides=(1,1,1))
        # spatial residual network
        self.c3 = Conv3D(filters=24, kernel_size=(3,3,128), strides=(1,1,1))
        self.spatial_res_blk1 = SSResBlk(kernel_size=(3,3,24), filters=filters,
                                         strides=(1,1,1), is_bn=is_bn, is_dropout=is_dropout)
        self.spatial_res_blk2 = SSResBlk(kernel_size=(3,3,24), filters=filters,
                                         strides=(1,1,1), is_bn=is_bn, is_dropout=is_dropout)
        self.p1 = GlobalAveragePooling3D()
        self.f1 = Dense(nc, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.spectral_res_blk1(x)
        x = self.spectral_res_blk2(x)
        x = self.c2(x)
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = self.c3(x)
        x = self.spatial_res_blk1(x)
        x = self.spatial_res_blk2(x)
        x = self.p1(x)
        y = self.f1(x)
        return y



