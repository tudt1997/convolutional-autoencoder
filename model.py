from keras.layers import Input, Dense, Conv2D, Flatten, Activation, concatenate, \
                         MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Reshape
from keras.models import Sequential, Model

def autoencoder():
    input_img = Input(batch_shape=(None, 32, 32, 3))

    x = Conv2D(32, (3, 3), activation='elu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(512)(x)

    x = Dense(4 * 4 * 128)(encoded)
    x = Reshape((4, 4, 128))(x)
    x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    return autoencoder

def custom_unet():
    inputs = Input((32, 32, 3))

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    up4 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='elu', padding='same')(up4)
    conv4 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv4)
    up5 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='elu', padding='same')(up5)
    conv5 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv5)

    conv5 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv5)
    
    model = Model(inputs=[inputs], outputs=[conv5])

    return model

def unet():
    inputs = Input((32, 32, 3))

    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv6)
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv7)
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv8)
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv9)

    conv10 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model