from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential
from model import autoencoder, custom_unet, unet
# from custom_mobilenet_v2 import CustomMobileNetV2

# Set TensorFlow to allow for growth. Helps compatibility.
import tensorflow as tf
from keras import backend as ktf
ktf.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
ktf.set_session(session)

input_shape = (32, 32, 3)
target_size = input_shape[:2]
batch_size = 64

model = autoencoder()
# model = UNET()

model.compile(loss = "mean_squared_error", optimizer = Adam(lr = 0.001))

model.summary()

image_generator = ImageDataGenerator(
                              # rotation_range=15,
                            #    width_shift_range=0.1,
                            #    height_shift_range=0.1,
                            #    shear_range=0.01,
                            #    zoom_range=[0.9, 1.25],
                               horizontal_flip=False,
                               vertical_flip=False,
                              #  fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.7, 1.2],
                               rescale=1./255,
                               validation_split=0.2)

train_generator = image_generator.flow_from_directory('dataset',
                                target_size=target_size,
                                batch_size=batch_size,
                                color_mode='rgb',
                                class_mode='input', # For autoencoder
                                subset='training')

validation_generator = image_generator.flow_from_directory('dataset',
                                target_size=target_size,
                                batch_size=batch_size,
                                color_mode='rgb',
                                class_mode='input', # For autoencoder
                                subset='validation')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=5)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath='param/ae-left-weights.{epoch:02d}-{val_loss:.5f}.h5', verbose=1, save_best_only=True)
losses = model.fit_generator(train_generator,
                            steps_per_epoch=train_generator.n // batch_size,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.n // batch_size,
                            verbose=1,
                            epochs=20,
                            callbacks=[early_stopping, checkpointer])