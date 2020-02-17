import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_augmentation(image, rotation_range=5, width_shift_range=0.15,
                      height_shift_range=0.15, zoom_range=None,
                      brightness_range=(0.5, 1.3),
                      horizontal_flip=True, vertical_flip=True, shear=0.05, fill_mode='constant'):
    if zoom_range is None:
        zoom_range = [0.7, 1.2]
    datagen = ImageDataGenerator(rotation_range=rotation_range,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 zoom_range=zoom_range,
                                 brightness_range=brightness_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip, shear_range=shear,
                                 fill_mode=fill_mode, data_format='channels_last',
                                 dtype='int8')
    img = next(datagen.flow(image, shuffle=False))
    img = img / 255.0
    return img


if __name__ == '__main__':
    img = tf.io.read_file('./Dataset/1. Original Images/a. Training Set/IDRiD_001.jpg')
    img = tf.io.decode_image(img)
    img = tf.image.crop_to_bounding_box(img, 0, 266, 2848, 3426)
    img = tf.image.pad_to_bounding_box(img, 288, 0, 3424, 3426)
    img = tf.image.resize(img, (256, 256))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)
    data_augmentation(img)