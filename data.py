import tensorflow as tf
import pandas as pd
import sys
from absl import logging

logging.set_verbosity(logging.INFO)


def generate_tf_records(img_dir, csv_dir, file_name, oversampling=True):
    logging.info('Generating {}'.format(file_name))
    writer = tf.io.TFRecordWriter(file_name)
    def preprocessing(img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_image(img)
        img = tf.image.crop_to_bounding_box(img, 0, 266, 2848, 3426)
        img = tf.image.pad_to_bounding_box(img, 288, 0, 3424, 3426)
        img = tf.image.resize(img, (256, 256), antialias=True)
        img = tf.cast(img, tf.uint8)
        img = tf.io.encode_jpeg(img, quality=99)
        return img

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # separate different classes
    df = pd.read_csv(csv_dir)
    df_class_0 = df[df['Retinopathy grade'] == 0]
    df_class_1 = df[df['Retinopathy grade'] == 1]
    df_class_2 = df[df['Retinopathy grade'] == 2]
    df_class_3 = df[df['Retinopathy grade'] == 3]
    df_class_4 = df[df['Retinopathy grade'] == 4]

    df_nrdr = pd.concat([df_class_0, df_class_1], sort=False)
    print(df_nrdr)
    df_rdr = pd.concat([df_class_2, df_class_3, df_class_4], sort=False)

    # oversampling of the less samples' dataset
    logging.info('Samples of NRDR: {}'.format(len(df_nrdr)))
    logging.info('Samples of RDR: {}'.format(len(df_rdr)))
    if oversampling:
        if len(df_rdr) <= len(df_nrdr):
            df_rdr = df_rdr.sample(len(df_nrdr), replace=True)
        else:
            df_nrdr = df_nrdr.sample(len(df_rdr), replace=True)

    # Define a new label rdr
    df_rdr['rdr'] = 1
    df_nrdr['rdr'] = 0
    df_over = pd.concat([df_nrdr, df_rdr], sort=False)
    df_over_shuffle = df_over.sample(frac=1)
    if oversampling:
        logging.info('Oversampling completed, numbers of samples: {}'.format(len(df_over_shuffle)))
    else:
        logging.info('Oversampling is off, sampling completed, numbers of samples: {}'.format(len(df_over_shuffle)))

    r = 0
    for img_name in df_over_shuffle['Image name']:
        r += 1
        if r % 20 == 0 or r == len(df_over_shuffle['Image name']):
            logging.info('Preprocessing data: {}/{}'.format(r, len(df_over_shuffle['Image name'])))
            sys.stdout.flush()
        image_path = img_dir + '/' + img_name + '.jpg'
        img = preprocessing(image_path)
        label = df_over_shuffle.loc[df_over_shuffle['Image name'] == img_name, 'rdr'].values[0]
        feature = {'label': _int64_feature([label]),
                   'image': _bytes_feature([img.numpy()]),
                   'name': _bytes_feature([img_name.encode('utf-8')])}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


def make_dataset(BATCH_SIZE, file_name='test_tf.record', split=False, split_train_size=0.8):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    raw_dataset = tf.data.TFRecordDataset([file_name])
    feature_description = {'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                           'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
                           'name': tf.io.FixedLenFeature([], tf.string, default_value='')}

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    if not split:
        count = 0
        for _ in parsed_dataset:
            count += 1
        dataset = parsed_dataset.map(lambda x: (tf.io.decode_jpeg(x['image']), x['label'], x['name']))
        test_set = dataset.shuffle(count).prefetch(buffer_size=AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)
        logging.info('Test Set is made.')
        return test_set

    if split:
        count = 0
        for _ in parsed_dataset:
            count += 1
        dataset = parsed_dataset.map(lambda x: (tf.io.decode_jpeg(x['image']), x['label'], x['name'])).shuffle(buffer_size=count)
        train_size = int(count * split_train_size)
        train_set = dataset.take(train_size)
        valid_set = dataset.skip(train_size)
        train_set = train_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        valid_set = valid_set.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
        logging.info('Training Set and Validation Set are made.')
        return train_set, valid_set


if __name__ == '__main__':
    generate_tf_records('./Dataset/1. Original Images/a. Training Set',
                        './Dataset/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv',
                        'train_tf_record', True)
    generate_tf_records('./Dataset/1. Original Images/b. Testing Set',
                        './Dataset/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv',
                        'test_tf_record', False)
    # training_set, valid_set = make_dataset(BATCH_SIZE=8, file_name='train_tf_record', split=True)
    test_set = make_dataset(BATCH_SIZE=16, file_name='test_tf_record', split=False)



