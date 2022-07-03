import glob
import keras.initializers
import keras.layers
import tensorflow as tf
import tensorflow_addons as tfa

CLASSES = 5

COLORS = [
    'red', 'blue', 'green',
    'orange', 'pink'
]

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (900, 1440)


def load_img(img, mask):
    """загрузка изображений"""
    img = tf.io.read_file(img)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, OUTPUT_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img / 255.0

    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    masks = []

    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, SAMPLE_SIZE + (CLASSES,))
    return img, masks


def augmentate_img(img, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    img = tf.image.central_crop(img, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        img = tf.image.flip_left_right(img)
        masks = tf.image.flip_left_right(masks)

    """формируем выходной размер данных"""
    img = tf.image.resize(img, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)
    return img, masks


def input_layer():
    return keras.layers.Input(shape=SAMPLE_SIZE + (3,))


def downsample_block(filters, size, batch_norm=True):
    initializer = tf.keras.initializers.GlorotNormal()

    result = keras.Sequential()

    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(keras.layers.BatchNormalization())

    result.add(keras.layers.LeakyReLU())
    return result


def main():
    images = glob.glob('train_dataset_train/train/images/*.png')
    masks = glob.glob('train_dataset_train/train/mask/*.png')

    dataset_img = tf.data.Dataset.from_tensor_slices(images)
    dataset_mask = tf.data.Dataset.from_tensor_slices(masks)

    dataset = tf.data.Dataset.zip((dataset_img, dataset_mask))

    dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat(10)
    dataset = dataset.map(augmentate_img, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = dataset.take(7000).cache()
    test_dataset = dataset.skip(7000).take(1203).cache()

    train_dataset = train_dataset.batch(16)
    test_dataset = test_dataset.batch(16)
