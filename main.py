import os.path
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import scipy
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

print("Test load vgg: ")
tests.test_load_vgg(load_vgg, tf)
print()

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)
    layer4_1x1 = tf.layers.conv2d(layer4_out_scaled, num_classes, 1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output, layer4_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 4, 2, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    layer3_1x1 = tf.layers.conv2d(layer3_out_scaled, num_classes, 1, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    output = tf.add(output, layer3_1x1)
    output = tf.layers.conv2d_transpose(output, num_classes, 16, 8, padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01), kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return output

print("Test layers: ")
tests.test_layers(layers)
print()

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + sum(reg_losses)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return logits, train_op, cross_entropy_loss

print("Test optimize: ")
tests.test_optimize(optimize)
print()

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        print("EPOCH {}...".format(epoch))
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                    feed_dict={input_image: images, correct_label: labels,
                        keep_prob: 0.5, learning_rate: 0.0003})
            total_loss += loss
            batches += 1
        print("loss: {}".format(total_loss/batches))
        print()

print("Test train_nn: ")
tests.test_train_nn(train_nn)
print()

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    print("Test for kitti dataset: ")
    tests.test_for_kitti_dataset(data_dir)
    print()

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()

        # Train NN using the train_nn function
        epochs = 25
        batch_size = 16
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)

        saver.save(sess, 'checkpoints/semantic_segmentation.ckpt')
        print('Model saved')

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video

def inference():
    num_classes = 2
    image_shape = (160, 576)
    vgg_path = './data/vgg'
    with tf.Session() as sess:
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits = tf.reshape(nn_last_layer, [-1, num_classes])

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
        print('Model restored')

        # Run segmentic segmenation on video
        def process_image(img):
            img = scipy.misc.imresize(img, image_shape)
            img_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_input: [img]})
            img_softmax = img_softmax[0][:,1].reshape(image_shape[0], image_shape[1])
            segmentation = (img_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            merged_img = scipy.misc.toimage(img)
            merged_img.paste(mask, box=None, mask=mask)
            return np.array(merged_img)

        video_file = 'video/driving.mp4'
        clip = VideoFileClip(video_file)
        new_clip = clip.fl_image(process_image)
        video_output = 'video/driving_output.mp4'
        new_clip.write_videofile(video_output, audio=False)

        print('Done')

if __name__ == '__main__':
    run()
    #inference()
