# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)

[sample1]: ./output/sample1.png "Sample1"
[sample2]: ./output/sample2.png "Sample2"
[sample3]: ./output/sample3.png "Sample3"
[sample4]: ./output/sample4.png "Sample4"
[sample5]: ./output/sample5.png "Sample5"
[sample6]: ./output/sample6.png "Sample6"
[sample7]: ./output/sample7.png "Sample7"
[sample8]: ./output/sample8.png "Sample8"
[output_animation]: ./output/output.gif "Output Animation"

### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.

## Rubric Points

[Rubric points](https://review.udacity.com/#!/rubrics/989/view)

### Does the project load the pretrained vgg model?

Yes, the function 'load_vgg' loads pretrained vgg model saved in 'data/vgg' and returns tensors required.

```
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

```

### Does the project learn the correct features from the images?

The function 'layers' creates upsampling layers for fully convolutional network with skip connections. The output of layers 3 and 4 are scaled before they are fed to 1x1 convolution following this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100).

```
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

```

### Does the project optimize the neural network?

Yes, an Adam optimizer is used.
The loss function includes all the regularization losses and softmax cross entropy loss.

```
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + sum(reg_losses)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

```

### Does the project train the neural network?

Yes. The function will print average loss for each epoch while training the network.

```
    for epoch in range(epochs):
        total_loss = 0
        batches    = 0
        print("EPOCH {}...".format(epoch))
        for images, labels in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
            				    feed_dict={input_image: images, correct_label: labels, keep_prob: 0.5, learning_rate: 0.0005})
            total_loss += loss
            batches += 1
        print("loss: {}".format(total_loss/batches))
        print()

```

### Does the project train the model correctly?
Yes, the average loss drops from 0.67 of the first epoch to 0.07 of the last one.

### Does the project use reasonable hyperparameters?

The hyperparameters used in the project are:

| Hyperparameter        |     Value             |
|:---------------------:|:---------------------:|
| keep_prob             | 0.5                   |
| learning_rate         | 0.0003                |
| epochs                | 25                    |
| batch_size            | 16                    |

### Does the project correctly label the road?

The followings are some inference samples:

![Animation][output_animation]

![Sample1][sample1]

![Sample2][sample2]

![Sample3][sample3]

![Sample4][sample4]

![Sample5][sample5]

![Sample6][sample6]

![Sample7][sample7]

![Sample8][sample8]



