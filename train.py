import tensorflow as tf
from nn import WaldoNN
import numpy as np
import config

images_tf = tf.placeholder(tf.float32, [None, 64, 64, 3], name="images")
labels_tf = tf.placeholder(tf.int64, name="labels")
keep_prob = tf.placeholder(tf.float32)

waldoNN = WaldoNN()

logits = waldoNN.conv_net(images_tf, keep_prob)
prediction = tf.nn.softmax(logits)

loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_tf))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss_tf)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_tf, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as session:
    tf.global_variables_initializer().run()
    train_data, test_data = waldoNN.get_data()
    exit()
    train_images = np.array(list(map(lambda x: waldoNN.load_image(x), train_data[:, 0])))
    # TODO batch operation
    for step in range(config.n_epochs):
        session.run(optimizer, feed_dict={images_tf: train_images,
                                          labels_tf: train_data[:, 1],
                                          keep_prob: config.dropout})

        if step % config.display_step == 0 or step == 1:
            loss, acc = session.run([loss_tf, accuracy], feed_dict={images_tf: np.array(list(map(lambda x: waldoNN.load_image(x), train_data[:, 0]))),
                                                                 labels_tf: train_data[:, 1],
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Loss= " +
                  "{:.4f}".format(loss) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
    print("Optimization Finished!")

    print("Testing Accuracy:",
          session.run(accuracy, feed_dict={images_tf: np.array(list(map(lambda x: waldoNN.load_image(x), test_data[:, 0]))),
                                            labels_tf: test_data[:, 1],
                                            keep_prob: 1.0}))
