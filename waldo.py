import tensorflow as tf
from nn import WaldoNN
import numpy as np
import config
import sys
import os
from predict import Predict

images_tf = tf.placeholder(tf.float32, [None, 64, 64, 3], name="images")
labels_tf = tf.placeholder(tf.int64, name="labels")
keep_prob = tf.placeholder(tf.float32)

waldoNN = WaldoNN()

logits = waldoNN.conv_net(images_tf, keep_prob)
prediction = tf.nn.softmax(logits)

loss_tf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_tf))
optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(loss_tf)

correct_pred = tf.equal(tf.argmax(prediction, 1), labels_tf)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
i = tf.argmax(prediction, 1)

saver = tf.train.Saver()


with tf.Session() as session:
    if tf.train.checkpoint_exists(config.models_path + "/model.ckpt") and "--retrain" not in sys.argv:
        saver.restore(session, config.models_path + "/model.ckpt")
        Predict(session, sys.argv[1]).find_waldo(prediction, images_tf, keep_prob)
        exit()

    train_images_batch, train_labels_batch, test_images, test_labels = waldoNN.get_batch()
    session.run(tf.local_variables_initializer())
    session.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    for step in range(config.n_epochs):
        images, labels = session.run([train_images_batch, train_labels_batch])
        session.run(optimizer, feed_dict={images_tf: images,
                                          labels_tf: labels,
                                          keep_prob: config.dropout})
       
        if step % config.display_step == 0 or step == 1:
            loss, acc = session.run([loss_tf, accuracy], feed_dict={images_tf: images,
                                                                 labels_tf: labels,
                                                                 keep_prob: 1})
            print("Step " + str(step) + ", Loss=",
                  loss, ", Training Accuracy=", acc)
    print("Optimization Finished!")

    saver.save(session, config.models_path + "/model.ckpt")

    acc, pred = session.run([accuracy, prediction], feed_dict={images_tf: np.array(list(map(lambda x: waldoNN.load_image(x), test_images))),
                                        labels_tf: test_labels,
                                        keep_prob: 1.0})
    print("Testing accuracy", acc)

    results = np.column_stack((test_images, np.argmax(pred.astype("f"), axis=1), test_labels))

    print(results[10:])

    print(results[np.random.choice(results.shape[0], 10, replace=False), :])


    coord.request_stop()
    coord.join(threads)
    session.close()


