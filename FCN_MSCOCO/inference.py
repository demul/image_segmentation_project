import data_load
import FCN
import tensorflow as tf
import util
from matplotlib.pyplot import imshow, hist, show,figure, subplot, subplots_adjust, setp

batch_size = 5
image_idx = 30
masker = util.masker()

img_loader = data_load.ImgLoader('pascal')
FCN = FCN.FCN(batch_size, 0.001)

img_loader.run('val') # train or val
input_batch, label_batch = img_loader.nextbatch_for_inference(batch_size, image_idx)
_, label_batch_cal_loss = img_loader.nextbatch(batch_size, image_idx)


##########################################
##########################################

X = tf.placeholder(tf.float32, [None, input_batch.shape[1], input_batch.shape[2], input_batch.shape[3]])  # h * w * 3
Y = tf.placeholder(tf.float32,[None, label_batch.shape[1], label_batch.shape[2], len(masker.class_color_list)])  # h * w * (class+1) <= 배경포함(+1)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

_, pred, loss, _ = FCN.train(X, Y, keep_prob, is_training)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
saver.restore(sess, ckpt.model_checkpoint_path)

# sample2see, loss_ = sess.run([pred, loss], feed_dict={X: input_batch, Y: label_batch_cal_loss, keep_prob: 1.0, is_training: False})
sample2see, loss_ = sess.run([pred, loss], feed_dict={X: input_batch, Y: label_batch_cal_loss, keep_prob: 1.0, is_training: False})

mask = masker.make_mask_from_label(sample2see)

IOU = util.intersection_over_union(sample2see, label_batch_cal_loss)

figure()
for i in range(batch_size) :
    ax = subplot(batch_size, 3, 3 * i+1)
    imshow(input_batch[i]/255)
    ax.set_title('INPUT')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)

    ax=subplot(batch_size, 3, 3 * i + 2)
    imshow(mask[i])
    ax.set_title('PREDICTION')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)

    ax=subplot(batch_size, 3, 3 * i + 3)
    imshow(label_batch[i]/255)
    ax.set_title('GROUND TRUTH')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)
subplots_adjust(wspace=0.2, hspace=0.4)
print('loss :', loss_, '   IOU :', IOU)
show()