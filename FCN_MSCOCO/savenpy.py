import data_load
import FCN
import numpy as np
import tensorflow as tf
import util
from imageio import imread
from skimage.transform import resize
from skimage import color
from matplotlib.pyplot import imshow, hist, show,figure, subplot, subplots_adjust, setp


######################input file name of image u want to predict#########################
filename = 'aas.jpg'
#########################################################################################

def save_pred_arr(pred, class_num, filename, is_single = False):
    pred = np.squeeze(pred, axis=3)
    output_batch = np.empty([pred.shape[0], pred.shape[1], pred.shape[2], class_num], dtype=np.bool)

    for i in range(class_num):
        tmpmask = pred == i
        output_batch[:, :, :, i] = tmpmask

    if is_single :
        output_batch = np.squeeze(output_batch, axis=0)

    np.save(filename, output_batch)

def masking(img, mask, alpha):
    fg = np.any(mask != [0, 0, 0], axis=2)
    fg = np.dstack((fg, fg, fg))
    bg = ~fg

    img_fg = img * fg
    img_bg = img * bg

    img_hsv = color.rgb2hsv(img_fg)
    mask_hsv = color.rgb2hsv(mask)

    img_hsv[:, :, 0] = mask_hsv[:, :, 0]
    img_hsv[:, :, 1] = mask_hsv[:, :, 1] * alpha

    img_fg = color.hsv2rgb(img_hsv)

    img = img_bg + img_fg

    return img


batch_size = 1
image_idx = 15
masker = util.masker()
class_num = len(masker.class_color_list)

# img_loader = data_load.ImgLoader()
FCN = FCN.FCN(batch_size, 0.001)

# img_loader.run('train') # train or val
# input_batch, _ = img_loader.nextbatch(batch_size, image_idx)

input_img = imread(filename)

input_batch = resize(input_img[:, :, :], (320, 320, 3), order=1) *255

input_batch = np.expand_dims(input_batch, axis=0)


##########################################
##########################################

X = tf.placeholder(tf.float32, [None, input_batch.shape[1], input_batch.shape[2], input_batch.shape[3]])  # h * w * 3
Y = tf.placeholder(tf.float32,[None, input_batch.shape[1], input_batch.shape[2], class_num])  # h * w * (class+1) <= 배경포함(+1)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

_, pred, _, _ = FCN.train(X, Y, keep_prob, is_training)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
saver.restore(sess, ckpt.model_checkpoint_path)

# sample2see, loss_ = sess.run([pred, loss], feed_dict={X: input_batch, Y: label_batch_cal_loss, keep_prob: 1.0, is_training: False})
sample2see = sess.run(pred, feed_dict={X: input_batch, keep_prob: 1.0, is_training: False})

mask = masker.make_mask_from_label(sample2see)

# save_pred_arr(sample2see, class_num ,filename, batch_size==1)



figure()
for i in range(batch_size) :
    ax = subplot(batch_size, 3, 3 * i+1)
    imshow(input_batch[i]/255)
    ax.set_title('INPUT')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)

    ax=subplot(batch_size, 3, 2 * i + 2)
    if (batch_size == 1) :
        imshow(mask)
    else:
        imshow(mask[i])
    ax.set_title('PREDICTION')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)

    ax = subplot(batch_size, 3, 2 * i + 3)
    if (batch_size == 1):
        imshow(masking(input_batch[i], mask, 0.8)/255)
    else:
        imshow(masking(input_batch[i], mask[i], 0.8)/255)
    ax.set_title('PREDICTION')
    setp(ax.get_xticklabels(), visible=False)
    setp(ax.get_yticklabels(), visible=False)



subplots_adjust(wspace=0.2, hspace=0.4)
show()




