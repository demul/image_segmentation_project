import data_load
import FCN_inference
import numpy as np
import tensorflow as tf
import util
from imageio import imread
from cv2 import resize
from matplotlib.pyplot import imshow, hist, show,figure, subplot, subplots_adjust, setp
import cv2

batch_size = 1
masker = util.masker()
class_num = len(masker.class_color_list)


FCN = FCN_inference.FCN(batch_size, 0.001)

#
# filename = 'aaa'
#
# input_img = imread(filename + '.jpg')
#
# input_batch = resize(input_img[:, :, :], (256, 256, 3), order=1) *255
#
# input_batch = np.expand_dims(input_batch, axis=0)


##########################################
##########################################

X = tf.placeholder(tf.float32, [None, 320, 320, 3])  # h * w * 3
Y = tf.placeholder(tf.float32,[None, 320, 320, class_num])  # h * w * (class+1) <= 배경포함(+1)
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

_, pred, _, _ = FCN.train(X, Y, keep_prob, is_training)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
saver.restore(sess, ckpt.model_checkpoint_path)
capture = cv2.VideoCapture(0)

capture.set(3, 320)
capture.set(4, 320)


while (1):
    ret, X_test = capture.read()


    X_test = resize(X_test[:, :, :], (320, 320, 3))

    b, g, r = cv2.split(X_test)  # img파일을 b,g,r로 분리
    X_test2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge

    real_input = np.expand_dims(X_test2, axis=0) *255
    prediction2see = sess.run(pred, feed_dict={X: real_input, keep_prob: 1.0, is_training: False})

    mask = masker.make_mask_from_label(prediction2see).astype(np.float64)/255

    r, g, b = cv2.split(mask)  # img파일을 r,g,b로 분리
    mask = cv2.merge([b, g, r])  # b, r을 바꿔서 Merge

    print('running')

    cv2.imshow('webcam', X_test)
    cv2.imshow('webcam2', mask)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

capture.release()
cv2.destroyAllWindows()

    # sample2see, loss_ = sess.run([pred, loss], feed_dict={X: input_batch, Y: label_batch_cal_loss, keep_prob: 1.0, is_training: False})





