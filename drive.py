# Import all packages
import cv2
import numpy as np
import tensorflow as tf
import scipy
import pandas as pd
import matplotlib.pyplot as plt

CHECKPOINT = "./train_model.ckpt"
DATA_FILE = "driving_dataset/data.txt"

# Ignoring the INFO from the tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

loaded_graph = tf.Graph()

# Load the file to extract the ground truth
f = open(DATA_FILE, 'r')
info = []
for line in f:
    info.append(line.split())

data = pd.DataFrame(info, columns=['image_name','steering_angle'])

# Start the loaded graph session
with tf.Session(graph=loaded_graph) as sess:

    # Load the saved model
    loader = tf.train.import_meta_graph(CHECKPOINT + '.meta')
    loader.restore(sess, CHECKPOINT)

    # Load the required parameters from the graph
    final_layer = loaded_graph.get_tensor_by_name('fc5/BiasAdd:0')
    input_layer = loaded_graph.get_tensor_by_name('input:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    # Function which returns the predicted steering angle
    def steering_angle_predict(img):
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, 66, 200, 3))
        
        test_pred = sess.run(final_layer, feed_dict={input_layer: img, keep_prob: 1.0})
            
        return np.squeeze(test_pred) 

    steer = cv2.imread('steering_wheel_image.jpg', 0)
    rows, cols = steer.shape
    smoothed_angle = 0
    my_labels = {"x1" : "Ground Value", "x2" : "Predicted Value"}

    i = 0
    while (cv2.waitKey(10) != ord('q')):

        ground_value = float(data['steering_angle'].iloc[i])

        full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg")
        # Resize every frame into 66 x 200
        gray = cv2.resize(full_image[-150:], (200,66)) / 255.0
        pred_value = steering_angle_predict(gray) * 180.0 / scipy.pi

        cv2.imshow('frame', cv2.resize(full_image, (600, 400), interpolation=cv2.INTER_AREA))

        # Smoothing the predicted steering angle 
        smoothed_angle += 0.2 * pow(abs((pred_value - smoothed_angle)), 2.0 / 3.0) * (pred_value - smoothed_angle) / abs(pred_value - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(steer, M, (cols, rows))
        cv2.imshow("steering wheel", cv2.resize(dst, (180, 180)))

        # Plot the predicted value and the ground value to compare the results for every 20 iterations 
        if i % 20 == 0:
            plot1, = plt.plot(i, ground_value, '-x', color='r', label=my_labels["x1"])
            plot2, = plt.plot(i, pred_value, '-+', color='b', label=my_labels["x2"])
            plt.ylim((-200, 550))
            plt.legend(handles=[plot1, plot2])
            plt.pause(0.05)

        if i % 400 == 0:
            plt.cla()

        i += 1

    cv2.destroyAllWindows()