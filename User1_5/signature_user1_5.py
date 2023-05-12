import pandas as pd
import os
import glob as gb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image
from tensorflow import keras
import shutil

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam

import logging

learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07

custom_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
model_path = r"../Offline Signature Recognition/User1_5/identify_sign.h5"
loaded_model = keras.models.load_model(model_path, compile=False)

loaded_model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Define the save_keypoints_plot function
# def save_keypoints_plot(image, keypoints, output_filename):
#   keypoints_image = np.copy(image)
#   cv2.drawKeypoints(image, keypoints, keypoints_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#   plt.imshow(cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB))
#   plt.axis('off')
#   plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
#   plt.close()
def save_keypoints_plot(image, keypoints, output_path, index):
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(f"{output_path}.png", bbox_inches='tight', pad_inches=0)
    plt.close()



def signature_verification(train_user, test_image_path):

  image = cv2.imread(test_image_path)
  plt.imshow(image)

  image_fromarray = Image.fromarray(image, 'RGB')
  resize_image = image_fromarray.resize((128, 128))
  expand_input = np.expand_dims(resize_image,axis=0)
  input_data = np.array(expand_input)
  input_data = input_data/255
  # C:\PythonAlgorithm\commission7\Offline-Signature-Recognition-Using-CNN-and-SIFT\Offline-Signature-Recognition-Using-CNN-and-SIFT\User1-5\Signature_detect\real\00100001.png
  pred = loaded_model.predict(input_data)
  user = int(train_user[4:5])
  input_user = user

  #original_signature = cv2.imread("/content/drive/My Drive/Signature_classify/train/User"+ str(input_user)+"/00"+str(input_user)+"0100"+str(input_user)+".png")
  original_signatures = []
  for i in range(5):
    original_signatures.append(r"../Offline Signature Recognition/User1_5/Signature_classify/train/"+ str(train_user) +"/00"+str(input_user)+"0"+str(i)+"00"+str(input_user)+".png")

  print(original_signatures)
  

  input_image = image

  def SIFT(image1,image2):
    # Convert the training image to RGB
    training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    testing_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # Convert the training image to gray scale
    training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    testing_gray = cv2.cvtColor(testing_image, cv2.COLOR_RGB2GRAY)
    # Display traning image and testing image
    fx, plots = plt.subplots(1, 2, figsize=(20,10))
    plots[0].set_title("Training Image")
    plots[0].imshow(training_image)
    plots[1].set_title("Testing Image")
    plots[1].imshow(testing_image)
    #now checking whether the image matches using SIFT algorithm
    surf = cv2.SIFT_create()

    train_keypoints, train_descriptor = surf.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = surf.detectAndCompute(testing_gray, None)

    keypoints_without_size = np.copy(training_image)
    keypoints_with_size = np.copy(training_image)

    keypoints_without_size1 = np.copy(testing_image)
    keypoints_with_size1 = np.copy(testing_image)

    cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

    cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.drawKeypoints(testing_image, test_keypoints, keypoints_without_size1, color = (0, 255, 0))

    cv2.drawKeypoints(testing_image, test_keypoints, keypoints_with_size1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display image with and without keypoints size
    fx, plots = plt.subplots(1, 2, figsize=(20,10))

    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')

    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')

    # Print the number of keypoints detected in the training image
    print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

    # Print the number of keypoints detected in the query image
    print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

    # Perform the matching between the SURF descriptors of the training image and the test image
    matches = bf.match(train_descriptor, test_descriptor)

    # The matches with shorter distance are the ones we want.
    matches = sorted(matches, key = lambda x : x.distance)
    similar_regions=[i for i in matches if i.distance<1000 ]

    return (len(similar_regions)/len(matches)), len(train_keypoints), len(test_keypoints),train_keypoints,test_keypoints

  max_identical = 0
  sift_similarity = 0
  all_train_keypoints = []
  all_test_keypoints = []

  for i in range(len(original_signatures)):
    original_signature = cv2.imread(original_signatures[i])
    sift_similarity, train_keypoints, test_keypoints,image_train_point,image_test_point = SIFT(original_signature, input_image)
    all_train_keypoints.append(train_keypoints)
    all_test_keypoints.append(test_keypoints)
    if max_identical < sift_similarity:
        max_identical = sift_similarity

    # Save training image keypoints plot
    save_keypoints_plot(original_signature, image_train_point, f"""static/keypoints/train_keypoints{i+1}""", i)

    # Save testing image keypoints plot
    save_keypoints_plot(input_image, image_test_point,f"""static/keypoints/test_keypoints{i+1}""", i)


  test_image_basename = os.path.basename(test_image_path)
  destination_path = os.path.join('static', 'test', test_image_basename)
  shutil.copy(test_image_path, destination_path)

  result_dict = {
    'similarity': max_identical,
    'signature_belongs_to': train_user if max_identical > 0.2 else 'Unknown User',
    'signature_status': 'Genuine Signature' if max_identical > 0.2 else 'Forged Signature',
    'result_signature_path': os.path.basename(os.path.relpath(original_signatures[np.argmax([SIFT(cv2.imread(sig), input_image)[0] for sig in original_signatures])], start="../Offline Signature Recognition/User1_5/Signature_classify/train/User1").replace('\\', '/')),
    'test_signature_path': os.path.basename(destination_path.replace('\\', '/')),
    'train_keypoints': all_train_keypoints,
    'test_keypoints': all_test_keypoints,
  }

  print(result_dict)

  return result_dict