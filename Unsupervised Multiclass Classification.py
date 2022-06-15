# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/gdrive')

import csv
import os
from PIL import Image
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import tensorflow as tf
from sklearn.svm import SVC
from itertools import combinations
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.color import rgb2gray
from skimage import feature


class ModelAutoencoder(Model):
    def __init__(self):
        super(ModelAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, input_shape=(128, 128, 3)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(512)])

        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='sigmoid'),
            layers.Reshape((8, 8, 8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ModelEncoder(Model):
    def __init__(self):
        super(ModelEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(128, 128, 3)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, input_shape=(128, 128, 3)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(512)])

        self.decoder = tf.keras.Sequential([
            layers.Dense(512, activation='sigmoid'),
            layers.Reshape((8, 8, 8)),
            layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        #print(encoded.shape)
        return encoded

#Function for renaming the images (for special characters that need certain data type extensions)
def rename_images():
  for i in range(len(list_train)):
    image = imread(images_folder + "/" + list_image_names[i])
    plt.imsave(images_folder + "/" + str(i)+".jpg", image)


#Feature extraction, HOG
coef = 1
def feature_extraction(coef=coef):
  for i in range(int(len(paths)/4)):
    image = imread(images_folder + "/" + list_image_names[i])
    resized_arr = resize(image, (128*coef, 128*coef))
    plt.imsave(folder_resized_img + "/resized_" + str(i)+".jpg", resized_arr)
    fd, hog_img = hog(resized_arr, orientations=9, pixels_per_cell=(2, 2),
              cells_per_block=(1, 1), visualize=True, multichannel=True)
    plt.imsave(folder_hog_img + "/hog_" + str(i)+".jpg", hog_img, cmap="gray")                #save the hog images in black and white


#Loading HOG and resized images
def read_images2(path_names, folder_hog_img, hog=True):
  if hog:
    image = imread(folder_hog_img + "/hog_" + path_names[0] + ".jpg")
  else:
    image = imread(folder_hog_img + "/resized_" + path_names[0] + ".jpg")
  hog_images = np.expand_dims(image, axis=0)

  for i in path_names:
    if hog:
      image = imread(folder_hog_img + "/hog_" + i + ".jpg")
    else:
      image = imread(folder_hog_img + "/resized_" + i + ".jpg")
    image = np.expand_dims(image, axis=0)
    hog_images = np.append(hog_images, image, axis=0)
  return hog_images


"""Clustering by unmasking, implemented manually"""
def clustering_unmasking(features_hog_train, labels_train):
  clusters = {}
  true_labels = {}

  svc = SVC()

  for i in range(len(features_hog_train)):
    clusters[features_hog_train[i]] = labels_train[i]
    true_labels[features_hog_train[i]] = labels_train[i]

  s = 2
  n = 5

  while len(clusters.keys()) > len(labels_dict.keys()):
    #we start with n_of_smaples clusters
    scores = {}
    for i in range(len(clusters)):
      if len(clusters.keys()) == len(labels_dict.keys()):
        break
      for j in range(i+1, len(clusters)):
        cluster1 = clusters[i]
        cluster2 = clusters[j]
        #score = cluster_sim(cluster1, cluster2)
        accuracies = []
        for l in range(n):
          labels_i = [true_labels[k] for k in cluster1]
          labels_j = [true_labels[k] for k in cluster2]
          svc.fit(cluster1, labels_i)
          predicitions = svc.predict(cluster2)
          accuracies.append(sum(predictions == labels_j)/len(labels_j))
          indd = find_weights(svc.coef_, s)
          cluster1, cluster2, svc.coef_ = remove_weights(indd, cluster1, cluster2, svc.coef_)
        score = cluster_sim(cluster1, cluster2)
        scores[(cluster1, cluster2)] = score
    max_score_ind = np.argmax(scores.values())
    cluster1, cluster2 = scores.keys()[max_score_ind]
    clasa = clusters[cluster1]
    clusters.pop(cluster1)
    clusters.pop(cluster2)

    if len(cluster1.shape) < 2:
      cluster1 = np.expand_dims(cluster1, axis=0)
    if len(cluster2.shape) < 2:
      cluster2 = np.expand_dims(cluster2, axis=0)
    join = np.append(cluster1, cluster2, axis=0)
    clusters[join] = clasa
  return clusters

#Removes weights of highest value and returns the remaining weights and features
def remove_weights(indd, cluster1, cluster2, coef):
  coef = np.array([coef[i] for i in range(len(coef)) if i not in indd])
  cluster1 = np.array([cluster1[i] for i in range(len(cluster1))if i not in indd])
  cluster2 = np.array([cluster2[i] for i in range(len(cluster2)) if i not in indd])
  return cluster1, cluster2, coef

#Finds the weights that need to be removed
def find_weights(array, s):
  arr = []
  for i in range(len(array)):
    arr.append(abs(sum(array[i])))
  indd = np.argpartition(arr, -1*s)[-1*s:]
  return indd

"""Function that calculates cluster similarity"""
def cluster_sim(cluster1, cluster2):
  sum = 0
  comb = len(combinations(range(len(cluster1)+len(cluster2)), 2))
  for feature1 in cluster1:
    for feature2 in cluster2:
      sum += similarity(feature1, feature2)
  return sum/comb

"""Function that calculates data similarity after being reshaped as a 1-D array"""
#I decided to make my own function which will calculate the similarity for hog images, after encoding
def similarity(im1, im2):
  sum = 0
  for i in range(len(im1)):
    if abs(im1[i] - im2[i]) <=0.014:     #maximum difference is around 0.1, so 0.1/n_classes = 0.1/7
      sum += 1
  return sum/len(im1)

"""--------------------------------------------------------------------------"""

def encoding_images(encoder, array):
  features = encoder(array)
  return features


"""Extract patches for BOVW"""
def extract_patches(images_folder, paths_images, size_patch, n_patches, random_state=7):
  patches = []
  for i in paths_images:
    image = imread(images_folder + "/" + i + ".jpg")
    vw = extract_patches_2d(image, patch_size=size_patch, max_patches=n_patches, random_state=random_state)
    patches.append(vw)
  return np.array(patches)


"""Returns LBP histogram of BOVW patches"""
def lbp(image, radius=1, sampling_pixels=8):  # extracts features, similarly to hog
    if len(image.shape) > 2:
        image = rgb2gray(image)

    image = image.astype("float32") / 255.  # normalization
    lbp = feature.local_binary_pattern(image, sampling_pixels, radius, method="uniform")
    (histog, _) = np.histogram(lbp.ravel(), bins=np.arange(0, sampling_pixels + 3), range=(0, sampling_pixels + 2))
    histog = histog.astype("float32")
    histog /= (histog.sum() + 1e-6)  # adding a small epsilon as to not divide by 0
    return histog

"""Loading the training data"""

folder = "/content/gdrive/My Drive/Project2_PML"
train_folder = folder + "/archive"
images_folder = train_folder+"/animal_images"
csv_file = train_folder+"/animal_data_img.csv"

f = open(csv_file, "r")
f_csv = csv.reader(f, delimiter=",")
lists = list(f_csv)

header = lists[0]
list_train = lists[1:]
list_image_names = os.listdir(images_folder)
list_image_names.sort()


rename_images()

labels_dict = {}
for i in range(len(list_train)):
  label = list_train[i][1]
  if label not in labels_dict.keys():
    labels_dict[label] = len(labels_dict.keys())

labels = []
names = []
paths = []
for i in range(len(list_train)):
  names.append(list_train[i][0])
  labels.append(labels_dict[list_train[i][1]])
  paths.append(list_train[i][2])

print("Labels: {}".format(labels[:10]))
print("Names: {}".format(names[:10]))
print("Paths: {}".format(paths[:10]))
print("List image names: {}".format(list_image_names[:10]))

paths_images_train, paths_images_test, labels_train, labels_test = train_test_split([str(i) for i in range(int(len(paths)/4))], 
                                                                                    labels[:int(len(paths)/4)], test_size=0.15, random_state=42)

labels_dict = {}
for label in labels_train:
  if label not in labels_dict.keys():
    labels_dict[label] = len(labels_dict.keys())


print("paths_images_train: {}".format(paths_images_train[:10]))
print("Labels_train: {}".format(labels_train[:10]))

folder_resized_img = train_folder + "/image_resize2"
folder_hog_img = train_folder + "/hog_images2"


feature_extraction()

"""Plotting an example of a HOG image"""
f, ax = plt.subplots(1,3)
ax[0].axis("off")
ax[0].imshow(imread(images_folder + "/" + list_image_names[0]))

ax[1].axis("off")
ax[1].imshow(imread(folder_resized_img + "/resized_0.jpg"))

ax[2].axis("off")
ax[2].imshow(imread(folder_hog_img + "/hog_0.jpg"))


#Loading the saved resized and HOG images
train_res = read_images2(paths_images_train, folder_resized_img, False)
test_res = read_images2(paths_images_test, folder_resized_img, False)

train_hog = read_images2(paths_images_train, folder_hog_img)
test_hog = read_images2(paths_images_test, folder_hog_img)


autoencoder = ModelAutoencoder()

#normalizing data
train = train_res.astype('float32') / 255.
test = test_res.astype('float32') / 255.


autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')

#Training the autoencoder
autoencoder.fit(train, train,
                epochs=5,
                shuffle=True,
                validation_data=(test, test))

#saving the weights of the autoencoder
checkpoint = tf.train.Checkpoint(autoencoder)

save_path = checkpoint.save(train_folder + "/check/cp.ckpt")



encoder = ModelEncoder()

checkpoint2 = tf.train.Checkpoint(encoder)
#loading the saved weights to the encoder
checkpoint2.restore(train_folder + "/check/cp.ckpt-1")

#normalization of HOG data
hog_train = train_hog.astype('float32') / 255.
hog_test = test_hog.astype('float32') / 255.


#encoding the HOG data
features_hog_train = encoding_images(encoder, hog_train)
features_hog_test = encoding_images(encoder, hog_test)



"""Finding optimal value for epsilon for DBSCAN"""

nn = NearestNeighbors(n_neighbors=2)
nbrs = nn.fit(features_hog_train)
dist, ind = nbrs.kneighbors(features_hog_train)

dist = np.sort(dist, axis=0)
dist = dist[:,1]
plt.plot(dist)

#eps = 0.1375

#fitting the training data
dbscan = DBSCAN(eps=0.1375, min_samples=len(labels_dict.keys())).fit(features_hog_train)
predicted_l = dbscan.labels_
n_clusters = max(predicted_l) + 1                                                    #number of clusters



cluster_samples = dict()
for cluster_n in range(n_clusters):
    cluster_samples[cluster_n] = features_hog_train[predicted_l == cluster_n]


#predicting the test data
test_pred = []
for test_sample in features_hog_test:
    min_distances = []
    for cluster_n in range(n_clusters):
        dist = np.sum(np.square(cluster_samples[cluster_n] - test_sample), axis=1)
        cluster_min_distance = np.min(dist)
        min_distances.append(cluster_min_distance)
    test_pred.append(np.argmin(min_distances))
test_pred = np.array(test_pred)

#building the confusion matrix
confusion_matrix = np.zeros((len(labels_dict.keys()), n_clusters))
for train_label, predicted_label in zip(labels_train, predicted_l):
    confusion_matrix[train_label][predicted_label] += 1
    
translate = dict()
for i in range(n_clusters):
    translate[i] = np.argmax(confusion_matrix[:, i])

test_pred = np.array([
    translate[label]
    for label in test_pred
])

print("Accuracy of DBSCAN with HOG: {}".format(np.mean(test_pred == labels_test)))


"""Plotting the confusion matrix of DBSCAN with HOG"""
plt.imshow(confusion_matrix)
plt.colorbar(label='')


clusters = clustering_unmasking(features_hog_train, labels_train)

confusion_matrix = np.zeros((len(labels_dict.keys()), len(clusters.keys())))

#Since the data in clusters is not in the same order with the initial train_labels, we will build 2 new lists of labels and predictions
labels_train_new = []
predictions = []
for features in clusters.keys():
  for feature in features:
    i = features_hog_train.index(feature)
    labels_train_new.append(labels_train[i])
    predictions.append(clusters[features])

print("Accuracy of Clustering by Unmasking with HOG: {}".format(np.mean(predictions == labels_train)))

#building the confusion matrix of Clustering by Unmasking with HOG
for train_label, predicted_label in zip(labels_train_new, predictions):
    confusion_matrix[train_label][predicted_label] += 1

"""Plotting the confusion matrix of Clustering by Unmasking with HOG"""
plt.imshow(confusion_matrix)
plt.colorbar(label='')


"""Feature extraction with BOVW"""
size_patch = (15, 15)
n_patches = 5
random_state = 1
#Patch extraction
patches_train = extract_patches(images_folder, paths_images_train, size_patch, n_patches, random_state)
patches_test = extract_patches(images_folder, paths_images_test, size_patch, n_patches, random_state)

#Plotting a few patches of the same image
f, axarr = plt.subplots(1,5)
for i in range(5):
  axarr[i].axis("off")
  axarr[i].imshow(patches_train[0][i])


#Reshaping the patches
patches_train_res = patches_train.reshape((patches_train.shape[0] * patches_train.shape[1], size_patch[0], size_patch[1], 3))
patches_test_res = patches_test.reshape((patches_test.shape[0] * patches_test.shape[1], size_patch[0], size_patch[1], 3))

lbp_train, lbp_test = [], []

#Extracting the histogram with LBP
for vw in patches_train_res:
  feat = lbp(vw, 2, 8)
  lbp_train.append(feat)

for vw in patches_test_res:
  feat = lbp(vw, 2, 8)
  lbp_test.append(feat)  

lbp_train = np.array(lbp_train)
lbp_test = np.array(lbp_test)

"""Finding the optimal Epsilon for DBSCAN"""
nn = NearestNeighbors(n_neighbors=2)
nbrs = nn.fit(lbp_train)
distances, indices = nbrs.kneighbors(lbp_train)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
#eps = 0.02 , 0.045

#DBSCAN with BOVW, dictionary
n_dic = 25 # size of the dictionary
random_state = 1
dbscan_bovw = DBSCAN(eps=0.045, min_samples=len(labels_dict.keys())).fit(lbp_train)

plt.scatter(lbp_train[:, 0], lbp_train[:, 1], c=dbscan_bovw.labels_)
plt.title('2 LBP components and its labels')
plt.axis('off')


image_feats = []

predicted_l = dbscan_bovw.labels_
n_clusters = max(predicted_l) + 1                                                    #number of clusters


cluster_samples = dict()
for cluster_n in range(n_clusters):
    cluster_samples[cluster_n] = lbp_train[predicted_l == cluster_n]


#predicting test data
test_pred = []
for test_sample in lbp_test:
    min_dist = []
    for cluster_n in range(n_clusters):
        dist = np.sum(np.square(cluster_samples[cluster_n] - test_sample), axis=1)
        cluster_min_distance = np.min(dist)
        min_dist.append(cluster_min_distance)
    test_pred.append(np.argmin(min_dist))
test_predictions = np.array(test_pred)


#building the confusion matrix
confusion_matrix = np.zeros((len(labels_dict.keys()), n_clusters))
for train_label, predicted_label in zip(labels_train, predicted_l):
    confusion_matrix[train_label][predicted_label] += 1
    
translate = dict()
for i in range(n_clusters):
    translate[i] = np.argmax(confusion_matrix[:, i])

test_predictions = np.array([
    translate[label]
    for label in test_predictions
])
print("Accuracy of DBSCAN with BOVW: {}".format(np.mean(test_predictions == labels_test)))

#Plotting the confusion matrix of DBSCAN with BOVW
plt.imshow(confusion_matrix)
plt.colorbar(label='')

"""Clustering by Unmasking with BOVW"""
clusters = clustering_unmasking(lbp_train, labels_train)
confusion_matrix = np.zeros((len(labels_dict.keys()), len(clusters.keys())))

labels_train_new = []
predictions = []
for features in clusters.keys():
  for feature in features:
    i = features_hog_train.index(feature)
    labels_train_new.append(labels_train[i])
    predictions.append(clusters[features])


print("Accuracy of Clustering by Unmasking with BOVW: {}".format(np.mean(predictions == labels_train)))


#building the confusion matrix
for train_label, predicted_label in zip(labels_train_new, predictions):
    confusion_matrix[train_label][predicted_label] += 1

#plotting the confusion matrix of Clustering by unmasking with BOVW
plt.imshow(confusion_matrix)
plt.colorbar(label='')