#coding=utf-8
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline
from collections import Counter
import numpy as np
index=np.random.randint(n_train)
plt.imshow(X_train[index],cmap='gray')
print(y_train[index])
 
sign_count_test = Counter(y_train)
 
range_x = np.array(range(n_classes))
range_y = [sign_count_test[i] for i in range_x]
plt.figure(figsize=(9,5))
plt.bar(range_x,range_y)
plt.xticks(list(range(n_classes)))
plt.xlabel("class")
plt.ylabel("numbers")
plt.title("the train data distribution")
plt.show
 
sign_count_valid = Counter(y_valid)
 
range_x = np.array(range(n_classes))
range_y = [sign_count_valid[i] for i in range_x]
plt.figure(figsize=(9,5))
plt.bar(range_x,range_y)
plt.xticks(list(range(n_classes)))
plt.xlabel("class")
plt.ylabel("numbers")
plt.title("the valid data distribution")
plt.show
 
sign_count_test = Counter(y_test)
 
range_x = np.array(range(n_classes))
range_y = [sign_count_test[i] for i in range_x]
plt.figure(figsize=(9,5))
plt.bar(range_x,range_y)
plt.xticks(list(range(n_classes)))
plt.xlabel("class")
plt.ylabel("numbers")
plt.title("the test data distribution")
plt.show

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2
def normalized(images):
    xmin = np.min(images)
    xmax = np.max(images)
    image = (images - xmin)/(xmax-xmin)
    return image
 
def grayscale(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return np.expand_dims(gray,axis = 2)
 
def preprocess(images):
    list1 = []
    for image in images:
        list1.append(grayscale(image))
    list1 = np.array(list1)
    return normalized(list1)
 
def one_hot(images):
    list1 = np.zeros((len(images),n_classes))
    for i,label in enumerate(images):
        list1[i][label] = 1
    return list1

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from sklearn.utils import shuffle

X_train,y_train = shuffle(X_train,y_train)
X_train = preprocess(X_train)
y_train = one_hot(y_train)
X_valid = preprocess(X_valid)
y_valid = one_hot(y_valid)
X_test = preprocess(X_test)
y_test = one_hot(y_test)
print(X_train.shape,y_train.shape)

inputs = tf.placeholder(tf.float32,shape = [None,32,32,1],name = 'inputs')
labels = tf.placeholder(tf.int32,shape = [None,n_classes],name = 'labels')

#from tensorflow.contrib.layers import flatten
#from keras.layers.core import Flatten
from numpy import *
#from tensorflow.keras.layers import Flatten


#from tensorflow.keras.layers import flatten

def model(images,add_dropout = True):
    mu = 0
    sigma = 0.1
    dropout = 0.5 
    
    conv1_W = tf.Variable(tf.truncated_normal(shape = (5,5,1,12),mean = mu,stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1 = tf.nn.conv2d(images,conv1_W,strides=[1,1,1,1],padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID',name = 'conv1')
    #14x14x12
    
    conv2_W = tf.Variable(tf.truncated_normal(shape = (5,5,12,24),mean = mu,stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(24))
    conv2 = tf.nn.conv2d(conv1,conv2_W,strides = [1,1,1,1],padding='VALID',name = 'conv2') + conv2_b
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #5,5,24
    
    #fc0 = flatten(conv2)
    #conv=array(conv2)
    #fc0=conv2.flatten
    fc0=tf.layers.Flatten()(conv2)
    #fc0=tf.Variable(fc0)
    #600
    
    fc1_W = tf.Variable(tf.truncated_normal(shape = (600,400),mean = mu,stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1 = tf.matmul(fc0,fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)
    
    if add_dropout:
        fc1 = tf.nn.dropout(fc1,dropout)
        
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(120))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    if add_dropout:
        fc2 = tf.nn.dropout(fc2,dropout)
    
    fc3_W = tf.Variable(tf.truncated_normal(shape = (120,84),mean = mu,stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(84))
    fc3 = tf.matmul(fc2,fc3_W) +fc3_b
    fc3 = tf.nn.relu(fc3)
    
    if add_dropout:
        fc3 = tf.nn.dropout(fc3,dropout)
    
    fc4_w = tf.Variable(tf.truncated_normal(shape = (84,43),mean=mu,stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc3,fc4_w) + fc4_b
    
    return logits
 
def get_batches(X,y,batch_size=128):
    length=len(X)
    n_batches=length//batch_size+1
    for i in range(n_batches):
        yield X[batch_size*i:min(length,batch_size*(i+1))], y[batch_size*i:min(length,batch_size*(i+1))]
        
EPOCHS = 40
max_acc = 0
save_model_path = 'Traffice_sign_classifier'
logits = model(inputs)
logits = tf.identity(logits,name = 'logits')
cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels,logits),name = 'cost')

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(labels,1),tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')
 
with tf.Session()  as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        for i,(X,y) in enumerate(get_batches(X_train,y_train)):
            sess.run(optimizer,feed_dict = {inputs:X,labels:y})
            if i%50==0:
                valid_acc=sess.run(accuracy, feed_dict={inputs:X_valid, labels:y_valid})
                train_acc=sess.run(accuracy, feed_dict={inputs:X, labels:y})
        print('epoch : ',epoch+1,' training accuracy is : ',train_acc,' valid accuracy is :',valid_acc)
        if valid_acc > max_acc:
            max_acc = valid_acc
            saver = tf.train.Saver(max_to_keep=1)
            save_path = saver.save(sess,save_model_path)

import pandas as pd
loaded_graph = tf.Graph()
save_model_path = './Traffice_sign_classifier'

with tf.Session(graph=loaded_graph) as sees:
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sees, save_model_path)
    loaded_inputs=loaded_graph.get_tensor_by_name('inputs:0')
    loaded_labels=loaded_graph.get_tensor_by_name('labels:0')
    loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
    loaded_acc=loaded_graph.get_tensor_by_name('accuracy:0')
    
    test_acc=sees.run(loaded_acc,feed_dict={loaded_inputs:X_test,loaded_labels:y_test})
    print('The test accuracy is:',test_acc)


