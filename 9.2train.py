from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from sklearn.datasets import fetch_openml

# import MNIST and re-shape input/output data
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
train_X = x.reshape(-1, 28, 28)
train_X = train_X.astype(np.uint8)
train_y = y.astype(np.uint8)

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28,28,1]))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss=categorical_crossentropy,
             optimizer=Adadelta(lr=0.1),
             metrics=['accuracy'])

# convolution
# Laplace extension
sz = np.array([[1, 1, 1],
            [1,-8, 1],
            [1, 1, 1]])
train_X2 = [signal.convolve2d(i, sz, mode="same") for i in train_X]

# compare two graph
plt.imshow(train_X2[0], cmap=plt.get_cmap('gray'))
plt.show()
plt.imshow(train_X[0], cmap=plt.get_cmap('gray'))
plt.show()

# reshape to fit model dimension 4
train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')

train_X2 = np.array(train_X2).reshape(-1, 28, 28, 1)
train_X2 = train_X2.astype('float32')


# transfer data range
train_X /= 255
train_X2 /= 255
train_y = to_categorical(train_y, 10)

# Calculate, fit into model
history = model.fit(train_X, train_y, batch_size=200, shuffle=False,epochs=20, validation_split=2/7)
history2 = model.fit(train_X2, train_y, batch_size=200, shuffle=False,epochs=20, validation_split=2/7)

# cost plot
base_acc = history.history['accuracy'] 
edge_acc = history2.history['accuracy']
base_loss = history.history['loss']
edge_loss = history2.history['loss']
epochs = range(1, len(base_acc) + 1) 
plt.plot(epochs, base_loss,'bo-', label='mini-batch gradient cost')
plt.plot(epochs, edge_loss,'g*-', label=' cost after edge detection convolution')
plt.title('cost value')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.legend()
plt.savefig("cost_value.png",dpi=200,bbox_inches = 'tight')
plt.show()
# misclassification
base_miss = [(1-i)*50000 for i in base_acc]
edge_miss = [(1-i)*50000 for i in edge_acc]
plt.plot(epochs, base_miss,'ro-', label='mini_batch gradient misclassifications')
plt.plot(epochs, edge_miss,'b*-', label='edge detection after convolution misclassifications')
plt.title('number of missclassifications')
plt.xlabel('iteration')
plt.ylabel('misclassification number')
plt.legend()
plt.savefig("missclassifications.png",dpi=200,bbox_inches = 'tight')
plt.show()
# compare base cost with Validation
'''
acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1) 
plt.plot(epochs, loss,'b*-', label='Training loss')
plt.plot(epochs, val_loss,'k*-', label='Validation loss')
plt.title('cost value')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend()
plt.savefig("cost_value.png",dpi=200,bbox_inches = 'tight')
plt.show()
############################ compare base misclassification with Validation 
miss = [(1-i)*50000 for i in acc]
val_miss = [(1-i)*20000 for i in acc]
plt.plot(epochs, miss,'b*-', label='Training missclassifications')
plt.plot(epochs, val_miss,'k*-', label='Validation missclassifications')
plt.title('number of missclassifications')
plt.xlabel('iteration')
plt.ylabel('number')
plt.legend()
plt.savefig("missclassifications.png",dpi=200,bbox_inches = 'tight')
plt.show()
'''

