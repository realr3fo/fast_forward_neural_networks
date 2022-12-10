# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 10:53:43 2022

@author: Alberto Ortiz
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.utils import to_categorical

def to_label(M, y):

	if M == 2:
	    # round predictions
	    y_ = [round(x[0]) for x in y]        
	else:
	    # take max output
	    y_ = np.argmax(y, axis=1)
	    
	return y_
    
def plot_class(c, X, y):
    # plot samples of 'X' for class 'c'
    
    i = np.where(y == c)[0]
    plt.scatter(X[i,0],X[i,1])

def show_class_map_(model, X, y):
    
    if X.shape[1] != 2:
        print('This dataset is not two-dimensional ...')
        return
    
    M = len(np.unique(y))
    
    # plot data
    plt.figure()
    for c in range(M):
        plot_class(c, X, y)
    plt.axis('equal')

    # plot the decision boundaries
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create the grid to evaluate the model at discrete feature space points
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    ZZ_ = model.predict(xy)
    ZZ  = to_label(M, ZZ_)

    # plot the boundaries
    for c in range(M):
        ax.contour(XX, YY, ZZ_[:,c].reshape(XX.shape), levels=[0.5], alpha=0.5, linestyles=['--'])
        
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.title('contour map')
    plt.savefig('results/ic_lab1_c.png')
    plt.show()
    
    # plot the classification map
    plt.figure()
    plt.imshow(ZZ.reshape(XX.shape).T, origin='lower', extent=(xlim[0], xlim[1], ylim[0], ylim[1]), cmap='RdYlGn')
    plt.colorbar()
    for c in range(M):
        plot_class(c, X, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.title('classification map')
    plt.savefig('results/ic_lab1_m.png')
    plt.show()    

def do_show(model, history, X_train, y_train, X_test, y_test):
    
    # RESULTS
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print(accuracy[-1], val_accuracy[-1])
                               
    plt.figure()
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.title('loss function')
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.plot(accuracy, label='train')
    plt.plot(val_accuracy, label='validation')
    plt.title('accuracy')
    plt.legend()
    plt.show(block=False)

    M = len(np.unique(y_train))

    y_pred_ = model.predict(X_train)
    y_pred = to_label(M, y_pred_)
    cm = confusion_matrix(y_train, y_pred)
    print(cm)
    print('accuracy = %f' % (accuracy_score(y_train, y_pred)))
    # print('accuracy = %f' % ((cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])))
    
    y_train_ = to_categorical(y_train)
    score = model.evaluate(X_train, y_train_, verbose=0)
    print("Train loss:", score[0])
    print("Train accuracy:", score[1])    
    
    show_class_map_(model, X_train, y_train)

    y_pred_ = model.predict(X_test)
    y_pred = to_label(M, y_pred_)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('accuracy = %f' % (accuracy_score(y_test, y_pred)))
    # print('accuracy = %f' % ((cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])))

    y_test_ = to_categorical(y_test)
    score = model.evaluate(X_test, y_test_, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])    

from sklearn.model_selection import KFold
def nfoldcv_keras(model, X, y):

	M = len(np.unique(y))
	n_folds = 5
	
	results = np.zeros(n_folds)
	kf = KFold(n_splits=n_folds, shuffle=True)
	for k, (train, test) in enumerate(kf.split(X)):
		# training
		X_train, y_train = X[train], y[train]
		y_train_ = to_categorical(y_train)
		model.fit(X_train, y_train_, epochs=100, batch_size=10, 
							validation_split=0.2, verbose=0)
		# testing
		X_test, y_test = X[test], y[test]
		y_pred_ = model.predict(X_test)
		y_pred = to_label(M, y_pred_)
		results[k] = accuracy_score(y_test, y_pred)

	return results

