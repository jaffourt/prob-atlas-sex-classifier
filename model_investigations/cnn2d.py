import nibabel as nib
import numpy as np
import pandas as pd
import json
import os
import time
import sys
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from scipy import stats

def iterate_dir(p):
    print('Finding nifti files...',end='')
    d=[]
    for f in os.listdir(p):
        fp=os.path.join(p,f)
        d.append(fp)
    print('Done.')
    return d

def load(l, subjs):
    print('Loading nifti files...',end='')
    v=OrderedDict()
    for s in subjs:
        for d in l:
            if s in d:
                img=nib.load(d)
                dat=img.get_fdata()
                dat[np.isnan(dat)]=0
                #dat_2d = dat.reshape((dat.shape[0]*dat.shape[1]), dat.shape[2])
                #dat_2d = dat_2d.transpose()
                v[d.split('/')[-1]]=dat
    print('Done.')
    return v

def get_labels(f):
    print('Loading labels...',end='')
    df=pd.read_csv(f)
    subjs=df.subject
    cnames=df.label.unique()
    mapping = dict(zip(cnames, range(len(cnames))))
    labels=df[['label']]
    labels=labels.replace({'label':mapping})
    #coerce dtype obj->int
    labels=labels.astype(str).astype(int)
    labels=labels['label'].values
    print('Done.')
    #try randomizing the order
    z=list(zip(subjs,labels))
    random.shuffle(z)
    subjs,labels=zip(*z)
    return [labels, mapping, subjs]

def format_data(d):
    print('Formatting numpy array...',end='')
    #default MNI mapped to 2d
    s=np.empty([len(d),91,109,91])
    i=0
    for k in d:
        #temp=np.expand_dims(d[k])
        #z=stats.zscore(d[k])
        #z[np.isnan(z)]=0
        #s[i]=d[k]/np.max(d[k])
        s[i]=d[k]
        i+=1
    print('Done.')
    return s

def Conv(filters=16, kernel_size=(3,3), activation='relu', input_shape=None):
    if input_shape:
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv2D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)

def build_model():
    model = Sequential()
    model.add(Conv(8, (3,3), input_shape=(91,109,91)))
    model.add(Conv(16, (3,3)))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv(32, (3,3)))
    model.add(Conv(64, (3,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

def shuffle_splits(imgs,labels,split):
    if split == 0:
        return imgs, labels
    i=int(len(labels)*.8)
    a=imgs[0:i]
    b=imgs[i:]
    c=labels[0:i]
    d=labels[i:]
    ba=tf.concat((b,a),0)
    dc=np.concatenate([d,c])
    return [ba, dc]

def classify_images(imgs,labels):
    labels=np.asarray(labels)
    gen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)
    tot=len(labels)
    for split in range(0,5):
        imgs,labels=shuffle_splits(imgs,labels,split)
        train=int(tot*0.8)
        train_labels=labels[0:train]
        train_images=imgs[0:train]
        test_labels=labels[train:]
        test_images=imgs[train:]
        y_train = to_categorical(train_labels, num_classes=2)
        y_test = to_categorical(test_labels, num_classes=2)
        X_train, X_val, y_train, y_val = train_test_split(train_images, y_train, test_size=0.15, random_state=42)
        model=build_model()
        epochs=20
        batch_size=64
        scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
        model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
        model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, y_val),
                            verbose=2, steps_per_epoch=X_train.shape[0]//batch_size,callbacks=[scheduler])
        test_loss, test_acc = model.evaluate(test_images,  y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

def main():
    t=time.time()
    [all_labels, mapping,subj] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../classifiers/SN_betaweights_n812')
    imgs = load(hdrs,subj)
    all_images = format_data(imgs)
    classify_images(all_images,all_labels)
    #j_save(all_images,all_labels,mapping)
    print("It took %ds to run" % (time.time() - t))

if __name__ == '__main__':
    main()
