#this is working very well as is: ~80-85% accuracy
import nibabel as nib
import numpy as np
import pandas as pd
import json
import os
import time
import sys
import random
import tensorflow as tf
from collections import OrderedDict
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

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
                dat_2d = dat.reshape((dat.shape[0]*dat.shape[1]), dat.shape[2])
                dat_2d = dat_2d.transpose()
                v[d.split('/')[-1]]=dat_2d
    print('Done.')
    return v

def get_labels(f):
    print('Loading labels...',end='')
    t_df=pd.read_csv(f)
    #sample n200 of each class
    f=t_df[t_df.label=='F'].sample(n=200)
    m=t_df[t_df.label=='M'].sample(n=200)
    m_test=m.sample(n=40)
    m_train=m[m.subject.isin(m_test.subject) == False]
    f_test=f.sample(n=40)
    f_train=f[f.subject.isin(f_test.subject) == False]
    train=pd.concat([m_train,f_train])
    test=pd.concat([m_test,f_test])
    subjs_train=train.subject
    subjs_test=test.subject
    cnames=train.label.unique()
    mapping = dict(zip(cnames, range(len(cnames))))
    labels_train=train[['label']]
    labels_train=labels_train.replace({'label':mapping})
    #coerce dtype obj->int
    labels_train=labels_train.astype(str).astype(int)
    labels_train=labels_train['label'].values
    labels_test=test[['label']]
    labels_test=labels_test.replace({'label':mapping})
    #coerce dtype obj->int                                                                                                                                   
    labels_test=labels_test.astype(str).astype(int)
    labels_test=labels_test['label'].values
    print('Done.')
    #try randomizing the order
    z=list(zip(subjs_train,labels_train))
    random.shuffle(z)
    subjs_train,labels_train=zip(*z)
    z=list(zip(subjs_test,labels_test))
    random.shuffle(z)
    subjs_test,labels_test=zip(*z)
    return [labels_train,labels_test, subjs_train, subjs_test]

def format_data(d):
    print('Formatting numpy array...',end='')
    #default MNI mapped to 2d
    s=np.empty([len(d),91,9919])
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

def Conv(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=None):
    if input_shape:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation, input_shape=input_shape)
    else:
        return Conv3D(filters=filters, kernel_size=kernel_size, padding='Same', activation=activation)

def build_model():
    model = Sequential()
    model.add(Conv(8, (3,3,3), input_shape=(91,109,91,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool3D(pool_size=(2,2,2)))
    model.add(Dropout(0.25))
    #model.add(Conv(32, (3,3,3)))
    #model.add(MaxPool3D())
    #model.add(Dropout(0.25))
    model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def shuffle_splits(a,b,c,d,split):
    if not split:
        return [a,b,c,d]
    ba=tf.concat((b,a),0)
    train=ba[0:320]
    test=ba[320:]
    dc=np.concatenate([d,c])
    l_train=dc[0:320]
    l_test=dc[320:]
    return [train,test,l_train,l_test]

def classify_images(imgs,labels):
    labels=np.asarray(labels)
    gen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)
    tot=len(labels)
    accuracies = []
    info={}
    for split in range(0,5):
        info[split]=[]
        imgs,labels=shuffle_splits(imgs,labels,split)
        train=int(tot*0.8)
        train_labels=labels[0:train]
        train_images=imgs[0:train]
        test_labels=labels[train:]
        test_images=imgs[train:]
        #y_train = to_categorical(train_labels, num_classes=2)
        y_test = to_categorical(test_labels, num_classes=2)
        #X_train, X_val, y_train, y_val = train_test_split(train_images, y_train, test_size=0.15, random_state=42)
        model=build_model()
        epochs=20
        batch_size=64
        scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=1e-5)
        model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=["accuracy"])
        model.fit(train_images, train_labels, validation_split=0.2, epochs=epochs)
        #model.fit_generator(gen.flow(X_train, y_train, batch_size=batch_size),
        #            epochs=epochs, validation_data=(X_val, y_val),
        #                    verbose=2, steps_per_epoch=X_train.shape[0]//batch_size,callbacks=[scheduler])
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        for layer in model.layers:
            info[split].append(layer.get_weights())
        accuracies.append(test_acc)
    return accuracies, info 

def main():
    t=time.time()
    [labels_train, labels_test, subj_train, subj_test] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../classifiers/SN_betaweights_n812')
    imgs_train = load(hdrs,subj_train)
    imgs_test=load(hdrs,subj_test)
    all_images_train = format_data(imgs_train)
    all_images_test = format_data(imgs_tes
    [accuracies, info] = classify_images(all_images,all_labels)
    print("All accuracies:", accuracies)
    print("Avg accuracy:", sum(accuracies)/len(accuracies))
    try:
        import cPickle as pickle
    except ImportError:  # python 3.x
        import pickle
    with open('data.p', 'wb') as fp:
        pickle.dump(info, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #j_save(all_images,all_labels,mapping)
    print("It took %ds to run" % (time.time() - t))
