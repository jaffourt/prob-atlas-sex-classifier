import nibabel as nib
import numpy as np
import pandas as pd
import json
import os
import time
import sys
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
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
                dat=dat.reshape(-1,dat.shape[1],1)
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
    s=np.empty([len(d),415140,1])
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

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(415140,1)),
        keras.layers.Dense(20),
        keras.layers.Dense(16),
        keras.layers.Dense(2),
        ])
#    model.summary()
    model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
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
    tot=len(labels)
    for split in range(0,5):
        imgs,labels=shuffle_splits(imgs,labels,split)
        train=int(tot*0.8)
        #train_labels=to_categorical(labels[0:train])
        train_labels=labels[0:train]
        train_images=imgs[0:train]
        #test_labels=to_categorical(labels[train:])
        test_labels=labels[train:]
        test_images=imgs[train:]
        model=build_model()
        model.fit(train_images, train_labels, validation_split=0.2, epochs=5)
        test_lost, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

def main():
    t=time.time()
    [all_labels, mapping,subj] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../classifiers/SN_betaweights_noNaN')
    imgs = load(hdrs,subj)
    all_images = format_data(imgs)
    classify_images(all_images,all_labels)
    #j_save(all_images,all_labels,mapping)
    print("It took %ds to run" % (time.time() - t))

if __name__ == '__main__':
    main()
