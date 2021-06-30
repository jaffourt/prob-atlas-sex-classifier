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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

def classify_images(imgs,labels,random_grid):
    labels=np.asarray(labels)
    tot=len(labels)
    for split in range(0,5):
        imgs,labels=shuffle_splits(imgs,labels,split)
        train=int(tot*0.8)
        train_labels=labels[0:train]
        #print('M - training set: %d/%d' % (np.sum(train_labels),train))
        train_images=imgs[0:train]
        test_labels=labels[train:]
        test_images=imgs[train:]
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, 
                param_distributions = random_grid, n_iter = 10, cv = 3, 
                                       verbose=2, random_state=42)
        #dt = RandomForestClassifier(n_estimators=500)
        rf_random.fit(np.reshape(train_images, (train_images.shape[0], -1)),train_labels)
        print('Best Params:')
        print(rf_random.best_params_)
        best_random_model=rf_random.best_estimator_
        print('\nTest accuracy:', best_random_model.score(np.reshape(test_images, (test_images.shape[0], -1)),test_labels))

def random_search(all_images,all_labels):
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    classify_images(all_images,all_labels,random_grid)

def main():
    t=time.time()
    [all_labels, mapping,subj] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../classifiers/SN_betaweights_n812')
    imgs = load(hdrs,subj)
    all_images = format_data(imgs)
    random_search(all_images,all_labels)
    print("It took %ds to run" % (time.time() - t))

if __name__ == '__main__':
    main()
