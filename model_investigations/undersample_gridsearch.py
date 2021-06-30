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
from sklearn.model_selection import GridSearchCV

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

def make_random_grid():
    n_estimators = [int(x) for x in np.linspace(start = 300, stop = 700, num = 15)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 120, num = 10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    param_grid = {
        'bootstrap': [False],
        'max_depth': [80, 90, 100, 110, None],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_samples_split': [2, 8, 10, 12],
        'n_estimators': [int(x) for x in np.linspace(start = 300, stop = 700, num = 15)]
    }
    rf=RandomForestClassifier()
    #grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 75, cv = 3, verbose=0, random_state=42, n_jobs = -1)
    return rf_random

def classify_images(imgs_train, imgs_test, labels_train, labels_test):
    labels_train=np.asarray(labels_train)
    labels_test=np.asarray(labels_test)
    for split in range(0,5):
        imgs_train,imgs_test,labels_train,labels_test=shuffle_splits(imgs_train,imgs_test,labels_train,labels_test,split)
        rf_random=make_random_grid()
        rf_random.fit(np.reshape(imgs_train, (imgs_train.shape[0], -1)), labels_train)
        print(rf_random.best_params_)
#        with open("sample_file.json", "r+") as file:
#            data = json.load(file)
#            data.update(rf_random.best_params_)
#            file.seek(0)
#            json.dump(data, file)
        best_random = rf_random.best_estimator_
        best_random.score(np.reshape(imgs_test, (imgs_test.shape[0], -1)),labels_test)
        print('Test accuracy:', best_random.score(np.reshape(imgs_test, (imgs_test.shape[0], -1)),labels_test))
        print("M - Train: %d\nM - Test %d\n\n" % (np.sum(labels_train),np.sum(labels_test)))
        #dt = RandomForestClassifier(n_estimators=450)
#       dt.fit(np.reshape(imgs_train, (imgs_train.shape[0], -1)),labels_train)
        #model=build_model(train_images.shape)
        #model.fit(train_images, train_labels, validation_split=0.2, epochs=5)
        #print('\nTest accuracy:', dt.score(np.reshape(imgs_test, (imgs_test.shape[0], -1)),labels_test))
    #predict(model,test_images,test_labels)
        
def main():
    t=time.time()
    [labels_train, labels_test, subj_train, subj_test] = get_labels('classes_restricted.csv')
    hdrs = iterate_dir('../classifiers/SN_betaweights_n812')
    imgs_train = load(hdrs,subj_train)
    imgs_test=load(hdrs,subj_test)
    all_images_train = format_data(imgs_train)
    all_images_test = format_data(imgs_test)
    classify_images(all_images_train, all_images_test, labels_train, labels_test)
    print("It took %ds to run" % (time.time() - t))

if __name__ == '__main__':
    main()
