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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
                dat_3d=img.get_fdata()
                dat_3d[np.isnan(dat_3d)]=0
                dat_2d = dat_3d.reshape((dat_3d.shape[0]*dat_3d.shape[1]), dat_3d.shape[2])
                dat_2d = dat_2d.transpose()
                dat_1d = dat_2d.reshape(dat_2d.shape[0]*dat_2d.shape[1])
                v[d.split('/')[-1]]=dat_1d
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
    s=np.empty([len(d),902629])
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

def build_model(x_train,y_train):
    param_grid = {'clf__kernel': ['linear'], 'clf__C': [x*0.001 + 0.001 for x in range(500)]}
    cv_steps = [('scaler', StandardScaler()), ('clf', SVC(max_iter=100000))]
    iterations=100
    cv_pipeline = Pipeline(cv_steps)
    nested_scores = np.zeros(iterations)
    best_params = ['']*iterations
    for iter in range(iterations):
        print("Nested CV - Loop %d" % (iter + 1))
        inner_cv = StratifiedKFold(n_splits = 5, shuffle = True)
        outer_cv = StratifiedKFold(n_splits = 5, shuffle = True)
        clf = GridSearchCV(cv_pipeline, param_grid = param_grid, cv = inner_cv, 
                           scoring='roc_auc', iid = False, refit = True, verbose=0)
        clf.fit(x_train, y_train)
        best_params[iter] = clf.best_params_
        nested_score = cross_val_score(clf, X = x_train, y = y_train.ravel(), cv = outer_cv, 
                                       scoring = 'roc_auc', verbose = 0)
        nested_scores[iter] = nested_score.mean()
        print(nested_score.mean())
    with open('best_params.txt', 'w') as f:
        for listitem in best_params:
            f.write('%s\n' % listitem)
    cv_params = pd.read_csv('best_params.txt', sep='\s+', header=None)
    cv_params = cv_params[1].str.replace(',','')
    best_c = np.zeros(iterations)
    for i in range(len(best_c)):
        best_c[i] = eval(cv_params[i])
    optimised_c[idx] = np.mean(best_c)
    print("Optimised C = %1.4f" %(optimised[idx]))
    model = SVC(C = optimised_c[idx], kernel='linear', probability=True);
    return model

def predict(model,test_images,test_labels):
    print('Creating probability model...',end='')
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    print('Done.')
    correct=0
    for i in range(len(test_labels)):
        if np.argmax(predictions[i])==test_labels[i]:
            correct+=1
    print('Prediction accuracy: %d/%d' % (correct,len(test_labels)))

def optimize_params():
    return 0

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
        train_labels=labels[0:train]
        #print('M - training set: %d/%d' % (np.sum(train_labels),train))
        train_images=imgs[0:train]
        test_labels=labels[train:]
        test_images=imgs[train:]
        svm = build_model(train_images,train_labels)
        scaler = StandardScaler()
        scaler.fit(train_images)
        x_train_t = scaler.transform(train_images)
        x_test_t = scaler.transform(test_images)
        svm.fit(x_train_t, train_labels);
        svm.score(x_test_t,test_labels)
        #svm.fit(np.reshape(train_images, (train_images.shape[0], -1)),train_labels)
        #model=build_model(train_images.shape)
        #model.fit(train_images, train_labels, validation_split=0.2, epochs=5)
        #print('\nTest accuracy:', svm.score(np.reshape(test_images, (test_images.shape[0], -1)),test_labels))
    #predict(model,test_images,test_labels)
        
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
