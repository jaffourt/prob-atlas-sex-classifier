import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import pickle
import sys


class Dataset:
    def __init__(self, **kwargs):
        self.file_dir = kwargs.get('dir', None)
        self.files = kwargs.get('files', [])
        self.pkl = kwargs.get('data', [])
        self.data = []
        self.labels = []
        self.initialize()

    def initialize(self):
        if not self.files and self.file_dir:
            print('Loading nifti files from dir: %s' % self.file_dir)
            self.files = glob(self.file_dir + '/*.nii')
            print('Found %d files in directory' % len(self.files))
            self.load_nii()
        elif self.files:
            print('Using file paths provided. N= %d' % len(self.files))
            self.load_nii()
        elif self.pkl:
            print('Loading data structure: %s' % self.pkl)
            with open('dataset.pkl', 'rb') as f:
                d = pickle.load(f)
                self.labels = d[0]
                self.data = d[1]
        else:
            raise AttributeError('No was data provided.')

    def preprocess(self):
        mask = nib.load('language_ROIs.nii').get_data() > 0
        data = []
        labels = []
        print('Preprocessing... ', end='')
        for label, mmap in self.data:
            masked = mmap[mask]
            # masked[np.isnan(masked)] = 0
            data.append(masked)
            labels.append(label)

        # scale data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data[np.isnan(data)] = 0

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(X)

        # labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(labels)

        self.labels = le.transform(labels)
        self.data = np.array(data)
        print('Done')

    def save_pkl(self):
        data = [self.labels, self.data]
        pickle.dump(data, open('dataset.pkl', 'wb'))

    def shuffle(self):
        shuffled_ind = np.random.permutation(len(self.labels))
        self.labels = self.labels[shuffled_ind]
        self.data = self.data[shuffled_ind]

    def load_nii(self):
        classes = self.classes_from_csv()
        print('Loading nifti...')
        for f in tqdm(self.files):
            nib.load(f).get_data()
            self.data.append((classes[f.split('/')[-1].split('.nii')[0]], nib.load(f).get_data()))

    @staticmethod
    def classes_from_csv():
        df = pd.read_csv('classes.csv')
        return pd.Series(df.label.values, index=df.subject).to_dict()


def load(l):
    print('Loading nifti files...', end='')
    v = {}
    for d in l:
        img = nib.load(d)
        dat = img.get_fdata()
        dat[np.isnan(dat)] = 0
        dat_2d = dat.reshape((dat.shape[0] * dat.shape[1]), dat.shape[2])
        dat_2d = dat_2d.transpose()
        v[d.split('/')[-1]] = dat_2d
    print('Done.')
    return v


def get_labels(f):
    print('Loading labels...', end='')
    df = pd.read_csv(f)
    cnames = df.label.unique()
    mapping = dict(zip(cnames, range(len(cnames))))
    labels = df[['label']]
    labels = labels.replace({'label': mapping})
    # coerce dtype obj->int
    labels = labels.astype(str).astype(int)
    labels = labels['label'].values
    print('Done.')
    return [labels, mapping]


def format_data(d):
    print('Formatting numpy array...', end='')
    # default MNI mapped to 2d
    s = np.empty([len(d), 91, 9919])
    i = 0
    for k in d:
        s[i] = d[k]
        i += 1
    print('Done.')
    maxval = np.max(s)
    # normalize
    s = s / maxval
    return s


def main():
    [all_labels, mapping] = get_labels('classes.csv')
    hdrs = iterate_dir('../../SN_betaweights_n812')
    imgs = load(hdrs)
    all_images = format_data(imgs)


if __name__ == '__main__':
    main()
