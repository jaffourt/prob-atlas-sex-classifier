from Dataset import Dataset
from sklearn.neural_network import MLPClassifier
#
dataset = Dataset(dir='SN_betaweights_n812')
dataset.preprocess()
dataset.save_pkl()
#
# dataset = Dataset(data='dataset.pkl')
dataset.shuffle()
# train = int(len(dataset.data)*0.8)
#
# mlp = MLPClassifier(solver='adam', alpha=1e-5,
#                     hidden_layer_sizes=(5, 2), random_state=1, verbose=True, early_stopping=True)
#
# mlp.fit(dataset.data[0:train], dataset.labels[0:train])
# predictions = mlp.predict(dataset.data[train:])
# target = dataset.labels[train:]

