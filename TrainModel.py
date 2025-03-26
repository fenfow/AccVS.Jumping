import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
import h5py

# Author: Andrea Pereira
# this file corresponds to part 6 - training the model

# process data class i made
from FINAL import ProcessData

with h5py.File('dataset.h5', 'r') as hdf:
    train_dataset = pd.DataFrame(hdf['dataset/train/combined_train'][:])
    test_dataset = pd.DataFrame(hdf['dataset/train/combined_train'][:])

    train_dataset = ProcessData.processData(train_dataset)
    test_dataset = ProcessData.processData(test_dataset)

# train the model, these are the features it looks at
train_dataset.loc[train_dataset['zvariance'] <= 0.7, 'activity'] = 0
train_dataset.loc[train_dataset['zvariance'] > 0.7, 'activity'] = 1
test_dataset.loc[test_dataset['zvariance'] <= 0.7, 'activity'] = 0
test_dataset.loc[test_dataset['zvariance'] > 0.7, 'activity'] = 1
train_dataset.loc[train_dataset['zmedian'] >= 0.3, 'activity'] = 0
train_dataset.loc[train_dataset['zmedian'] < 0.3, 'activity'] = 1
test_dataset.loc[test_dataset['zmedian'] >= 0.3, 'activity'] = 0
test_dataset.loc[test_dataset['zmedian'] < 0.3, 'activity'] = 1

data = train_dataset.iloc[:, 1:-1]
labels = train_dataset.iloc[:, -1]

test_data = test_dataset.iloc[:, 1:-1]
test_labels = test_dataset.iloc[:, -1]

scaler = StandardScaler()

l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

clf.fit(data, labels)

y_pred = clf.predict(test_data)
y_clf_prob = clf.predict_proba(test_data)

# accuracy
acc = accuracy_score(test_labels, y_pred)

# confusion matrix
cm = confusion_matrix(test_labels, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# F1 score
f1 = f1_score(test_labels, y_pred)

# plot ROC
fpr, tpr, _ = roc_curve(test_labels, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# find AUC
auc = roc_auc_score(test_labels, y_clf_prob[:, 1])

import joblib

# Save the model to a pickle file
joblib.dump(clf, './classifier.sav')