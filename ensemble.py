import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

phonemes = ["t", "th", "tt"]

# read features from csv
features = pd.read_csv('training_features.csv')
test_features = pd.read_csv('test_features.csv')

# create feature dataframe
feature_names = ["mean", "stdev", "skew", "kurtosis", "zcr_mean", "zcr_stdev",
                 "rmse_mean", "rmse_stdev", "tempo"] + \
                ['mfccs_' + str(i+1) + '_mean' for i in range(20)] + \
                ['mfccs_' + str(i+1) + '_stdev' for i in range(20)] + \
                ['chroma_' + str(i+1) + '_mean' for i in range(12)] + \
                ['chroma_' + str(i+1) + '_stdev' for i in range(12)] + \
                ["centroid_mean", "centroid_stdev"] + \
                ['contrast_' + str(i+1) + '_mean' for i in range(7)] + \
                ['contrast_' + str(i+1) + '_std' for i in range(7)] + \
                ["rolloff_mean", "rolloff_stdev", "phoneme"]

param_names = feature_names[1:-1]
label_names = feature_names[-1]

# extract parameters and labels
params = features.loc[:, param_names].values
labels = features.loc[:, label_names].values
test_params = test_features.loc[:, param_names].values

# normalize data
params_norm = StandardScaler().fit_transform(params)
test_params_norm = StandardScaler().fit_transform(test_params)

# x_train, x_test, y_train, y_test = train_test_split(params_norm, labels, test_size = 0.25)
x_train = params_norm
x_test = test_params_norm
y_train = labels

# Run KNN
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train, y_train)
knn_predict = knn.predict_proba(x_test)

# Run Logistic Regression
mlr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
mlr.fit(x_train, y_train)
mlr_predict= mlr.predict_proba(x_test)

# Run MLP
mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(100), learning_rate_init=0.01, max_iter=10000, random_state=1)
mlp.fit(x_train, y_train)
mlp_predict = mlp.predict_proba(x_test)

# Run SVM
svm = SVC(gamma = 0.01, probability=True) # gamma = 0.01
svm.fit(x_train, y_train)
svm_predict = svm.predict_proba(x_test)

# Run Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy', max_features=15)
rf.fit(x_train, y_train)
rf_predict = rf.predict_proba(x_test)

# Aggregate Data
combined_probs = np.hstack((rf_predict, svm_predict, mlp_predict, mlr_predict, knn_predict))
pred = mlr.classes_[np.argmax(combined_probs, axis=1)%3]

# print(accuracy_score(pred, y_test))

# create final dataframe
pred_frame = pd.DataFrame(columns=['filename', 'label'])
filenames = test_features.loc[:, 'filename'].values
pred_frame = pd.DataFrame({'filename': filenames, 'label': pred})

# save features to csv
pred_frame.to_csv('ensemble-max.csv', index=False)