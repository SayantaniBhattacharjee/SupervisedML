import pandas
from sklearn import metrics
import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import warnings

class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing
    df_train = pandas.read_csv(
        '/assign3_students_train.txt',
        names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
               'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport',
               'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
               'Dalc', 'Walc', 'health', 'absences', 'G3'], sep='\t')
    df_test = pandas.read_csv(
        '/assign3_students_test.txt',
        names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
               'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport',
               'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
               'Walc', 'health', 'absences', 'G3'], sep='\t')
    df_train = pandas.get_dummies(data=df_train,
                                  columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                                           'guardian', 'nursery', 'higher', 'internet', 'romantic'])

    df_test = pandas.get_dummies(data=df_test,
                                 columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
                                          'guardian', 'nursery', 'higher', 'internet', 'romantic'])

    df_train = pandas.DataFrame(df_train)
    df_test = pandas.DataFrame(df_test)

    n_samples = 800
    rescaled_dim = 20

    df_train['split_tags'] = df_train['edusupport'].map(lambda row: row.split(" "))
    df_test['split_tags'] = df_test['edusupport'].map(lambda row: row.split(" "))
    lb = MultiLabelBinarizer()
    y = lb.fit_transform(df_train['split_tags'])
    y = y[:n_samples]
    X = df_train[df_train.columns.difference(['edusupport', 'split_tags'])]
    X_new = SelectKBest(chi2, k=47).fit_transform(X, y)
    X = MinMaxScaler().fit_transform(X_new, y)

    Y_test = lb.fit_transform(df_test['split_tags'])
    Y_test = Y_test[:n_samples]
    X_test = df_test[df_test.columns.difference(['edusupport', 'split_tags'])]

    def __init__(self):
        print("================Task 3================")
        return

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        clf_SVR = OneVsRestClassifier(SVR(kernel='rbf', C=30.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #t0=time.time()
            clf_SVR.fit(self.X, self.y)
            #print("training time:", round(time.time() - t0, 3), "s")
        # Evaluate learned model on testing data, and print the results.
        print("Accuracy\t" + str(metrics.accuracy_score(self.Y_test, clf_SVR.predict(self.X_test))) + "\tHamming loss\t" + str(metrics.hamming_loss(self.Y_test, clf_SVR.predict(self.X_test))))
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        clf_DT = OneVsRestClassifier(
            DecisionTreeClassifier(random_state=None, max_features='sqrt', min_samples_leaf=1000))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            #t0=time.time()
            clf_DT.fit(self.X, self.y)
            #print("training time:", round(time.time() - t0, 3), "s")

        # Evaluate learned model on testing data, and print the results.
        print("Accuracy\t" + str(metrics.accuracy_score(self.Y_test, clf_DT.predict(self.X_test))) + "\tHamming loss\t" + str(metrics.hamming_loss(self.Y_test, clf_DT.predict(self.X_test))))
        return
