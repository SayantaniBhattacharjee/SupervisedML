import pandas
import numpy as np
import time
from sklearn import metrics, feature_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings

class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    list_train = []
    list_test = []
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

    df_train = pandas.DataFrame(df_train)
    df_test = pandas.DataFrame(df_test)

    df_train_multicat = pandas.get_dummies(data=df_train,
                                           columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                                                    'reason', 'guardian', 'edusupport', 'nursery', 'higher', 'internet',
                                                    'romantic'])

    df_test_multicat = pandas.get_dummies(data=df_test,
                                          columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                                                   'reason', 'guardian', 'edusupport', 'nursery', 'higher', 'internet',
                                                   'romantic'])

    def mjobOrdinalColumn(df, list):
        Mjob_list = list
        for i in df['Mjob']:
            if i == 'teacher':
                Mjob_list.append(1)
            elif i == 'health':
                Mjob_list.append(2)
            elif i == 'services':
                Mjob_list.append(3)
            elif i == 'at_home':
                Mjob_list.append(4)
            else:
                Mjob_list.append(5)
        return Mjob_list

    mjobOrdinalColumn(df_train, list_train)
    df_train = df_train.assign(Mjob=list_train)

    mjobOrdinalColumn(df_test, list_test)
    df_test = df_test.assign(Mjob=list_test)

    df_train = pandas.get_dummies(data=df_train,
                                  columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Fjob', 'reason',
                                           'guardian', 'edusupport', 'nursery', 'higher', 'internet', 'romantic'])

    df_test = pandas.get_dummies(data=df_test,
                                 columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Fjob', 'reason',
                                          'guardian', 'edusupport', 'nursery', 'higher', 'internet', 'romantic'])

    def __init__(self):
        warnings.filterwarnings("ignore")
        print("================Task 2================")
        return

    def create_df(self,rel_features_list, feature_name):
        rel_features = []
        X = []
        Y = []
        rel_features.append(rel_features_list)
        feature_df = pandas.DataFrame(rel_features)
        for col in feature_df.columns:
            if col == feature_name:
                Y.append(col)
            else:
                X.append(col)
        return X, Y

    def knn_model_result(self, X_train, Y_train, X_test, Y_test, samples):
        knn_model = KNeighborsClassifier(n_neighbors=samples)
        #t0 = time.time()
        knn_cv = knn_model.fit(X_train, Y_train)
        #print("training time:", round(time.time() - t0, 3), "s")
        # On test dataset
        knn_y_pred = knn_model.predict(X_test)
        return (metrics.accuracy_score(Y_test, knn_y_pred), metrics.f1_score(Y_test, knn_y_pred, average='macro'),
                metrics.recall_score(Y_test, knn_y_pred, average='macro'),
                metrics.precision_score(Y_test, knn_y_pred, average='macro'))

    def logreg_model_result(self,X_train, Y_train, X_test, Y_test, iter):
        logreg_model = LogisticRegression(max_iter=iter)
        rfe = feature_selection.RFE(logreg_model, 20)
        #t0=time.time()
        rfe = rfe.fit(X_train, Y_train)
        #print("training time:", round(time.time() - t0, 3), "s")
        logreg_y_pred = rfe.predict(X_test)
        return (metrics.accuracy_score(Y_test, logreg_y_pred), metrics.f1_score(Y_test, logreg_y_pred, average='macro'),
                metrics.recall_score(Y_test, logreg_y_pred, average='macro'),
                metrics.precision_score(Y_test, logreg_y_pred, average='macro'))

    def X_list(self,list):
        Xtrain_list = []
        Xtest_list = []
        for i in list:
            if i in self.df_test_multicat.columns:
                Xtrain_list.append(self.df_train_multicat[[i]])
                Xtest_list.append(self.df_test_multicat[[i]])
        return Xtrain_list, Xtest_list

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        Y_train = self.df_train['Mjob']
        X_train = self.df_train[['Medu', 'Fedu', 'Fjob_other']]
        X_test = self.df_test[['Medu', 'Fedu', 'Fjob_other']]
        Y_test = self.df_test[['Mjob']]
        MAccuracy, MF1, MRecall, MPrecision = self.knn_model_result(X_train, Y_train, X_test, Y_test, 10)
        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(MAccuracy, MPrecision, MRecall, MF1)
        cor = self.df_train_multicat.corr()
        categories = ["teacher", "health", "service", "at_home", "other"]

        for category in categories:
            category1 = "Mjob_"+category
            if category1 in self.df_train_multicat.columns:
                X_train_cat = pandas.DataFrame()
                X_test_cat = pandas.DataFrame()
                cor_target = abs(cor[category1])
                relevant_features_teacher = cor_target[cor_target > 0.2]
                X, Y = self.create_df(relevant_features_teacher, category1)
                for i in X:
                    if i in self.df_test_multicat.columns:
                        X_train_cat[[i]] = pandas.DataFrame(self.df_train_multicat[[i]])
                        X_test_cat[[i]] = pandas.DataFrame(self.df_test_multicat[[i]])
                Y_train_cat = self.df_train_multicat[[category1]]
                Y_test_cat = self.df_test_multicat[[category1]]

                Accuracy, F1, Recall, Precision = self.knn_model_result(X_train_cat, Y_train_cat, X_test_cat, Y_test_cat, 10)
            self.print_category_results(category, Precision, Recall, F1)
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        Y_train = self.df_train['Mjob']
        X_train = self.df_train[['Medu', 'Fedu', 'Fjob_other']]
        X_test = self.df_test[['Medu', 'Fedu', 'Fjob_other']]
        Y_test = self.df_test[['Mjob']]
        MAccuracy, MF1, MRecall, MPrecision = self.logreg_model_result(X_train, Y_train, X_test, Y_test, 4000)

        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(MAccuracy, MPrecision, MRecall, MF1)
        cor = self.df_train_multicat.corr()
        categories = ["teacher", "health", "service", "at_home", "other"]

        for category in categories:
            category1 = "Mjob_" + category
            if category1 in self.df_train_multicat.columns:
                X_train_cat = pandas.DataFrame()
                X_test_cat = pandas.DataFrame()
                cor_target = abs(cor[category1])
                relevant_features_teacher = cor_target[cor_target > 0.2]
                X, Y = self.create_df(relevant_features_teacher, category1)
                for i in X:
                    if i in self.df_test_multicat.columns:
                        X_train_cat[[i]] = pandas.DataFrame(self.df_train_multicat[[i]])
                        X_test_cat[[i]] = pandas.DataFrame(self.df_test_multicat[[i]])
                Y_train_cat = self.df_train_multicat[[category1]]
                Y_test_cat = self.df_test_multicat[[category1]]
                Accuracy, F1, Recall, Precision = self.logreg_model_result(X_train_cat, Y_train_cat, X_test_cat, Y_test_cat, 4000)
            self.print_category_results(category, Precision, Recall,F1)
        return
