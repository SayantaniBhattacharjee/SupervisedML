import pandas
import numpy
import matplotlib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, feature_selection
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier

class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    school_rep = []
    add_urban = []
    P_status = []
    fam_size = []
    m_job = []
    f_job = []
    guard = []
    edu_support = []
    nursery = []
    high_edu = []
    internet = []
    romantic = []
    test_high = []

    df_train = pandas.read_csv('/assign3_students_train.txt',
                               names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                      'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport',
                                      'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout',
                                      'Dalc', 'Walc', 'health', 'absences', 'G3'], sep='\t')
    df_test = pandas.read_csv('/assign3_students_test.txt',
                              names=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
                                     'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'edusupport',
                                     'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
                                     'Walc', 'health', 'absences', 'G3'], sep='\t')

    df_train = pandas.DataFrame(df_train)
    df_test = pandas.DataFrame(df_test)


    for i in df_train['reason']:
        if i == 'course' or i == 'reputation':
            school_rep.append(1)
        else:
            school_rep.append(0)
    df_train = df_train.assign(school_rep=school_rep)

    for i in df_train['address']:
        if i == 'U':
            add_urban.append(1)
        else:
            add_urban.append(0)
    df_train = df_train.assign(address=add_urban)

    for i in df_train['Pstatus']:
        if i == 'T':
            P_status.append(1)
        else:
            P_status.append(0)
    df_train = df_train.assign(Pstatus=P_status)

    for i in df_train['famsize']:
        if i == 'LE3':
            fam_size.append(1)
        else:
            fam_size.append(0)
    df_train = df_train.assign(famsize=fam_size)

    for i in df_train['Mjob']:
        if i == 'at_home':
            m_job.append(1)
        else:
            m_job.append(0)
    df_train = df_train.assign(Mjob=m_job)

    for i in df_train['Fjob']:
        if (i != 'at_home' or i != 'other'):
            f_job.append(1)
        else:
            f_job.append(0)
    df_train = df_train.assign(Fjob=f_job)

    for i in df_train['guardian']:
        if i != 'other':
            guard.append(1)
        else:
            guard.append(0)
    df_train = df_train.assign(guardian=guard)

    for i in df_train['edusupport']:
        if i != 'paid':
            edu_support.append(1)
        else:
            edu_support.append(0)
    df_train = df_train.assign(edusupport=edu_support)

    for i in df_train['nursery']:
        if i == 'yes':
            nursery.append(1)
        else:
            nursery.append(0)
    df_train = df_train.assign(nursery=nursery)

    for i in df_train['higher']:
        if i == 'yes':
            high_edu.append(1)
        else:
            high_edu.append(0)
    df_train = df_train.assign(higher=high_edu)

    for i in df_train['internet']:
        if i == 'yes':
            internet.append(1)
        else:
            internet.append(0)
    df_train = df_train.assign(internet=internet)

    for i in df_train['romantic']:
        if i == 'no':
            romantic.append(1)
        else:
            romantic.append(0)
    df_train = df_train.assign(romantic=romantic)

    for i in df_test['higher']:
        if i == 'yes':
            test_high.append(1)
        else:
            test_high.append(0)
    df_test = df_test.assign(higher=test_high)

    # Correlation with output variable
    cor = df_train.corr()
    cor_target = abs(cor["G3"])
    # Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0.26]

    Y_train = df_train['G3']
    X_train = df_train[['Medu', 'failures', 'higher']]
    X_test = df_test[['Medu', 'failures', 'higher']]
    Y_test = df_test[['G3']]


    def __init__(self):
        print("================Task 1================")
        return

    def model_1_run(self):

        print("Model 1: Linear Regression Model with Ridge regularization, alpha = 0.001 and solver='sag'")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        lin_reg = Ridge(alpha=0.001, solver='sag')
        #t0 = time.time()
        lin_reg.fit(self.X_train, self.Y_train)
        #print("training time:", round(time.time() - t0, 3), "s")
        # print('Coefficients: \n', lin_reg.coef_)
        linreg_y_pred = lin_reg.predict(self.X_test)
        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error\t" + str(metrics.mean_squared_error(self.Y_test, linreg_y_pred)))
        return

    def model_2_run(self):



        print("--------------------\nModel 2: KNN model with n=9 and BallTree algorithm")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        knn_model = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree')
        #t0 = time.time()
        knn_model.fit(self.X_train, self.Y_train)
        #print("training time:", round(time.time() - t0, 3), "s")
        knn_y_pred = knn_model.predict(self.X_test)
        # Evaluate learned model on testing data, and print the results.
        print("Mean squared error\t" + str(metrics.mean_squared_error(self.Y_test, knn_y_pred)))
        return
