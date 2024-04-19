from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def normalized(any_matrix, inv=False):
    normlize = MinMaxScaler()
    data = normlize.fit_transform(any_matrix)
    data_inv = normlize.inverse_transform(any_matrix)
    if inv:
        return data
    else:
        return data_inv

def standardized(any_matrix, inv=False):
    standardize = StandardScaler()
    data = standardize.fit_transform(any_matrix)
    data_inv = standardize.inverse_transform(any_matrix)
    if inv:
        return data
    else:
        return data_inv

def inv_normalized(any_matrix):
    inv_normalize = MinMaxScaler()
    data = inv_normalize.inverse_transform(any_matrix)
    return data


def train_data(X, y, i, importance=False):
    X = normalized(X)
    y = standardized(y)
    # X = dimen_reduction(X)
    # y = np.ravel(y).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    print('训练集及测试参数：')
    print('X_train.shape={}\n y_train.shape={}\n X_test.shape={}\n y_test.shape={}\n'.format(X_train.shape,
                                                                                             y_train.shape,
                                                                                             X_test.shape,
                                                                                             y_test.shape))
    # linreg = LinearRegression()
    # 训练
    # model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=10000)
    model_decision_tree_regression = tree.DecisionTreeRegressor(random_state=i)

    # model_svm = svm.SVR(kernel='poly', C=55, coef0=0.2, epsilon=0.45)
    # model_k_neighbor = neighbors.KNeighborsRegressor()
    # model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=2000)
    # mae, mae_train = try_different_method(model_decision_tree_regression, "model_random_forest_regressor", X_train, X_test, y_train, y_test, y, i)
    return try_different_method(model_decision_tree_regression, "model_random_forest_regressor", X, X_train, X_test, y_train, y_test, y, i, importance=importance)

def linear_train(X, y, i):
    normlize = MinMaxScaler()
    X = normlize.fit_transform(X)
    standardize = StandardScaler()
    y = standardize.fit_transform(y)
    # X = dimen_reduction(X)
    # y = np.ravel(y).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    a = linear_regression.coef_[0]
    b = linear_regression.intercept_[0]
    result = linear_regression.predict(X_test)

    return standardize.inverse_transform(result)
    # print(a)

def train_NN(X, y, i, importance=False):
    X = normalized(X)
    y = standardized(y)
    # X = dimen_reduction(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    print('训练集及测试参数：')
    print('X_train.shape={}\n y_train.shape={}\n X_test.shape={}\n y_test.shape={}\n'.format(X_train.shape,
                                                                                             y_train.shape,
                                                                                             X_test.shape,
                                                                                           y_test.shape))
    mp = MLPRegressor(activation='tanh', alpha=1e-5, batch_size='auto', beta_1=0.9,
                  beta_2=0.999, early_stopping=True, epsilon=1e-08,
                  hidden_layer_sizes=(20,10), learning_rate='constant',
                  learning_rate_init=0.001, max_iter=1000000, momentum=0.9,
                  nesterovs_momentum=True, power_t=0.5, shuffle=True,
                  solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                  warm_start=False)
    mp.fit(X_train, y_train)
    # MLPRegressor(activation='relu', alpha=1e-5, batch_size='auto', beta_1=0.9,
    #               beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #               hidden_layer_sizes=(5,2), learning_rate='constant',
    #               learning_rate_init=0.001, max_iter=1000, momentum=0.9,
    #               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #               warm_start=False
    #               )
    y_pred = mp.predict(X_test)
    y_pred_train = mp.predict(X_train)
    not_to = []
    to = []
    not_to_train = []
    to_train = []
    for k, j in enumerate(y_pred):
        if np.abs(j) >= 3 or np.abs(y_test[k]) >= 3:
            not_to.append(k)
            continue
        to.append(k)
    for k, j in enumerate(y_pred_train):
        if np.abs(j) >= 3 or np.abs(y_train[k]) >= 3:
            not_to_train.append(k)
            continue
        to_train.append(k)
    sum_mean = 0
    MAE = mean_absolute_error(y_test[to], y_pred[to])
    MAE_train = mean_absolute_error(y_train[to_train], y_pred_train[to_train])
    RMSE = mean_squared_error(y_test[to], y_pred[to])
    r2 = r2_score(y_test[to], y_pred[to])
    # calculate_RMSE
    print("Y-variance:", RMSE_y(y))
    print("RMSE by hand:", RMSE)
    print("R2:", r2)
    print("MAE:", MAE)
    plt.plot(range(len(y_pred[to])), y_pred[to], "ro-", label="predict")
    plt.plot(range(len(y_pred[to])), y_test[to], "go-", label="test")
    plt.legend(loc="upper right")
    plt.xlabel("the number of adsorption")
    plt.ylabel("value of adsorption")
    plt.show()
    # plt.scatter(y_test[to], y_pred[to], alpha=0.5, label='test')
    plt.scatter(y_train[to_train], y_pred_train[to_train], alpha=0.3, label='train')
    print([np.min(y_test[to]), np.min(y_test[to])], [np.max(y_test[to]), np.max(y_test[to])])
    plt.plot(np.arange(-3, 4), np.arange(-3, 4), c='r')
    # plt.legend()
    plt.savefig(f'scatter{i}.png')
    plt.show()

    return MAE, MAE_train


def dimen_reduction(X):
    pca = PCA(n_components='mle')
    pca.fit(X)
    print(pca.explained_variance_ratio_)

def RMSE_y(y):
    means = np.mean(y)
    sum_mean = 0
    for i in range(len(y)):
        sum_mean += (y[i] - means) ** 2
    sum_erro = np.sqrt(sum_mean / len(y))
    return sum_erro

def scatter_y(y_fred, y_test, y):
    plt.scatter(y_test, y_fred)
    plt.plot(y,y)
    plt.show()

def try_different_method(model, method, X, X_train, X_test, y_train, y_test,y,i, importance):
    model.fit(X_train, y_train)
    imp = model.tree_.compute_feature_importances()
    score = model.score(X_test, y_test)
    result = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    not_to = []
    to = []
    not_to_train = []
    to_train = []
    for k, j in enumerate(result):
        if np.abs(y_test[k]) >= 3:
            not_to.append(k)
            continue
        to.append(k)
    for k, j in enumerate(y_pred_train):
        if np.abs(j) >= 3 or np.abs(y_train[k]) >= 3:
            not_to_train.append(k)
            continue
        to_train.append(k)
    sum_mean = 0
    mae = 0
    # for i in range(len(result)):
    #     if i not in not_to:
    #         sum_mean += (result[i] - y_test[i]) ** 2
    #         mae += np.abs(result[i] - y_test[i])
    MAE = mean_absolute_error(y_test[to], result[to])
    MAE_train = mean_absolute_error(y_train[to_train], y_pred_train[to_train])
    RMSE = mean_squared_error(y_test[to], result[to])
    r2 = r2_score(y_test[to], result[to])
    # sum_erro = np.sqrt(sum_mean / len(result))
    # mae = mae / len(result)
    # calculate_RMSE
    print("Y-variance:", RMSE_y(y))
    print("RMSE by hand:", RMSE)
    print("mae:", MAE)
    plt.figure()
    plt.plot(np.arange(len(result[to])), y_test[to], "go-", label="True value")
    plt.plot(np.arange(len(result[to])), result[to], "ro-", label="Predict value")
    plt.title(f"method:{method}---score:{score}")
    plt.legend(loc="best")
    # plt.show()
    # plt.savefig('1.png')
    plt.show()
    plt.scatter(np.array(y_test[to]), result[to], alpha=0.1)
    plt.plot(np.arange(-3,4), np.arange(-3,4), c='r')
    # plt.savefig(f'scatter{i}.png')
    plt.show()
    if importance:
        return imp
    else:
        return float(MAE), float(MAE_train)