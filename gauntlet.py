import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from yellowbrick.model_selection.validation_curve import ValidationCurve
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.model_selection import LearningCurve
from yellowbrick.regressor import PredictionError
from yellowbrick.target import FeatureCorrelation
from matplotlib import pyplot as plt



# Models to support:
# MLP Regressor
# kNN Regressor
# Gradient Boosting Trees
# Linear Regression
# Decision Tree Regressor

def mae(actual_values, predicted_values):
    return np.abs(actual_values - predicted_values).sum() / actual_values.shape[0]

def _cross_validate(model_cls, data, labels, cv=5, **kwargs):
    errs = []
    for i in range(5):
        X_train, Y_train, X_test, Y_test = train_test_split(data, labels)
        mask = np.random.rand(data.shape[0]) < 0.8
        train_data = data[mask]
        train_labels = labels[mask]
        test_data = data[np.logical_not(mask)]
        test_labels = labels[np.logical_not(mask)]

        model = model_cls(**kwargs).fit(train_data, train_labels)
        pred_labels = model.predict(test_data)
        err = mae(test_labels, pred_labels)
        errs.append(err)
    
    errs = np.array(err)
    errs_std = errs.std()
    errs_avg = errs.mean()
    return errs_avg, errs_std

_models = {
    'mlp': MLPRegressor,
    'knn': KNeighborsRegressor,
    'tree': DecisionTreeRegressor,
    'gbt': GradientBoostingRegressor
}

def test_models(train_data, train_labels, models='all', quiet=False, cv=5):
    if models != 'all':
        models = {name: models[name] for name in models}
    else:
        models = _models

    train_results = {}
    for name, model_fn in models.items():

        err_avg, err_std = _cross_validate(model_fn, train_data, train_labels)
        if not quiet:
            print('Model: %s'%model_fn.__name__)
            print('\tAverage MAE: %f, MAE Standard Dev: %f\n'%(err_avg, err_std))
        train_results[name] = {
            'error_average': err_avg,
            'error_stdev': err_std
    }

    return train_results

def compare_datasets(data1, data1_labels, data2, data2_labels):
    print('Dataset 1 Row Count: %i, Feature Count: %i'%(data1.shape[0], data1.shape[1]))
    print('Dataset 2 Row Count: %i, Feature Count: %i'%(data2.shape[0], data2.shape[1]))

    data1_results = test_models(data1, data1_labels, quiet=True)
    data2_results = test_models(data2, data2_labels, quiet=True)

    for data1_key, data1_result in data1_results.items():
        data2_result = data2_results[data1_key]

        d1_error = data1_result['error_average']
        d2_error = data2_result['error_average']
        pct_change = (d1_error - d2_error) / d1_error
        print('Model: %s, Data 1 error: %.3f, Data 2 error: %.2f, change: %.3f%%' \
            %(data1_key, d1_error, d2_error, pct_change * 100))

def viz_model(model, data, labels):
    model = _models[model]

    viz_tools = [LearningCurve, ResidualsPlot, PredictionError]
    # fig, axes = plt.subplots(ncols=len(viz_tools), figsize=(20,10))

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels)

    for i, viz_tool in enumerate(viz_tools):
        # viz = viz_tool(model(), ax=axes[i])
        viz = viz_tool(model())
        viz.fit(X_train, Y_train)
        try:
            viz.score(X_test, Y_test)
        except NotFittedError:
            # some visualizers don't need to be scored
            pass

        viz.poof()

    # plt.show()

if __name__ == '__main__':
    viz_model(MLPRegressor, )
