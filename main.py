import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import BaggingRegressor

class Regression:
    def __init__(self, df_path):
        # Read input data from given path
        df = pd.read_csv(df_path, delimiter=',', index_col=False)

        # Split training and testing data 70:30
        X_train, X_test, y_train, y_test = train_test_split(
            df.iloc[:, 1:-1],
            df.iloc[:, -1],
            test_size=0.3,
            random_state=1,
        )

        # Scale numerical values with same method as paper being reproduced
        scaler = QuantileTransformer(n_quantiles=1000, random_state=1)
        scaler.fit(X_train)

        # Store X any y values in regression object instance
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def grid_search(self, regressor, parameters, print_results=False):
        # Measures used for each model
        scoring = ['neg_root_mean_squared_error', ]

        # Cross Validator used for Grid Search
        cv = ShuffleSplit(
            test_size=0.3, random_state=1
        )

        # Find the best model using grid-search with above cross-validation
        clf = GridSearchCV(
            regressor,
            param_grid=parameters,
            scoring=scoring,
            cv=cv,
            refit='neg_root_mean_squared_error'
        )
        clf.fit(X=self.X_train, y=self.y_train)

        # Print grid search results
        if print_results:
            print('Grid Search Results:')
            print(' (*) Best parameters set found on development set:', clf.best_params_)
            print(' (*) Best classifier score on development set:', clf.best_score_)
            print(' (*) Best classifier score on test set:', clf.score(self.X_test, self.y_test))

        # Return resulting estimator
        return(clf.best_estimator_)

    def evaluate_model(self, model, print_results=False):
        # Use given model to predict y values
        y_true, y_pred = self.y_test, model.predict(self.X_test)

        # Extract stats
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mdae = median_absolute_error(y_true, y_pred)

        # Print model metrics
        if print_results:
            print('Evaluating regressor:')
            print(' (*) R^2 Score:', r2)
            print(' (*) Mean Absolute Error:', mae)
            print(' (*) Mean Squared Error:', mse)
            print(' (*) Root Mean Squared Error:', rmse)
            print(' (*) Median Absolute Error:', mdae)

        # Return stats in dict
        stats = {
            'r2': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mdae': mdae
        }
        return(stats)

def run_grid_search(regressor=None, parameters=None, ds_number=6):
    # Use one of the 5 datasets from study to initialize regression model search
    if ds_number == 1:
        df_path=r'./dataset/researchDataset/DS07012.csv'   # DS1
    elif ds_number == 2:
        df_path=r'./dataset/researchDataset/DS07310.csv'   # DS2
    elif ds_number == 3:
        df_path=r'./dataset/researchDataset/DS07410.csv'   # DS3
    elif ds_number == 4:
        df_path=r'./dataset/researchDataset/DS07510.csv'   # DS4
    elif ds_number == 5:
        df_path=r'./dataset/researchDataset/DS07610.csv'   # DS5
    elif ds_number == 6:
        df_path=r'./dataset/newDataset/DS_K45.csv'         # DS6
    elif ds_number == 7:
        df_path=r'./dataset/newDataset/DS_LassoCV_k45.csv' # DS7
    elif ds_number == 8:
        df_path=r'./dataset/newDataset/DS_PCA_k45.csv'     # DS8
    elif ds_number == 9:
        df_path=r'./dataset/newDataset/DS_KBest_CV.csv'    # DS9
    elif ds_number == 10:
        df_path=r'./dataset/newDataset/DS_RFE_CV.csv'      # DS10
    
    # Run grid search on provided regression model and params
    reg_gs = Regression(df_path)
    best_model = reg_gs.grid_search(regressor, parameters)
    model_stats = reg_gs.evaluate_model(best_model)
    return(model_stats)

def regress(regressor=None, ds_number=6):
    # Use one of the 5 datasets from study to initialize regression model search
    if ds_number == 1:
        df_path=r'./dataset/researchDataset/DS07012.csv'   # DS1
    elif ds_number == 2:
        df_path=r'./dataset/researchDataset/DS07310.csv'   # DS2
    elif ds_number == 3:
        df_path=r'./dataset/researchDataset/DS07410.csv'   # DS3
    elif ds_number == 4:
        df_path=r'./dataset/researchDataset/DS07510.csv'   # DS4
    elif ds_number == 5:
        df_path=r'./dataset/researchDataset/DS07610.csv'   # DS5
    elif ds_number == 6:
        df_path=r'./dataset/newDataset/DS_K45.csv'         # DS6
    elif ds_number == 7:
        df_path=r'./dataset/newDataset/DS_LassoCV_k45.csv' # DS7
    elif ds_number == 8:
        df_path=r'./dataset/newDataset/DS_PCA_k45.csv'     # DS8
    elif ds_number == 9:
        df_path=r'./dataset/newDataset/DS_KBest_CV.csv'    # DS9
    elif ds_number == 10:
        df_path=r'./dataset/newDataset/DS_RFE_CV.csv'      # DS10
    
    # Run grid search on provided regression model and params
    reg = Regression(df_path)
    fit_reg = regressor.fit(reg.X_train, reg.y_train)
    model_stats = reg.evaluate_model(fit_reg)
    return(model_stats)

def compute_regressor_grid_search_stats(regressor, parameters, datasets=[i for i in range(1, 11)]):
    stats_dict = {}
    for ds_number in datasets:
        best_estimator_stats = run_grid_search(regressor, parameters, ds_number=ds_number)
        # For visualization and comparison of models we can update this to send stats somewhere other than print
        print(f'Best estimator stats on dataset {ds_number}:')
        print(best_estimator_stats)
        stats_dict[ds_number] = best_estimator_stats
    return(stats_dict)

def compute_regressor_stats(regressor, datasets=[i for i in range(1, 11)]):
    stats_dict = {}
    for ds_number in datasets:
        estimator_stats = regress(regressor, ds_number=ds_number)
        # For visualization and comparison of models we can update this to send stats somewhere other than print
        print(f'Best estimator stats on dataset {ds_number}:')
        print(estimator_stats)
        stats_dict[ds_number] = estimator_stats
    return(stats_dict)

def linear(datasets=[i for i in range(1, 11)]):
    regressor = linear_model.SGDRegressor(loss='huber', penalty='l2', learning_rate='invscaling', max_iter=50)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def dtr(datasets=[i for i in range(1, 11)]):
    regressor = DecisionTreeRegressor(random_state=1)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def rfr(datasets=[i for i in range(1, 11)]):
    regressor = RandomForestRegressor(n_estimators=150, max_depth=28, min_samples_leaf=2, criterion='squared_error')
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def hgbr(datasets=[i for i in range(1, 11)]):
    regressor = HistGradientBoostingRegressor(loss='squared_error', max_depth=18, min_samples_leaf=15, max_iter=500)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def svr(datasets=[i for i in range(1, 11)]):
    regressor = NuSVR(kernel='rbf', nu=0.5)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def mlpr(datasets=[i for i in range(1, 11)]):
    regressor = MLPRegressor(random_state=7, hidden_layer_sizes=(512, 256, 100), activation='tanh', learning_rate='constant')
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def vor(datasets=[i for i in range(1, 11)]):
    regressor1 = MLPRegressor(random_state=7, hidden_layer_sizes=(512, 256, 100), activation='tanh', learning_rate='constant')
    regressor2 = RandomForestRegressor(n_estimators=150, max_depth=28, min_samples_leaf=2, criterion='squared_error')
    regressor3 = HistGradientBoostingRegressor(loss='squared_error', max_depth=18, min_samples_leaf=15, max_iter=500)
    regressor = VotingRegressor([('MLP', regressor1), ('RFR', regressor2), ('HGBR', regressor3)])
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def adaboost(datasets=[i for i in range(1, 11)]):
    estimator = HistGradientBoostingRegressor(loss='squared_error', max_depth=18, min_samples_leaf=15, max_iter=500)
    regressor = AdaBoostRegressor(base_estimator=estimator)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def stacking(datasets=[i for i in range(1, 11)]):
    regressor1 = MLPRegressor(random_state=7, hidden_layer_sizes=(512, 256, 100), activation='tanh', learning_rate='constant')
    regressor2 = RandomForestRegressor(n_estimators=150, max_depth=28, min_samples_leaf=2, criterion='squared_error')
    regressor3 = HistGradientBoostingRegressor(loss='squared_error', max_depth=18, min_samples_leaf=15, max_iter=500)

    estimators = [
        ('lr', regressor1),
        ('svr', regressor2),
        ('rf', regressor3)
    ]

    regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10,
                                            random_state=42)
    )
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def bagging_hgbr(datasets=[i for i in range(1, 11)]):
    regressor = BaggingRegressor(estimator=HistGradientBoostingRegressor(loss='squared_error', max_depth=18, min_samples_leaf=15, max_iter=500), bootstrap=False)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def bagging_mlp(datasets=[i for i in range(1, 11)]):
    regressor = BaggingRegressor(estimator=MLPRegressor(random_state=7, hidden_layer_sizes=(512, 256, 100), activation='tanh', learning_rate='constant'), bootstrap=False)
    regressor_stats = compute_regressor_stats(regressor, datasets)
    return(regressor_stats)

def run_and_save_model_results(model_function, filepath):
    results = model_function()
    print(results)
    with open(filepath, 'wb') as fp:
        pickle.dump(results, fp)
        print('dictionary saved successfully to file')

def main():
    # Models and file paths
    models = {
        #linear: 'results/linear_results.pkl'
        #,dtr: 'results/dtr_results.pkl'
        #,rfr: 'results/rfr_results.pkl'
        #,hgbr: 'results/hgbr_results.pkl'
        #,svr: 'results/svr_results.pkl'
        #,mlpr: 'results/mlpr_results.pkl'
        #,vor: 'results/vor_results.pkl'
        #,adaboost: 'results/adaboost_results.pkl'
        #,stacking: 'results/stacking_results.pkl'
        bagging_hgbr: 'results/bagging_hgbr_results.pkl'
        ,bagging_mlp: 'results/bagging_mlp_results.pkl'
    }

    # Run model and pickle results dict to file
    for model, filepath in models.items():
        print(f"*********** {model.__name__} ***********")
        run_and_save_model_results(model, filepath)
    
    # Read pickled results
    # read_results(models)

def read_results(models):
    loaded_results = {}
    for model, filepath in models.items():
        with open(filepath, 'rb') as fp:
            results = pickle.load(fp)
            print('Read pickled dictionary')
            loaded_results[model.__name__] = results
    print(loaded_results)

if __name__ == '__main__':
    main()



