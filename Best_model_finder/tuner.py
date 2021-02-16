from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor


class tuner:
    """
               This class is used while model training . to get the models with best hyper parameters .
               all hyper parameter tuning is done here.

               Written By: richabudhraja8@gmail.com
               Version: 1.0
               Revisions: None
    """

    def __init__(self, file_object, logger_object):
        self.file_obj = file_object
        self.log_writer = logger_object
        pass

    def get_params_for_catboostReg(self, x_train, y_train):
        """
                                   Method Name: get_params_for_catboostReg
                                   Description: This method defines as catboost model for regression . It also performs grid search
                                                 to find the best hyper parameters for the regressor.
                                   Output: Returns a catboost model with best hyper parameters
                                   On Failure: Raise Exception

                                   Written By: richabudhraja8@gmail.com
                                   Version: 1.0
                                   Revisions: None

                                """
        estimator = CatBoostRegressor()
        params_grid = {"n_estimators": [100, 130,200],
                     'loss_function':['RMSE','MAE'],
                   'learning_rate':[0.01,0.5,1]
                    }



        self.log_writer.log(self.file_obj, "Entered get_params_for_catboostReg of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5, return_train_score=False)
            grid_cv.fit(x_train, y_train)

            # fetch the best estimator
            best_estimator = grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train )
            self.log_writer.log(self.file_obj,
                                "get best params for catboost successful"+ str(grid_cv.best_params_) +".Exited get_params_for_catboostReg of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_catboostReg of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for catboost unsuccessful .Exited get_params_for_catboostReg of tuner class!!')
            raise e


    def get_params_for_random_forest(self, x_train, y_train):
        """
                                           Method Name: get_params_for_random_forest
                                           Description: This method defines as random_forest model for regression . It also performs grid search
                                                         to find the best hyper parameters for the regressor.
                                           Output: Returns a random_forest model with best hyper parameters
                                           On Failure: Raise Exception

                                           Written By: richabudhraja8@gmail.com
                                           Version: 1.0
                                           Revisions: None

                                        """
        estimator = RandomForestRegressor()

        params_grid = {"n_estimators": [10, 50, 100, 130],
                       'criterion': ['mse'],
                       "max_depth": [None, 2, 3, 4],
                       "max_features": ['auto', 'log2']
                       }

        self.log_writer.log(self.file_obj, "Entered get_params_for_random_forest of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5, return_train_score=False)
            grid_cv.fit(x_train, y_train)

            # fetch the best estimator
            best_estimator = grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for random_forest successful ."+ str(grid_cv.best_params_) +".Exited get_params_for_random_forest of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_random_forest of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for random_forest unsuccessful .Exited get_params_for_random_forest of tuner class!!')
            raise e


    def get_params_for_linearReg(self, x_train, y_train):
        """
                           Method Name: get_params_for_linearReg
                           Description: This method defines as linearReg model for regression . It also performs grid search
                                         to find the best hyper parameters for the regressor.
                           Output: Returns a linearReg model with best hyper parameters
                           On Failure: Raise Exception

                           Written By: richabudhraja8@gmail.com
                           Version: 1.0
                           Revisions: None

                        """
        estimator = LinearRegression()
        params_grid = {
                'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]

             }
        self.log_writer.log(self.file_obj, "Entered get_params_for_linearReg of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5, return_train_score=False)
            grid_cv.fit(x_train, y_train)

            # fetch the best estimator
            best_estimator = grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for LinearReg successful ."+str(grid_cv.best_params_) +".Exited get_params_for_linearReg of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_linearReg of tuner class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'get best params for linearReg unsuccessful .Exited get_params_for_linearReg of tuner class!!')
            raise e


    def get_params_for_extratreeReg(self, x_train, y_train):
        """
                   Method Name: get_params_for_extratreeReg
                   Description: This method defines a Extra Tree Regressor model for regression . It also performs grid search
                                 to find the best hyper parameters for the regressor.
                   Output: Returns a Extra tree reg model with best hyper parameters
                   On Failure: Raise Exception

                   Written By: richabudhraja8@gmail.com
                   Version: 1.0
                   Revisions: None

                """
        estimator = ExtraTreesRegressor()
        params_grid={"n_estimators": [10, 50, 100, 130],
                     'criterion':['mse'],
                        "max_depth": [None,2,3,4],
                       "max_features": ['auto', 'log2']
                    }
        self.log_writer.log(self.file_obj, "Entered get_params_for_extratreeReg of tuner class!!")
        try:

            grid_cv = GridSearchCV(estimator, param_grid=params_grid, cv=5 ,return_train_score=False)
            grid_cv.fit(x_train, y_train)

            #fetch the best estimator
            best_estimator=grid_cv.best_estimator_
            best_estimator.fit(x_train, y_train)
            self.log_writer.log(self.file_obj,
                                "get best params for extratreeReg successful ."+ str(grid_cv.best_params_) +".Exited get_params_for_extratreeReg of tuner class!!")

            return best_estimator

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_params_for_extratreeReg of tuner class!! Exception message:' + str(e))
            self.log_writer.log(self.file_obj,
                                'get best params for extratreeReg unsuccessful .Exited get_params_for_extratreeReg of tuner class!!')
            raise e
