import numpy
import sklearn
from Best_model_finder.tuner import tuner


class model_finder:
    def __init__(self,  logger_object,file_object):

        self.file_obj = file_object
        self.log_writer = logger_object
        self.tuner = tuner(file_object, logger_object)

    def get_best_model(self,x_train, y_train, x_test, y_test):
        """
           Method Name: get_best_model
           Description: Find out the Model which has the best AUC score.
           Output: The best model name and the model object
           On Failure: Raise Exception

           Written By: richabudhraja8@gmail.com
           Version: 1.0
           Revisions: None

         """
        self.log_writer.log(self.file_obj, "entered get_best_model method in class model_finder!! ")
        try:

            #step 1 define all the names of model to be tested
            model_names = ["linearreg", "randomforest", "extratreereg", "catboost"]
            models=[]
            #step 2  a function definition for each model in a similar fashion to make calls
            # eg : get_params_for_xgboost(a,b) ,get_params_for_random_forest(a,b) etc where a and b are just passed to  the functions

            # model_functions_cal = []
            # for model_nme in model_names:
            #
            #     # model function get all function names within a string format
            #     # eg: "get_params_for_xgboost(a,b)" ,"get_params_for_random_forest(a,b)"..etc
            #
            #     # since all methods are in tuner class .function calls as object to  tuner dot functionname()
            #
            #     model_functions_cal.append('self.tuner.get_params_for_' + model_nme + '({trainX}, {trainY})'.format(trainX=x_train, trainY=y_train))

            #step 3 convert the functions in string to non string that would actually end up making the function call
            # eg : "get_params_for_xgboost(a,b)" to be replaced with  get_params_for_xgboost(a,b) , whatever the value function return will be now at its place
            #the functions return the model with best params so , models is a list of model with best parameters of each type of model in model_names list

            self.log_writer.log(self.file_obj, "making calls to functions for get_best_params for each model in tuner class !! ")
            print(len(x_train),len(y_train))
            models.append(self.tuner.get_params_for_linearReg( x_train, y_train))
            models.append(self.tuner.get_params_for_random_forest(x_train, y_train))
            models.append(self.tuner.get_params_for_extratreeReg(x_train, y_train))
            models.append(self.tuner.get_params_for_catboostReg( x_train, y_train))

            # models = [eval(x) for x in model_functions_cal]
            self.log_writer.log(self.file_obj, "Calls to functions for get_best_params for each model in tuner class ended successfully!! ")

            #step4 use each model to predict on x_test to get y_pred
            y_pred=[]
            for model in models:
                y_pred.append(model.predict(x_test))

            #step 5 for each model calculate the auc score /accuracy
              #step 5.1 if theres is just one label in y_test then use accuracy else use roc_auc_score
            scores=[]
            for y_pred_ in y_pred:
                    scores.append(sklearn.metrics.r2_score(y_test, y_pred_))

            #step 6 compare the scores for each model in scores list and get the index for best score and return the model at that index

            index = numpy.argmax(scores)
            best_model_name = model_names[index]
            best_model = models[index]

            self.log_writer.log(self.file_obj, "Exited the get_best_model method of the Clustering class. ")

            return best_model_name, best_model

        except Exception as e:
            self.log_writer.log(self.file_obj, "Error in  the get_best_model method of the Clustering class. %s" % e)
            raise e