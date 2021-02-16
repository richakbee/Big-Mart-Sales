from datetime import datetime

import numpy
import pandas

from Data_ingestion.data_loader import data_loader
from Data_preprocessing_prediction.preprocessing import preprocessing
from Model_functions.model_functions_fileops import model_functions
from Application_logging.logger import app_logger
from File_operation_prediction.file_operation import file_operation
import pandas as pd


class predict_from_model:
    def __init__(self):

        # open log writer and file object
        self.file_obj = open("Prediction_Logs/Prediction_main_log.txt", 'a+')
        self.log_writer = app_logger()
        # send log writer and file object for logging to other classes
        self.data_loader_obj = data_loader(self.log_writer, self.file_obj)
        self.preprocessor = preprocessing(self.log_writer, self.file_obj)
        self.model_functions_obj = model_functions(self.log_writer, self.file_obj)
        self.file_op_obj = file_operation()
        self.prediction_input_file = 'Prediction_FileFromDB/InputFile.csv'
        self.prediction_output_file = 'Prediction_Output_File/Predictions.csv'
        self.categorical_feature_list_loc='preprocessing_data/categorical_features.csv'
        self.columns_to_drop_list_loc = 'preprocessing_data/columns_to_remove.csv'

    def get_prediction_from_model(self):
        try:
            # step 1 delete prediction file from last run
            self.file_op_obj.delete_existing_file_create_new(self.prediction_output_file)

            self.log_writer.log(self.file_obj, "Start of Prediction!!")

            # step 2 get the data from prediction path
            data = self.data_loader_obj.get_data(self.prediction_input_file)

            # step 2 set  categorical features as true if there are categorical features in data (optional)
            are_categorical_features = False
            categorical_features = self.file_op_obj.read_list(self.categorical_feature_list_loc)

            if len(categorical_features)> 0:
                are_categorical_features = True


            # step2 preprocessing

            #'policy_no_index' will be used as index

            # step2.1 remove columns (no columns to remove)
            cols_to_remove = self.file_op_obj.read_list(self.columns_to_drop_list_loc)

            data = self.preprocessor.remove_columns(data, cols_to_remove)

            # replacing '?' in data with NAN values
            data.replace('?', numpy.NAN, inplace=True)

            # step2.2 handle /impute null values if present
            is_null_present, columns_with_null = self.preprocessor.is_null_present(data)

            if is_null_present:
                # check if null is in categorical variables then call categorical imputer .
                # print(data[columns_with_null].dtypes.value_counts()['category'])
                # if (data[columns_with_null].dtypes.value_counts()['category'] > 0):
                data = self.preprocessor.impute_Categorical_values(data, columns_with_null)

                #  nulls are in non categorical columns then
                # if categorical columns exist in data then encode them
                if are_categorical_features:
                    data = self.preprocessor.encode_categorical_columns_from_mapping_file(data, categorical_features)

                # then drop other na
                data = data.dropna()
                data.reset_index(inplace=True, drop=True)

            else:
                if are_categorical_features:
                    data = self.preprocessor.encode_categorical_columns_from_mapping_file(data, categorical_features)

            # One hot encoding categorical features in data X
            X_cat = self.preprocessor.one_hot_encode_cagtegorical_col(data, categorical_features)
            # Scaling Numerical Columns in data X
            X_num_scaled = self.preprocessor.scale_numerical_columns_from_training_Scaler(data, categorical_features)
            X_num_scaled.replace([numpy.inf, -numpy.inf], numpy.nan, inplace=True)

            # PCA on numerical columns
            X_num_pca = self.preprocessor.pcaTransformation(X_num_scaled)
            # concat numerical & categorical data together
            data = pandas.concat([X_num_pca, X_cat], axis=1)


            model_name = self.model_functions_obj.find_correct_model()
            model = self.model_functions_obj.load_model(model_name)

            # step 5.5 use the model to predict the label
            results = list(model.predict(data))  # predict is method of sklearn
            results = pandas.DataFrame(results, columns=['Prediction'])
            results["Prediction"] = results["Prediction"].map({ 0 : "Yes", 1: "No"})



                # step 5.7 save the final result into a csv
                # appends to the csv file
            results.to_csv(self.prediction_output_file, index=True, header=True, mode='a+')

            self.log_writer.log(self.file_obj, "End of Prediction!!")

            return self.prediction_output_file, results.head().to_json(orient="records")

        except Exception as e:
            self.log_writer.log(self.file_obj, "Error Occurred during Prediction!!Error :: %s" % e)
            raise e


