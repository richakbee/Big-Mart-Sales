import numpy
import pandas
import sklearn
from Data_ingestion.data_loader import data_loader
from Data_preprocessing.preprocessing import preprocessing
from Best_model_finder.model_finder import model_finder
from Model_functions.model_functions_fileops import model_functions
from File_operation.file_operation import file_operation
from Application_logging import logger



class train_model:
    def __init__(self):

        # open log writer and file object
        self.file_obj = open("Training_logs/ModelTrainingLog.txt", 'a+')
        self.log_writer = logger.app_logger()
        # send log writer and file object for logging to other classes
        self.data_loader_obj = data_loader(self.log_writer, self.file_obj)
        self.preprocessor = preprocessing(self.log_writer, self.file_obj)
        self.model_finder_obj = model_finder(self.log_writer, self.file_obj)
        self.model_functions_obj = model_functions(self.log_writer, self.file_obj)
        self.file_op_obj = file_operation()
        self.training_file_name = 'Training_FileFromDB/InputFile.csv'

    def training_model(self):
        try:
            self.log_writer.log(self.file_obj, "Training started!!")

            self.file_op_obj.createDirectoryForPreprocessing()

            # step1 get the data .
            data = self.data_loader_obj.get_data(self.training_file_name)

            # step 2

            categorical_features = ["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy",
                                     "rev_stop"]
            categorical_label = ["went_on_backorder"]
            categorical_columns = categorical_features + categorical_label

            #saving categorical features list at  'preprocessing_data/categorical_features . csv' to be used at prediction
            location_string_flist = 'preprocessing_data/categorical_features.csv'
            self.file_op_obj.save_data_to_file(categorical_features, location_string_flist )
            #step 3 encode categorical columns
            yes_no_col =["potential_issue", "deck_risk", "oe_constraint", "ppap_risk", "stop_auto_buy",
                               "rev_stop","went_on_backorder"]
            data=self.preprocessor.encode_categorical_columns(data, categorical_columns,yes_no_col )



            # step2.1 remove columns (no columns to remove)
            cols_to_remove = ['sku']

            # saving columns to remove at  'preprocessing_data/columns_to_remove.csv' to be used at prediction to drop same columns
            location_col_drop_list = 'preprocessing_data/columns_to_remove.csv'
            self.file_op_obj.save_data_to_file(cols_to_remove, location_col_drop_list)
            data=self.preprocessor.remove_columns(data, cols_to_remove)

            # replacing '?' in data with NAN values
            data.replace('?', numpy.NAN, inplace=True)
            # since null in lead_time was replaced with 0's while inserting into database
            data=data[data['lead_time']>0.0]
            data.reset_index(inplace=True,drop=True)

            # step2.2 handle /impute null values if present
            is_null_present, columns_with_null = self.preprocessor.is_null_present(data)

            if is_null_present:
                # check if null is in categorical variables then call categorical imputer .
                # then drop the null for others
                data= self.preprocessor.impute_Categorical_values(data, columns_with_null)
                data =data.dropna()
                data.reset_index(inplace=True, drop=True)



             # refer EDA . doing log transformation on numerical columns with outliers.
            # col_with_log=['national_inv','lead_time', 'in_transit_qty',
            #                 'forecast_3_month', 'forecast_6_month', 'forecast_9_month',
            #                 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month',
            #                  'min_bank','pieces_past_due' ,'local_bo_qty']
            #
            # location_col_with_log = 'preprocessing_data/columns_to_log_transform.csv'
            # self.file_op_obj.save_data_to_file(col_with_log, location_col_with_log)
            # data=self.preprocessor.transform_log(data, col_with_log)



            #  separate features & label

            X, Y = self.preprocessor.separate_features_and_label(data, label_column_name="went_on_backorder")

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output for both the labels (fraud & not fraud)
            # prepare the list of such columns to drop
            col_with_zero_std_deviation = self.preprocessor.get_col_with_zero_std_deviation(X)

            if len(col_with_zero_std_deviation) > 0:
                # appending column names to file
                location_col_drop_list = 'preprocessing_data/columns_to_remove.csv'
                self.file_op_obj.append_data_to_file(col_with_zero_std_deviation, location_col_drop_list)

                X = self.preprocessor.remove_columns(X, col_with_zero_std_deviation)

            # One hot encoding categorical features in data X
            X_cat = self.preprocessor.one_hot_encode_cagtegorical_col(X, categorical_features)
            # Scaling Numerical Columns in data X
            X_num_scaled = self.preprocessor.scale_numerical_columns(X, categorical_features)

            X_num_scaled.replace([numpy.inf,-numpy.inf],numpy.nan , inplace=True)


            #PCA on numerical columns
            pca_num_components = 8 #refer EDA, why 8
            X_num_pca = self.preprocessor.pcaTransformation(X_num_scaled, pca_num_components)

            #concat numerical & categorical data together
            X = pandas.concat([X_num_pca, X_cat], axis=1)


            # split data into test & train
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=1 / 3,
                                                                                                random_state=0)


            #  find best model
            best_model_name, best_model = self.model_finder_obj.get_best_model(x_train, y_train, x_test, y_test)

            # save the best model
            save_model_status = self.model_functions_obj.save_model(best_model, best_model_name )

            # logging the successful Training
            self.log_writer.log(self.file_obj, 'Training Successful!!')
            self.file_obj.close()


        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_obj, 'Training Unsuccessful!!')
            self.file_obj.close()
            raise e
