import csv
import pickle
from datetime import datetime
import os

import numpy
import pandas
import sklearn
from sklearn.impute import _knn




class preprocessing:
    """
            This class shall  be used to clean and transform the data before training.

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

            """

    def __init__(self, logger_object, file_object):

        self.file_obj = file_object
        self.log_writer = logger_object

    def remove_columns(self, data, column_list):
        """
                Method Name: remove_columns
                Description: TThis method removed the  the columns in the column list from the data.
                Output: Returns A pandas data frame.
                On Failure: Raise Exception

                Written By: richabudhraja8@gmail.com
                Version: 1.0
                Revisions: None

                """
        self.log_writer.log(self.file_obj, "Entered remove_columns of preprocessing class!!")
        try:
            new_data = data.drop(columns=column_list, axis=1)
            self.log_writer.log(self.file_obj,
                                "Column removal Successful.Exited the remove_columns method of the Preprocessor class!!")
            return new_data
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in remove_columns method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'removing columns from data  failed.Error in  remove_columns method of the Preprocessor class!!')
            raise e

    def separate_features_and_label(self, data, label_column_name):
        """
            Method Name: separate_features_and_label
            Description: This method separates the features and a Label Coulmns.
            Output: Returns two separate Dataframes, one containing features and the other containing Labels .
            On Failure: Raise Exception

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

        """
        self.log_writer.log(self.file_obj, "Entered separate_features_and_label of class preprocessing!!")
        try:
            Y = data[label_column_name]
            X = data.drop(columns=[label_column_name], axis=1)
            self.log_writer.log(self.file_obj,
                                "Label separation successful .Exited separate_features_and_label of class preprocessing!!")
            return X, Y
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                "Label separation unsuccessful !!")
            self.log_writer.log(self.file_obj, "Error in  separate_features_and_label of class preprocessing %s" % e)
            raise e

    def is_null_present(self, X):
        """
                    Method Name: is_null_present
                    Description: This method takes input as dataframe . and tells if there are nulls in any column
                    Output: Returns boolean yes or no .if yes then a csv will be stored will count of null for each column
                            at location "preprocessing_data/null_values.csv". Also returns a list of column names with null values
                    On Failure: Raise Exception

                    Written By: richabudhraja8@gmail.com
                    Version: 1.0
                    Revisions: None

                """
        null_df_loc = "preprocessing_data/"
        self.log_writer.log(self.file_obj,
                            "Entered is_null_present in class preprocessing. Checking for null values in training data")
        bool_value = False
        columns_with_null = []
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            count_null = X.isnull().sum()
            for count_ in count_null:
                if count_ > 0:
                    bool_value = True
                    break
            if bool_value:
                null_df = pandas.DataFrame(count_null).reset_index().rename(
                    columns={'index': 'col_name', 0: 'no_of_nulls'})
                file_name = 'null_values_' + str(date) + "_" + str(time) + '.csv'
                dest = os.path.join(null_df_loc, file_name)
                if not os.path.isdir(dest):
                    null_df.to_csv(dest, index=False, header=True)

                # list of columns that has null values
                columns_with_null = list(null_df[null_df['no_of_nulls'] > 0]['col_name'].values)

                self.log_writer.log(self.file_obj,
                                    "Finding missing values is a success.Data written to the null values file at {}. Exited the is_null_present method of the Preprocessor class".format(
                                        null_df_loc))
            return bool_value, columns_with_null
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise e

    def impute_missing_values_KNN(self, X):
        """
                    Method Name: impute_missing_values
                    Description: This method replaces all the missing values in the Dataframe using Categorical imputer then
                                encoding categorical variables and then KNN Imputer for numerical columns
                    Output: Returns A Dataframe which has all the missing values imputed.and
                    On Failure: Raise Exception

                    Written By: richabudhraja8@gmail.com
                    Version: 1.0
                    Revisions: None

                """
        self.log_writer.log(self.file_obj, "Entered the impute_missing_values_KNN method of the Preprocessor class!!")
        data = X
        try:

            imputer = _knn.KNNImputer(n_neighbors=3, weights='uniform', missing_values=numpy.nan)
            # the final result after imputing is an array
            imputed_data = imputer.fit_transform(data)
            # convert the array back to data frame
            new_data = pandas.DataFrame(imputed_data, columns=data.columns)
            self.log_writer.log(self.file_obj,
                                "Imputing missing values Successful. Exited the impute_missing_values_KNN method of the Preprocessor class!!")
            return new_data
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in impute_missing_values_KNN method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Imputing missing values failed. impute_missing_values_KNN method of the Preprocessor class!!')
            raise e

    def get_col_with_zero_std_deviation(self, X):
        """
        Method Name: get_col_with_zero_std_deviation
        Description: TThis method finds out the columns which have a standard deviation of zero.
        Output: Returns A list of all coulmns which have a standard deviation of zero.
        On Failure: Raise Exception

        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None

        """
        self.log_writer.log(self.file_obj, "Entered get_col_with_zero_std_deviation of preprocessing class!!")
        try:
            # get the standard deviation of all columns as data frame , where index is column name .
            std_columns = pandas.DataFrame(X.describe().loc['std'])
            col_list = std_columns[std_columns['std'] == 0].index.to_list()
            self.log_writer.log(self.file_obj,
                                "Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class!!")
            return col_list
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in get_col_with_zero_std_deviation method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'fetching column list with standard deviation of zero failed. get_col_with_zero_std_deviation method of the Preprocessor class!!')
            raise e



    def scale_numerical_columns(self, data, categorical_columns):
        """
           Method Name: scale_numerical_columns
           Description: This method get a data frame,& list of categorical columns in the daat & finds the numerical columns and scale them
                        using Minmax scaler() of preprocessing from sklearn.
                        the scaler is saved at location  'preprocessing_data/standardScaler.pkl' for use in prediction.
           Output: Returns data with numerical columns scaled .
           On Failure: Raise Exception

           Written By: richabudhraja8@gmail.com
           Version: 1.0
           Revisions: None

        """
        scaler_path = 'preprocessing_data/MinMaxScaler.pkl'
        self.log_writer.log(self.file_obj, "Entered scale_numerical_columns of preprocessing class!!")
        try:
            # find the numerical columns
            numerical_col_list = list(numpy.setdiff1d(list(data.columns), categorical_columns))
            # get data frames for tarin & test
            df_num= data[numerical_col_list]


            # define standard scaler object
            scaler = sklearn.preprocessing.MinMaxScaler()

            # fitting the scaler on data set
            scaler.fit(df_num)
            # saving the scaler obj
            pickle.dump(scaler, open(scaler_path, 'wb'))

            # transform data set
            df_scaled = scaler.transform(df_num)

            # scaled data is a array convert back to data frame
            Scaled_df= pandas.DataFrame(df_scaled, columns=df_num.columns.tolist(), index=df_num.index)

            self.log_writer.log(self.file_obj,
                                "Scaling numerical columns successful.Exited scale_numerical_columns of preprocessing class!!")
            return Scaled_df

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in scale_numerical_columns method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'Scaling numerical columns failed. Exited scale_numerical_columns method of the Preprocessor class!!')
            raise e

    def impute_Categorical_values(self, data, columns_with_null):
        """
                        Method Name: impute_Categorical_values
                        Description: TThis method get list of  the columns which have categorical value & has nulls
                                     cin the columns. it imputes the null with mode of the column
                        Output: Returns data with imputed value inplace of nulls
                        On Failure: Raise Exception

                        Written By: richabudhraja8@gmail.com
                        Version: 1.0
                        Revisions: None

                        """
        self.log_writer.log(self.file_obj, "Entered impute_Categorical_values of preprocessing class!!")
        try:
            for col in columns_with_null:
                # only for category columns in list of columns that has null values , fill them with mode
                if ((data[col].dtypes) == 'object') or ((data[col].dtypes) == 'category'):
                    data[col].fillna(data[col].mode().values[0], inplace=True)

            self.log_writer.log(self.file_obj,
                                "imputing null value in categorical features successful. Exited the impute_Categorical_values method of the Preprocessor class!!")
            return data

        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in impute_Categorical_values method of the Preprocessor class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'imputing null value in categorical features  failed. impute_Categorical_values method of the Preprocessor class!!')
            raise e
