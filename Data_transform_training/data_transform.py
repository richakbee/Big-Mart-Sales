import os

import pandas

from Application_logging import logger


class data_transform:

    def __init__(self):
        self.good_data_loc = "Training_Raw_files_validated/Good_Raw"
        self.log_writer = logger.app_logger()

    def replace_missing_values_with_null(self):
        """
             Method Name: replaceMissingWithNull
            Description: This method replaces the missing values in columns with "NULL" to
                        store in the table.
            Output:None
            On Failure: Exception
            Associated Log file :Training_Logs/dataTransformLog.txt

            written by : richabudhraja8@gmail.com
            version 1.0
            revisions : None

        """
        log_file = "Training_Logs/dataTransformLog.txt"
        # good_data_loc = "Training_Raw_files_validated/Good_Raw"
        all_files = [f for f in os.listdir(self.good_data_loc)]
        try:
            for file_name in all_files:
                file_obj = open(log_file, 'a+')
                file_loc = os.path.join(self.good_data_loc, file_name)
                csv = pandas.read_csv(file_loc)
                csv.fillna('NULL', inplace=True)
                csv.to_csv(file_loc, index=False, header=True)
                self.log_writer.log(file_obj, 'Null Values replaced successfully')

                file_obj.close()

        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj,
                                "Error occurred during data transformation while replacing missing values with null:: %s" % e)
            file_obj.close()
