from Raw_data_validation_training.raw_data_validation import raw_data_validation
from Data_transform_training.data_transform import data_transform
from Database_operation_training.db_operation import db_operation
from Application_logging.logger import app_logger
from File_operation.file_operation import file_operation


class train_validate:
    def __init__(self, path):
        self.raw_data_obj = raw_data_validation(path)
        self.data_transform_obj = data_transform()
        self.dbOperation_obj = db_operation()
        self.file_operation_obj = file_operation()
        # open log writer and write log into a Training_log file.
        self.file_obj = open("Training_logs/Training_main_log.txt", 'a+')
        self.log_writer = app_logger()
        self.training_db_name = "BackOrderTraining"

    def training_validation(self):
        try:
            # step 1 raw data validation
            self.log_writer.log(self.file_obj, " Starting  validation of files!!")

            # 1.1 extract values from schema file
            LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.raw_data_obj.values_from_schema()
            # 1.2 get regex for filename
            regex = self.raw_data_obj.mannual_regex_creation(LengthOfDateStampInFile, LengthOfTimeStampInFile)
            # 1.3 validate file name
            self.raw_data_obj.validate_file_name_raw(regex)
            # 1.4 validate column length or number of columns
            self.raw_data_obj.validate_column_length(noofcolumns)
            # 1.5 validate if a column is totally empty
            self.raw_data_obj.validate_whole_column_isnull()

            self.log_writer.log(self.file_obj, "raw validation ended!!")

            # step2  transform raw data
            self.log_writer.log(self.file_obj, "Starting data transformation!!")

            self.data_transform_obj.replace_missing_values_with_null()

            self.log_writer.log(self.file_obj, " data tranformation ended!!")

            # step 3  db Db operation
            self.log_writer.log(self.file_obj, "starting Data base operation!!")
            #creates Good_Raw_Data table in database and inserts all the content of good raw files into it.
            self.dbOperation_obj.data_base_operations(self.training_db_name, column_names)

            self.log_writer.log(self.file_obj, "Database operations ended!!")

            # step 4 get transformed data from data base into csv
            self.log_writer.log(self.file_obj, "Extracting csv file from table")

            self.dbOperation_obj.selectingDatafromtableintocsv(self.training_db_name)
            self.log_writer.log(self.file_obj, "Extracted csv file from table successfully")
            # step5 file operations
            self.log_writer.log(self.file_obj, "Deleting good training raw folder")
            self.file_operation_obj.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_obj, "good training raw folder deleted successfully")

            self.log_writer.log(self.file_obj, "Archiving bad raw data files & deleting the bad_raw folder")
            self.file_operation_obj.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_obj, "Archived bad raw data files & deleted the bad_raw folder successfully")

            self.log_writer.log(self.file_obj, " End of validation of Training files.Successful!!")
            self.file_obj.close()

        except Exception as e:
            self.log_writer.log(self.file_obj, "Validation of Training files.Unsuccessful!!")
            self.file_obj.close()
            raise e
