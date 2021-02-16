import json
import os
import re
import shutil

import pandas

from Application_logging.logger import app_logger
from File_operation_prediction.file_operation import file_operation

class raw_data_validation:

    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = 'Schema Files/schema_prediction.json'
        self.good_data_loc= "Prediction_Raw_files_validated/Good_Raw"
        self.bad_data_loc = "Prediction_Raw_files_validated/Bad_Raw"
        self.file_op_obj =file_operation()
        self.log_writer = app_logger()

    def values_from_schema(self):
        """
            Method Name: valuesFromSchema
            Description: This method extracts all the relevant information from the pre-defined "Schema" file.
            Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
            On Failure: Raise ValueError,KeyError,Exception
            Associated log file : "Prediction_Logs/valuesfromSchemaValidationLog.txt"

            Written By: richabudhraja8@gmail.com
            Version: 1.0
            Revisions: None

        """
        log_file = "Prediction_Logs/valuesfromSchemaValidationLog.txt"

        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            file_obj = open(log_file, 'a+')
            message = "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile + "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile + "\t " + "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.log_writer.log(file_obj, message)

            file_obj.close()



        except ValueError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "ValueError:Value not found inside schema_training.json")
            file_obj.close()
            raise ValueError

        except KeyError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "KeyError:Key value error incorrect key passed")
            file_obj.close()
            raise KeyError

        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, str(e))
            file_obj.close()
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns

    def mannual_regex_creation(self, LengthOfDateStampInFile, LengthOfTimeStampInFile):
        """
         Method Name: manualRegexCreation
         Description: This method contains a manually defined regex based on the sample "FileName" given in "Schema" file.
                      This Regex is used to validate the filename of the training data.
         Output: Regex pattern
         On Failure: None

         Written By: richabudhraja8@gmail.com
         Version: 1.0
         Revisions: None

    """
        first_word = 'salesPrediction'
        l1 = LengthOfDateStampInFile
        l2 = LengthOfTimeStampInFile
        regex = '^%s' % first_word + '_[0-9]{%s}' % l1 + '_[0-9]{%s}' % l2 + '.csv$'
        return regex
       

    def validate_file_name_raw(self, regex):
        """
        Method Name: validationFileNameRaw
        Description: This function validates the name of the training csv files as per given name in the schema!
                                         Regex pattern is used to do the validation.If name format do not match the file is moved
                                         to Bad Raw Data folder else in Good raw data.
        Output: None
        On Failure: Exception
        Associated log file :"Prediction_Logs/nameValidationLog.txt"
        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None

        """

        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.file_op_obj.deleteExistingBadDataPredictionFolder()
        self.file_op_obj.deleteExistingGoodDataPredictionFolder()
        # create new directories
        self.file_op_obj.createDirectoryForGoodBadRawData()
        log_file="Prediction_Logs/nameValidationLog.txt"

        all_files = [f for f in os.listdir(self.Batch_Directory)]
        pattern = re.compile(r"%s" % regex)
        try:
            file_obj = open(log_file, 'a+')
            for filename in all_files:
                origin_loc = os.path.join(self.Batch_Directory,filename)
                result = pattern.match(filename)
                if result is not None:
                    shutil.copy(origin_loc,self.good_data_loc )
                    self.log_writer.log(file_obj, "Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                else:
                    shutil.copy(origin_loc,self.bad_data_loc )
                    self.log_writer.log(file_obj, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    
            file_obj.close()

        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error occured while validating FileName %s" % e)
            file_obj.close()
            raise e

    def validate_column_length(self, noofcolumns):
        """
        Method Name: validateColumnLength
        Description: This function validates the number of columns in the csv files.
                        It is should be same as given in the schema file.
                        If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                        If the column number matches, file is kept in Good Raw Data for processing.

        Output: None
        Associated log file : "Prediction_Logs/columnValidationLog.txt"
        On Failure: Exception


        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None

    """
        log_file ="Prediction_Logs/columnValidationLog.txt"
        try:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Column Length Validation Started!!")
            for file in os.listdir(self.good_data_loc):
                file_loc= os.path.join(self.good_data_loc,file)
                csv = pandas.read_csv(file_loc)
                if csv.shape[1] == noofcolumns:
                    pass
                else:
                    shutil.move(file_loc, self.bad_data_loc)
                    self.log_writer.log(file_obj, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.log_writer.log(file_obj, "Column Length Validation Completed!!")
        except OSError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error Occured while moving the file :: %s" % OSError)
            file_obj.close()
            raise OSError
        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error Occured:: %s" % e)
            file_obj.close()
            raise e
        file_obj.close()

    def validate_whole_column_isnull(self):
        """
        Method Name: validateMissingValuesInWholeColumn
        Description: This function validates if any column in the csv file has all values missing.
                      If all the values are missing, the file is not suitable for processing.
                      SUch files are moved to bad raw data.
        Output: None
        Associated log file :"Prediction_Logs/missingValuesInColumn.txt"
        On Failure: Exception

        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None
        """
        log_file ="Prediction_Logs/missingValuesInColumn.txt"
        try:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Missing Values Validation Started!!")
            all_files=os.listdir(self.good_data_loc)
            for file in all_files:
                file_loc = os.path.join(self.good_data_loc, file)
                csv = pandas.read_csv(file_loc)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) != csv[columns].count()):
                        count += 1
                        shutil.move(file_loc, self.bad_data_loc)
                        self.log_writer.log(file_obj, "Invalid Column for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
            self.log_writer.log(file_obj, "Missing Values Validation completed !!")
            file_obj.close()

        except OSError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error Occured while moving the file :: %s" % OSError)
            file_obj.close()
            raise OSError
        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error Occured:: %s" % e)
            file_obj.close()
            raise e
        file_obj.close()


