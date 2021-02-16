import csv
import os


class create_log_directories:

    """
    This class shall be used when the application is run for the very first time.
    This class will be used to create log directories an d files
    """

    def __init__(self):
        self.Training_directory = 'Training_Logs'
        self.Prediction_directory = 'Prediction_Logs'
        self.training_log_files = ['columnValidationLog.txt',
                                   'DataBaseConnectionLog.txt',
                                   'dataTransformLog.txt',
                                   'DbInsertLog.txt',
                                   'DbTableCreateLog.txt',
                                   'ExportToCsv.txt',
                                   'GeneralLog.txt',
                                   'missingValuesInColumn.txt',
                                   'ModelTrainingLog.txt',
                                   'nameValidationLog.txt',
                                   'Training_main_log.txt',
                                   'valuesfromSchemaValidationLog.txt']
        self.prediction_log_files = ['columnValidationLog.txt',
                                     'DataBaseConnectionLog.txt',
                                     'dataTransformLog.txt',
                                     'DbInsertLog.txt',
                                     'DbTableCreateLog.txt',
                                     'ExportToCsv.txt',
                                     'GeneralLog.txt',
                                     'missingValuesInColumn.txt',
                                     'nameValidationLog.txt',
                                     'Prediction_main_log.txt',
                                     'valuesfromSchemaValidationLog.txt']

    def create_directories(self):

        try:
            if not os.path.isdir(self.Training_directory):
                os.mkdir(self.Training_directory)

            if not os.path.isdir(self.Prediction_directory):
                os.mkdir(self.Prediction_directory)

            for log_file in self.training_log_files:
                dest = os.path.join(self.Training_directory, log_file)
                if not os.path.exists(dest):
                    file = open(dest, "w")
                    file.close()

            for log_file in self.prediction_log_files:
                dest = os.path.join(self.Prediction_directory, log_file)
                if not os.path.exists(dest):
                    file = open(dest, "w")
                    file.close()
        except OSError :
            print('Exception occured while creating log directories %s'%OSError)
            raise OSError

        except Exception as e :
            print('Exception occured while creating log directories %s'%e)
            raise e