import csv
import os
import shutil
from datetime import datetime

from Application_logging.logger import app_logger


class file_operation:

    def __init__(self):
        self.good_data_loc = "Prediction_Raw_files_validated/Good_Raw"
        self.bad_data_loc = "Prediction_Raw_files_validated/Bad_Raw"
        self.output_data_loc ="Prediction_Output_File"
        self.archive_bad_data_loc = "PredictionArchiveBadData"
        self.log_writer = app_logger()

    def deleteExistingGoodDataPredictionFolder(self):
        """
            Method Name: deleteExistingGoodDataTrainingFolder
            Description: This method deletes the directory made  to store the Good Data
                         after loading the data in the table. Once the good files are
                        loaded in the DB,deleting the directory ensures space optimization.
            Output: none
            On Failure: OSError
            Associated Log file :"Prediction_Logs/GeneralLog.txt"

            written by : richabudhraja8@gmail.com
            version 1.0
            revisions : None
        """
        log_file = "Prediction_Logs/GeneralLog.txt"
        try:
            if os.path.isdir(self.good_data_loc):
                shutil.rmtree(self.good_data_loc)
                file_obj = open(log_file, 'a+')
                self.log_writer.log(file_obj,
                                    " Good Raw directory at location %s deleted succesfully" % self.good_data_loc)
                file_obj.close()

        except OSError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj,
                                " Error {} while deleting file at location {}".format(OSError, self.good_data_loc))
            file_obj.close()
            raise OSError

    def deleteExistingBadDataPredictionFolder(self):
        """
                   Method Name: deleteExistingBadDataTrainingFolder
                   Description: This method deletes the directory made  to store the Bad Data

                   Output: none
                   On Failure: OSError
                   Associated Log file :"Prediction_Logs/GeneralLog.txt"

                   written by : richabudhraja8@gmail.com
                   version 1.0
                   revisions : None
               """
        log_file = "Prediction_Logs/GeneralLog.txt"
        try:
            if os.path.isdir(self.bad_data_loc):
                shutil.rmtree(self.bad_data_loc)
                file_obj = open(log_file, 'a+')
                self.log_writer.log(file_obj,
                                    " Bad Raw directory at location %s deleted succesfully" % self.bad_data_loc)
                file_obj.close()

        except OSError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj,
                                " Error {} while deleting file at location {}".format(OSError, self.bad_data_loc))
            file_obj.close()
            raise OSError

    def createDirectoryForGoodBadRawData(self):
        """
          Method Name: createDirectoryForGoodBadRawData
          Description: This method creates directories to store the Good Data and Bad Data
                                                            after validating the training data.

          Output: None
          On Failure: OSError
          Associated Log file :"Prediction_Logs/GeneralLog.txt"

          written by : richabudhraja8@gmail.com
          version 1.0
          revisions : None
               """
        log_file = "Prediction_Logs/GeneralLog.txt"
        try:

            if not os.path.isdir(self.good_data_loc):
                os.makedirs(self.good_data_loc)
            if not os.path.isdir(self.bad_data_loc):
                os.makedirs(self.bad_data_loc)

        except OSError as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error while creating GOOD BAD raw Directory %s:" % e)
            file_obj.close()
            raise OSError

    def moveBadFilesToArchiveBad(self):
        """
        Method Name: moveBadFilesToArchiveBad
        Description: This method deletes the directory made  to store the Bad Data
                     after moving the data in an archive folder. We archive the bad
                      files to send them back to the client for invalid data issue.
        Output: none
        On Failure: OSError,Exception
        Associated Log file :"Prediction_Logs/GeneralLog.txt"

        written by : richabudhraja8@gmail.com
        version 1.0
        revisions : None
                """
        log_file = "Prediction_Logs/GeneralLog.txt"
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            if os.path.isdir(self.bad_data_loc):
                # step 1 check if archive folder exists else create it
                if not os.path.isdir(self.archive_bad_data_loc):
                    os.mkdir(self.archive_bad_data_loc)

                all_files = [f for f in os.listdir(self.bad_data_loc)]

                if len(all_files) > 0:
                    # step 2 create the file in archive folder names BadData_currentdate_currenttime
                    file_name = 'BadData_' + str(date) + "_" + str(time)
                    dest = os.path.join(self.archive_bad_data_loc, file_name)
                    if not os.path.isdir(dest):
                        os.mkdir(dest)

                    # step 3 move all the files from bad data location to archive dest

                    for file in all_files:
                        loc = os.path.join(self.bad_data_loc,file)
                        # step 4 move the file to archive dest if not aready there
                        if file not in os.listdir(dest):
                            self.move_file(loc, dest)

                        file_obj = open(log_file, 'a+')
                        self.log_writer.log(file_obj, "Archived {} bad data at {}" .format(file,dest))
                        file_obj.close()
                else:
                    file_obj = open(log_file, 'a+')
                    self.log_writer.log(file_obj, "NO bad data to archive at location ")
                    file_obj.close()

                # step5 delete  the bad raw data directory
                if os.path.isdir(self.bad_data_loc):
                    shutil.rmtree(self.bad_data_loc)
                    file_obj = open(log_file, 'a+')
                    self.log_writer.log(file_obj, "deleted the Bad Raw Data folder successfully")
                    file_obj.close()


        except OSError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "error {} while Archiving all bad data to location {}".format(OSError, dest))
            file_obj.close()

        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "error {} while Archiving all bad data to location {}".format(e, dest))
            file_obj.close()

    def move_file(self, file_at_origin_location, destination_location):
        """
                    Method Name: move_file
                    Description: This method moves a file from orignin location to destination location
                    Output: none
                    On Failure: OSError, Exception
                    Associated Log file :None

                    written by : richabudhraja8@gmail.com
                    version 1.0
                    revisions : None
                """
        try:
            shutil.move(file_at_origin_location, destination_location)
        except OSError:
            raise OSError

        except Exception as e:
            raise e

    def delete_existing_file_create_new(self, prediction_output_file_name):
        if not os.path.isdir(self.output_data_loc):
            os.mkdir(self.output_data_loc)

        if os.path.exists(prediction_output_file_name):
            os.remove('Prediction_Output_File/Predictions.csv')


    def read_list(self, location):
        list_ = []
        with open(location) as f:
            reader = csv.reader(f)
            for row in reader:
                list_.append(''.join(row[0].split(' ')))
        return list_
