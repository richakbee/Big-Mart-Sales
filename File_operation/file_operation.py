import os
import shutil
from datetime import datetime
import csv
from Application_logging.logger import app_logger


class file_operation:

    def __init__(self):
        self.good_data_loc = "Training_Raw_files_validated/Good_Raw"
        self.bad_data_loc = "Training_Raw_files_validated/Bad_Raw"
        self.archive_bad_data_loc = "TrainingArchiveBadData"
        self.preprocessing_data_loc ="preprocessing_data"
        self.log_writer = app_logger()

    def deleteExistingGoodDataTrainingFolder(self):
        """
            Method Name: deleteExistingGoodDataTrainingFolder
            Description: This method deletes the directory made  to store the Good Data
                         after loading the data in the table. Once the good files are
                        loaded in the DB,deleting the directory ensures space optimization.
            Output: none
            On Failure: OSError
            Associated Log file :"Training_Logs/GeneralLog.txt"

            written by : richabudhraja8@gmail.com
            version 1.0
            revisions : None
        """
        log_file = "Training_Logs/GeneralLog.txt"
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

    def deleteExistingBadDataTrainingFolder(self):
        """
                   Method Name: deleteExistingBadDataTrainingFolder
                   Description: This method deletes the directory made  to store the Bad Data

                   Output: none
                   On Failure: OSError
                   Associated Log file :"Training_Logs/GeneralLog.txt"

                   written by : richabudhraja8@gmail.com
                   version 1.0
                   revisions : None
               """
        log_file = "Training_Logs/GeneralLog.txt"
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
          Associated Log file :"Training_Logs/GeneralLog.txt"

          written by : richabudhraja8@gmail.com
          version 1.0
          revisions : None
               """
        log_file = "Training_Logs/GeneralLog.txt"
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
        Associated Log file :"Training_Logs/GeneralLog.txt"

        written by : richabudhraja8@gmail.com
        version 1.0
        revisions : None
                """
        log_file = "Training_Logs/GeneralLog.txt"
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            if os.path.isdir(self.bad_data_loc):
                # step 1 check if archive folder exists else create it
                if not os.path.isdir(self.archive_bad_data_loc):
                    os.mkdir(self.archive_bad_data_loc)

                all_files = [f for f in os.listdir(self.bad_data_loc)]

                if len(all_files)>0:
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
                        self.log_writer.log(file_obj, "Archived {}as bad data at location {}" .format(file, dest))
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


    def save_data_to_file(self, data, location):

        with open(location, 'w', newline='\n') as f:
            writer = csv.writer(f, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                writer.writerow(row)
                
    
    def append_data_to_file(self, data, location):

        with open(location, 'a', newline='\n') as f:
            writer = csv.writer(f, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in data:
                writer.writerow(row)


    def  createDirectoryForPreprocessing(self):
        """
                  Method Name: createDirectoryForGoodBadRawData
                  Description: This method creates directories to store the data while preprocessing during training.

                  Output: None
                  On Failure: OSError
                  Associated Log file :"Training_Logs/GeneralLog.txt"

                  written by : richabudhraja8@gmail.com
                  version 1.0
                  revisions : None
                       """
        log_file = "Training_Logs/GeneralLog.txt"
        try:

            if not os.path.isdir(self.preprocessing_data_loc):
                os.makedirs(self.preprocessing_data_loc)


        except OSError as e:
            file_obj = open(log_file, 'a+')
            self.logger.log(file_obj, "Error while creating preprocessing_data Directory %s:" % e)
            file_obj.close()
            raise OSError




