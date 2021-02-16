import csv
import os
import shutil
import urllib
import pandas
import pyodbc
from Application_logging import logger


class db_operation:
    def __init__(self):
        self.log_writer = logger.app_logger()
        # name of the table to be created in database to store good raw data
        self.table_name = 'Good_Raw_Data'

    def data_base_operations(self, training_db_name, column_names):
        self.createTable(training_db_name, column_names)
        self.insertIntoTableGoodData(training_db_name, column_names)

    def dataBaseConnection(self, databaseName):
        """
            Method Name: dataBaseConnection
            Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
            Output: Connection Engine to the DB
            On Failure: Raise ConnectionError
            Associated Log file :Prediction_Logs/DataBaseConnectionLog.txt

            written by : richabudhraja8@gmail.com
            version 1.0
            revisions : None

        """

        log_file = "Prediction_Logs/DataBaseConnectionLog.txt"

        db_details = pandas.read_csv('Database/dbdetails.csv')

        driver = db_details['driver'].values[0]
        server = db_details['server'].values[0]
        db = databaseName  # db_details['db'].values[0]
        username = db_details['username'].values[0]
        password = db_details['password'].values[0]



        try:
            file_obj = open(log_file, 'a+')

            conn = pyodbc.connect(
                driver + ";SERVER=" + server + ";DATABASE=" + db + ";UID=" + username + ";PWD=" + password)
            self.log_writer.log(file_obj, "Opened %s database successfully" % db)

            file_obj.close()

        except ConnectionError:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error while Connecting to Database  %s " % ConnectionError)
            file_obj.close()
            raise ConnectionError
        return conn

    def createTable(self, databaseName, column_names):
        """
                Method Name: createTable
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Create table for good raw data
                On Failure: Exception
                Associated Log file :Prediction_Logs/DbTableCreateLog.txt

                written by : richabudhraja8@gmail.com
                version 1.0
                revisions : None

        """
        log_file = "Prediction_Logs/DbTableCreateLog.txt"
        log_filedb = "Prediction_Logs/DataBaseConnectionLog.txt"

        try:
            conn = self.dataBaseConnection(databaseName)
            cursor = conn.cursor()
            # check if table already exists
            result = cursor.execute(
                "select count(TABLE_NAME) from {dbname}.INFORMATION_SCHEMA.TABLES where TABLE_NAME = '{tablename}'".format(
                    dbname=databaseName, tablename=self.table_name))
            if (result.fetchone()[0] == 1):  # table  exists
                file_obj = open(log_file, 'a+')
                self.log_writer.log(file_obj, "Table Already Exists")
                file_obj.close()

                conn.close()

                file_obj = open(log_filedb, 'a+')
                self.log_writer.log(file_obj, "closed %s database connection successfully" % databaseName)
                file_obj.close()
            else:

                for key in column_names.keys():
                    type_ = column_names[key]
                    if type_ == 'VARCHAR':
                        type_ = type_ + '(255)'
                    try:
                        cursor.execute(
                            'ALTER TABLE  {tablename} ADD  "{column_name}" {dataType}'.format(tablename=self.table_name,
                                                                                              column_name=key,
                                                                                              dataType=type_))

                    except:
                        cursor.execute(
                            'CREATE TABLE  {tablename} ({column_name} {dataType})'.format(tablename=self.table_name,
                                                                                          column_name=key,
                                                                                          dataType=type_))
                conn.commit()
                conn.close()

                file_obj = open(log_file, 'a+')
                self.log_writer.log(file_obj, "Table %s created successfully" % self.table_name)
                file_obj.close()

        except Exception as e:
            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error while creating table: %s " % e)
            file_obj.close()
            conn.rollback()
            conn.close()

            file_obj = open(log_filedb, 'a+')
            self.log_writer.log(file_obj, "Closed %s database successfully" % databaseName)
            file_obj.close()
            raise e

    def insertIntoTableGoodData(self, databaseName, column_names):
        """
                Method Name: insertIntoTableGoodData
                Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                Output: none
                On Failure: Exception
                Associated Log file :"Prediction_Logs/DbInsertLog.txt"

                written by : richabudhraja8@gmail.com
                version 1.0
                revisions : None
                """
        good_data_loc = "Prediction_Raw_files_validated/Good_Raw"
        bad_data_loc = "Prediction_Raw_files_validated/Good_Raw"
        log_file = "Prediction_Logs/DbInsertLog.txt"
        all_files = [f for f in os.listdir(good_data_loc)]
        conn = self.dataBaseConnection(databaseName)
        cursor = conn.cursor() #create a cursor
        file_obj = open(log_file, 'a+')
        type_ = list(column_names.values())

        for file_name in all_files:
            # step 1 open the file.
            file_loc = os.path.join(good_data_loc, file_name)
            try:
                with open(file_loc) as f:
                    next(f)  # skip the header
                    # step 2 create a reader csv object & enumerate for each line
                    reader = csv.reader(f, delimiter='\n')
                    for line in enumerate(reader):
                        # for list in each line insert into db
                        for list_ in line[1]:  # line[0] is index so skip the index
                            values_ = list_.split(',')
                            val_ = []
                            for i in range(len(values_)):
                                if type_[i] != 'VARCHAR':
                                    val_.append(eval(values_[i]))
                                elif type_[i] == 'VARCHAR':
                                    val_.append(str(values_[i]))
                            try:
                                cursor.execute(
                                    'Insert into {tablename} values {values}'.format(tablename=self.table_name,
                                                                                     values=tuple(val_)))

                                self.log_writer.log(file_obj, " inserting into %s table Successful" % self.table_name)
                                conn.commit()

                            except Exception as e:

                                raise e


            except Exception as e:
                # if mid way betweeen dile., error happens, cant save half values from file so rollback)
                conn.rollback()
                file_obj = open(log_file, 'a+')
                self.log_writer.log(file_obj, "Error while inserting into table %s" % e)
                # move that file to bad file path
                shutil.move(file_loc, bad_data_loc)
                self.log_writer.log(file_obj, 'moved file to bad data folder %s' % file_name)
                file_obj.close()
                conn.close()

        file_obj.close()
        conn.close()

    def selectingDatafromtableintocsv(self, databaseName):
        """
                Method Name: selectingDatafromtableintocsv
                Description: This method exports the data in Good_raw_Data table as a CSV file. in a given location.
                                            above created .
                Output: none
                On Failure: Exception
                Associated Log file :"Prediction_Logs/ExportToCsv.txt"

                written by : richabudhraja8@gmail.com
                version 1.0
                revisions : None
        """
        fileFromDb = 'Prediction_FileFromDB/'  # location where files will be saved
        fileName = 'InputFile.csv'  # name of the file
        log_file = "Prediction_Logs/ExportToCsv.txt"

        try:
            file_obj = open(log_file, 'a+')

            # step 1 create db connection & read all data from good_raw_data table in database
            conn = self.dataBaseConnection(databaseName)
            sql = "select * from {tablename}".format(tablename=self.table_name)

            # step 2 create a cursor , and get headers and results or use pandas to get data to dataframe
            data = pandas.read_sql(sql, conn)

            # step 3 create the output directory if doesnot already exist
            if not os.path.isdir(fileFromDb):
                os.mkdir(fileFromDb)

            # step 4 open the csv file for writing in case of cursor and write the headers & results
            # or for pandas directly write to csv
            file_loc = os.path.join(fileFromDb, fileName)
            data.to_csv(file_loc, index=False)

            self.log_writer.log(file_obj, "File exported succesfully at %s " % file_loc)

            file_obj.close()
            conn.close()
        except Exception as e:

            file_obj = open(log_file, 'a+')
            self.log_writer.log(file_obj, "Error while exporting File .%s " % e)
            file_obj.close()
            conn.close()
