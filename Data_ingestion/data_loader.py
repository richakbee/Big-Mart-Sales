import pandas


class data_loader:

    def __init__(self,  logger_object, file_object):

        self.file_obj = file_object
        self.log_writer = logger_object

    def get_data(self, file_name):
        """
                Method Name: get_data
                Description: This method reads the data from source.
                Output: A pandas DataFrame.
                On Failure: Raise Exception

                Written By: richabudhraja8@gmail.com
                Version: 1.0
                Revisions: None

                """
        try:
            data = pandas.read_csv(file_name)
            self.log_writer.log(self.file_obj, " Successful in getting %s data"%file_name)
            return data
        except Exception as e:
            self.log_writer(self.file_obj, " UnSuccessful in getting "+str(file_name)+" data. Error: %s" % e)
            raise e
