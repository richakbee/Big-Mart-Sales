import os
import pickle
import re
import shutil


class model_functions:
    """
        This class shall be used to save the model after training
        and load the saved model for prediction.

        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None

    """

    def __init__(self, logger_object, file_object):
        self.file_obj = file_object
        self.log_writer = logger_object
        self.models_location = 'models/'


    def save_model(self, model, model_name):
        """
        Method Name: save_model
        Description: Save the model file to directory at models_location
        Output: File gets saved
        On Failure: Raise Exception

        Written By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None
        """
        self.log_writer.log(self.file_obj, "Entered save_model of model_functions class!!")
        try:
            #make models directory
            if not os.path.isdir(self.models_location):
                os.mkdir(self.models_location)

            model_path = os.path.join(self.models_location, model_name)
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
                os.mkdir(model_path)
            else:
                os.mkdir(model_path)
            final_model = model_path + '/' + model_name + '.sav'
            with open(final_model, 'wb') as f:
                pickle.dump(model, f)
            self.log_writer.log(self.file_obj,
                                'saving the model %s successfully.Exited the save_model of model_functions class!!' % model_name)

            return "success"
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in save_model of model_functions class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'saving the model %s failed. save_model of model_functions class!!' % model_name)
            raise e

    def load_model(self, model_name):
        """
                        Method Name: load_model
                        Description: load model for a given model name
                        Output: the model
                        On Failure: Raise Exception

                        Written By: richabudhraja8@gmail.com
                        Version: 1.0
                        Revisions: None
                        """
        dest = self.models_location + model_name + '/' + model_name + '.sav'
        self.log_writer.log(self.file_obj, "Entered load_model of model_functions class!!")
        try:

            with open(dest, 'rb') as f:  # read the model in binary mode
                self.log_writer.log(self.file_obj, "loaded the model %s Successfully!!" % model_name)
                return pickle.load(f)
        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in load_model of model_functions class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                                'loading the model %s failed. load_model of model_functions class!!' % model_name)
            raise e

    def find_correct_model_for_cluster(self, cluster_number):
        """
                Method Name: find_correct_model_for_cluster
                Description: find the correct model for a given cluster number
                Output: exact location of the model file
                On Failure: Raise Exception

                Written By: richabudhraja8@gmail.com
                Version: 1.0
                Revisions: None
        """

        self.log_writer.log(self.file_obj, "Entered find_correct_model_for_cluster of model_functions class!!")
        regex="^[a-zA-Z]+\d$"
        pattern = re.compile(r"%s" % regex)
        try:
            all_models = os.listdir(self.models_location)
            for model_name in all_models:
                result = pattern.match(model_name)
                if result is not None:
                    if int(model_name[-1]) == int(cluster_number):
                        self.log_writer.log(self.file_obj, "returned  the model %s " % model_name+"Successfully for cluster number %s"%cluster_number)
                        return model_name


        except Exception as e:
            self.log_writer.log(self.file_obj,
                                'Exception occurred in find_correct_model_for_cluster of model_functions class!! Exception message:' + str(
                                    e))
            self.log_writer.log(self.file_obj,
                             "Could not return the model for cluster number %s ."%cluster_number)
            raise e