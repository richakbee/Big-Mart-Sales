from sklearn.cluster import _kmeans
import matplotlib.pyplot as plt
import kneed
from Model_functions.model_functions_fileops import model_functions


class clustering:
    """
         This class shall  be used to divide the data into clusters before training.


        Written
        By: richabudhraja8@gmail.com
        Version: 1.0
        Revisions: None
    """

    def __init__(self, logger_object, file_object):

        self.file_obj = file_object
        self.log_writer = logger_object

    def elbow_plot(self, X):
        """
                Method Name: elbow_plot
                Description: This method saves the plot to decide the optimum number of clusters to the file.
                Output: A picture saved to the directory at "preprocessing_data/K-Means_Elbow.PNG"
                On Failure: Raise Exception

                Written By: richabudhraja8@gmail.com
                Version: 1.0
                Revisions: None
            """

        location_elbow_plot_image = "preprocessing_data/K-Means_Elbow.PNG"
        wcss = []
        cluster_range = range(1, 11)
        self.log_writer.log(self.file_obj, "Entered the elbow_plot method of the Clustering class")
        try:
            for i in cluster_range:
                kmeans = _kmeans.KMeans(n_clusters=i, init='k-means++')
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            # generate the plot
            plt.plot(cluster_range, wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig(location_elbow_plot_image)  # saving the elbow plot locally

            # finding the value of the optimum cluster programmatically
            optimum_cluster = kneed.KneeLocator(cluster_range, wcss, curve='convex', direction='decreasing').knee
            self.log_writer.log(self.file_obj, 'The optimum number of clusters is: ' + str(
                optimum_cluster) + ' . Exited the elbow_plot method of the Clustering class')
            return optimum_cluster
        except Exception as e:
            self.log_writer.log(self.file_obj, "Error in  the elbow_plot method of the Clustering class. %s" % e)
            raise e

    def create_clusters(self, X, no_of_clusters):
        """
           Method Name: create_clusters
           Description: Create a new dataframe consisting of the cluster information. This method also calls the save model method
                        to save the cluster model for future use in prediction.
           Output: A datframe with "Cluster" column
           On Failure: Raise Exception

           Written By: richabudhraja8@gmail.com
           Version: 1.0
           Revisions: None
        """

        data = X
        model_func_obj = model_functions(self.log_writer, self.file_obj)
        self.log_writer.log(self.file_obj, "Entered the create_clusters method of the Clustering class")
        try:
            kmeans = _kmeans.KMeans(n_clusters=no_of_clusters, init='k-means++')
            kmeans.fit(data)
            data['Cluster'] = kmeans.labels_
            # save the model
            self.log_writer.log(self.file_obj,
                                "calling the save model method from create_clusters method of the Clustering class")
            model_func_obj.save_model(kmeans, "KMeans")

            self.log_writer.log(self.file_obj, "Exited the create_clusters method of the Clustering class")
            return data

        except Exception as e:
            self.log_writer.log(self.file_obj, "Error in  the create_clusters method of the Clustering class. %s" % e)
            raise e
