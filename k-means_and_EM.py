# Nadav Gover 308216340

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats


# Part A
class Kmeans(object):
    """A two dimensional implementation of k-means algorithm"""
    def __init__(self, gaussians, k=2, sample_amount=2000):
        """inputs:
            k: number of k-means
            sample_amount: number of samples to generate
            gaussains: array of gaussian distribiution in the form of [[w1,miu1,sigma1],[w2,miu2,sigma2],...]"""

        self.k = k
        self.sample_amount = sample_amount
        self.gaussians = gaussians
        self.wk = np.array([gaussian[0] for gaussian in self.gaussians])  # getting all the wk's
        self.means = np.array([gaussian[1] for gaussian in self.gaussians])  # getting all the means
        self.covs = np.array([gaussian[2] for gaussian in self.gaussians])  # getting all the covariances
        self.samples = self.generate_samples()  # samples[0] is the x values, samples[1] is the y values, This is the actual clusters
        self.centroids = self.initialize_centroids()
        self.initialization_mean_vectors = np.copy(self.centroids)
        self.clusters = [[] for _ in range(self.k)]  # These are the calculated clusters
        self.old_clusters = np.copy(self.clusters)
        self.mean_vectors_per_iteration = {0: self.initialization_mean_vectors}  # key is iteration, value is mean vector

    def generate_samples(self, sample_amount=None, wk=None, means=None, covs=None):
        """Generates samples out of a gaussian process
        Returns an array of shape (2, sample_amount)
        which output[0] is x values and output[1] is y values"""

        # Using the class fields if none are given
        if sample_amount is None:
            sample_amount = self.sample_amount
        if wk is None:
            wk = self.wk
        if means is None:
            means = self.means
        if covs is None:
            covs = self.covs

        xs = []  # like x in plural
        ys = []  # like y in plural
        for w, mean, cov in zip(wk, means, covs):
            # each gaussian gets a different sample amount depending on the w
            samples = int(w * sample_amount)  # the int() is just to make it a natural number
            x, y = np.random.multivariate_normal(mean, cov, samples).T  # generate random samples out of this distribution
            xs.extend(x)
            ys.extend(y)

        return np.array([xs, ys])

    def plot_samples(self, samples=None, show_plot=False):
        """Plots the samples
        Assuming k=2 because of the colors"""
        if samples is None:
            samples = self.samples

        plt.figure()
        colors = ["orange"] * 1000 + ["blue"] * 1000  # Assuming 1000 samples of each of the 2 gaussians, its true for this assignment
        plt.scatter(samples[0], samples[1], s=5, color=colors)

        means = self.means.T
        plt.scatter(means[0], means[1], color="black", s=50, alpha=0.5)  # plot the centroids
        plt.annotate("centroid: ({}, {})".format(means[0][0], means[1][0]), xy=means[:,0],
                     xytext=means[:, 0] - 3, arrowprops={"arrowstyle": "->"})
        plt.annotate("centroid: ({}, {})".format(means[0][1], means[1][1]), xy=means[:, 1],
                     xytext=means[:, 1] + 2, arrowprops={"arrowstyle": "->"})
        plt.axis('equal')
        plt.title("The Generated Data Set (True Label)")
        if show_plot:
            plt.show()


    def plot_clusters(self, clusters=None, centroids=None, show_plot=False, run=1):
        """Plots the clusters
        Assuming k=2 because of the colors"""
        if clusters is None:
            clusters = self.clusters
        if centroids is None:
            centroids = self.centroids

        plt.figure()
        colors = ["orange", "blue"]


        for i in range(len(clusters)):
            cluster = clusters[i]
            color = [colors[i]] * cluster.shape[1]
            plt.scatter(cluster[0], cluster[1], s=5, color=color)

        plt.scatter(centroids[0], centroids[1], color="black", s=50, alpha=0.5)  # plot the calculated centroids

        plt.annotate("centroid: ({:0.3f}, {:0.3f})".format(centroids[0][0], centroids[1][0]), xy=centroids[:, 0],
                     xytext=centroids[:,0] - 3, arrowprops={"arrowstyle": "->"})
        plt.annotate("centroid: ({:0.3f}, {:0.3f})".format(centroids[0][1], centroids[1][1]), xy=centroids[:, 1],
                     xytext=centroids[:, 1] + 2, arrowprops={"arrowstyle": "->"})
        plt.axis('equal')
        plt.title("Kmeans Clusters, Run number {}".format(run))

        if show_plot:
            plt.show()

    def print_mean_vectors_in_iteration(self, iterations=(0, 2, 10, 100), mean_vectors_per_iteration=None, run=1):
        """Prints the mean vectors in certain iterations"""
        if mean_vectors_per_iteration is None:
            mean_vectors_per_iteration = self.mean_vectors_per_iteration

        print("****************")
        print("Run number {}".format(run))

        for iteration in iterations:
            if iteration == 0:
                print("Initialization Vectors:\n {}\n".format(mean_vectors_per_iteration[iteration]))
            else:
                print("Mean Vectors in iteration {}:\n {}\n".format(iteration, mean_vectors_per_iteration[iteration]))

    def print_wk(self):
        print("The final Wk calculated is: {}".format(self.wk))

    def print_parameters(self, iterations=(0, 2, 10, 100), run=1):
        self.print_mean_vectors_in_iteration(iterations=iterations, run=run)
        self.print_wk()

    def plot_all(self):
        self.plot_clusters()
        self.plot_samples()
        plt.show()


    def initialize_centroids(self, samples=None, k=None):
        """Initialize the centroids, meaning the center of each cluster.
        Nothing too smart here, just picking a random point to be the center
        Returns array of shape (2, number of clusters k) each column is x, y of the i-th centroid"""

        if samples is None:
            samples = self.samples
        if k is None:
            k = self.k

        centroids = None
        indices = []
        for cluster in range(k):
            random_index = random.randint(0, len(samples[0]) - 1)  # just picking a random point to be the center

            # Don't choose the same center for different clusters
            while random_index in indices:  # unlikely to get in here, but just in case
                random_index = random.randint(0, len(samples[0]) - 1)  # pick again until not the same point

            indices.append(random_index)

            if centroids is None:
                centroids = np.array([[samples[0][random_index]], [samples[1][random_index]]])
            else:
                center = np.array([[samples[0][random_index]], [samples[1][random_index]]])
                centroids = np.hstack((centroids, center))

        centroids = np.array(centroids)

        # The next part is sorting the centroids
        # this part is not necessary but it makes the plotting of the clusters prettier
        argsort = np.argsort(centroids)
        centroids_sorted = np.zeros(centroids.shape)
        for i in range(len(argsort)):
            centroids_sorted[0][i] = centroids[0][argsort[0][i]]
            centroids_sorted[1][i] = centroids[1][argsort[0][i]]

        return centroids_sorted

    def reinitialize_centroids(self, samples=None, k=None):
        """Initialize the centroids, meaning the center of each cluster.
                Nothing too smart here, just picking a random point to be the center
                Returns array of shape (2, number of clusters k) each column is x, y of the i-th centroid"""

        if samples is None:
            samples = self.samples
        if k is None:
            k = self.k

        self.centroids = self.initialize_centroids(samples=samples, k=k)
        self.initialization_mean_vectors = np.copy(self.centroids)
        self.mean_vectors_per_iteration = {0: self.initialization_mean_vectors}

    def assign_samples_to_cluster(self, centroids=None, samples=None, update_wk=True):
        """Assigns a sample to a cluster
        A sample belongs to the cluster that its centroid is the closest to it
        Updates self.clusters
        If update_wk is True so also updates self.wk"""
        if centroids is None:
            centroids = self.centroids
        if samples is None:
            samples = self.samples

        # self.old_clusters = np.copy(self.clusters)  # remember the clusters before we assign_samples_to_cluster new clusters
        self.clusters = [[] for _ in range(self.k)]  # reinitialize the clusters from scratch (empty them)

        distances = None
        for i in range(self.k):
            centroid = centroids[:, i].reshape((2, 1))  # current centroid under test
            distance = self.euclidean_distance(centroid, samples).reshape((1, self.sample_amount))
            if distances is None:
               distances = distance
            else:
                distances = np.vstack((distances, distance))

        # By now each row in distances represents the distance of each sample from the i'th centroid

        closest_centroid_for_sample = np.argmin(distances, axis=0)

        # The actual assigning
        for i, closest_cluster in enumerate(closest_centroid_for_sample):
            sample = samples[:, i].reshape(2, 1)
            if type(self.clusters[closest_cluster]) == list:  # if the cluster is empty
                # self.clusters[closest_cluster].append(samples[:, i].reshape(2, 1))
                self.clusters[closest_cluster]= sample
            else:
                self.clusters[closest_cluster] = np.hstack((self.clusters[closest_cluster], sample))

        # update wk
        if update_wk:
            for i, cluster in enumerate(self.clusters):
                self.wk[i] = cluster.shape[1] / self.sample_amount

    def update_centroids(self, clusters=None):
        """Calculate the centroids of the clusters, updates self.centroids"""
        if clusters is None:
            clusters = self.clusters

        centroids = None
        for cluster in clusters:
            # cluster = np.array(cluster)
            # cluster = cluster.reshape(cluster.shape[1], cluster.shape[0])
            center = np.sum(cluster, axis=1) / cluster.shape[1]  # calculating the mean
            center = center.reshape((2, 1))
            if centroids is None:
                centroids = center
            else:
                centroids = np.hstack((centroids, center))

        self.centroids = centroids

    def euclidean_distance(self, centroid, samples):
        """Return the Euclidean distance between a point and all other points
        centroid: shape (2, 1)
        samples: shape(2, number of samples)"""

        return np.linalg.norm(centroid - samples, axis=0)

    def train(self, samples=None, iterations=100):
        if samples is None:
            samples = self.samples

        for i in range(iterations):
            self.assign_samples_to_cluster(samples=samples)
            self.update_centroids()

            # saving the values of the means of each iteration (for section b)
            self.mean_vectors_per_iteration[i + 1] = self.centroids


def execute_part_a(number_of_runs=2, show_plots=False):
    """Runs part a of the assignment.
    Returns the k-means parameters of the last run"""
    # Given in the assignment
    w1 = 0.5
    mean1 = np.array([1, 1])
    cov1 = np.array([[1, 0], [0, 2]])
    w2 = 0.5
    mean2 = np.array([3, 3])
    cov2 = np.array([[2, 0], [0, 0.5]])
    gaussians = np.array([[w1, mean1, cov1], [w2, mean2, cov2]])

    # Applying k-means algorithm
    kmeans = Kmeans(gaussians=gaussians, k=2, sample_amount=2000)
    kmeans.plot_samples()
    for run in range(number_of_runs):
        kmeans.train(iterations=100)
        kmeans.print_parameters(iterations=[0, 2, 10, 100], run=run+1)
        kmeans.plot_clusters(run=run+1)

        # Initialize different mean vectors for the next run
        if run != number_of_runs - 1:  # don't reinitialize in the last run so we save the centroids
            kmeans.reinitialize_centroids()

    if show_plots:
        plt.show()

    return kmeans

# Part B
class EM(object):
    """A 2d implementation of EM algorithm"""
    def __init__(self, k=2, samples=None, sample_amount=2000, wk=None, centroids=None, covs=None):
        self.k = k
        self.samples = samples
        self.sample_amount = sample_amount
        self.wk = wk
        self.centroids = centroids
        if covs is None:
            self.covs = self.initialize_covs()
        else:
            self.covs = covs

        self.mean_vectors_per_iteration = {0: self.centroids}
        self.covs_vectors_per_iteration = {0: self.covs}
        self.wk_vectors_per_iteration = {0: self.wk}

        self.validate()

    def validate(self):
        if self.samples is None or self.wk is None or self.centroids is None:
            raise ValueError("Missing input for samples ir wk or centroids")

        if self.samples.shape[0] != 2:
            raise ValueError("This algorithm supports only 2d")

        if self.samples.shape[1] != self.sample_amount:
            raise ValueError("Samples and samples_amount don't match")

    def initialize_covs(self, diagonal=True):
        """Initializes the covariance matrix with random values.
        The values are positive so the matrix will be positive semi definite"""
        covs = np.zeros((self.k, 2, 2))
        for i in range(self.k):
            random_number = random.randint(1, 5)
            if diagonal:
                cov = np.array([[random_number, 0], [0, random_number]])
            else:  # will never get here
                cov = np.array([[random_number, random_number], [random_number, random_number]])
            covs[i] = cov

        return covs

    def e_step(self):
        """Preforms the E-step of the EM algorithm
        The formulation is in the pdf"""
        t = self.calculate_responsibility()
        return t

    def m_step(self, responsibility):
        self.wk = self.calculate_new_wk(responsibility)
        self.centroids = self.calculate_new_centroids(responsibility)
        self.covs = self.calculate_new_covs(responsibility, self.centroids)

    def calculate_new_wk(self, responsibility):
        """Calculate new wk, see PDF for more details"""
        return np.sum(responsibility, axis=1) / self.sample_amount

    def calculate_new_centroids(self, responsibility):
        """Calculate new centroids, see PDF for more details"""
        return (np.dot(responsibility, self.samples.T) / np.sum(responsibility, axis=1)).T

    def calculate_new_covs(self, responsibility, new_centroids):
        """Calculate new covariances, see PDF for more details"""

        covs = []
        for k in range(self.k):
            sum_ = 0
            for i in range(self.sample_amount):
                x_minus_mu = self.samples[:, i].reshape((2, 1)) - new_centroids[:, k].reshape((2, 1))
                helper = np.dot(x_minus_mu, x_minus_mu.T)
                sum_ += responsibility[k][i] * helper
            cov = sum_ / np.sum(responsibility[k])
            covs.append(cov)

        covs = np.array(covs)
        return covs

    def calculate_responsibility(self, wk=None, centroids=None, covs=None):
        """Calculate the T matrix for the E step, according to the attached pdf"""
        if wk is None:
            wk = self.wk
        if centroids is None:
            centroids = self.centroids
        if covs is None:
            covs = self.covs

        pdf_1 = scipy.stats.multivariate_normal(mean=centroids[:,0].reshape((2,)), cov=covs[0]).pdf(self.samples.T)
        pdf_2 = scipy.stats.multivariate_normal(mean=centroids[:,1].reshape((2,)), cov=covs[1]).pdf(self.samples.T)
        denominator = wk[0] * pdf_1 + wk[1] * pdf_2
        responsibility = None
        for k in range(self.k):
            pdf_k = scipy.stats.multivariate_normal(mean=centroids[:, k].reshape((2,)), cov=covs[k]).pdf(self.samples.T)
            numerator = wk[k] * pdf_k
            t_ki = numerator / denominator
            if responsibility is None:
                responsibility = t_ki
            else:
                responsibility = np.vstack((responsibility, t_ki))

        return responsibility

    def plot_responsibilities(self, wk=None, centroids=None, covs=None, show_plot=False):
        """Plots the responsibilities"""
        if wk is None:
            wk = self.wk
        if centroids is None:
            centroids = self.centroids
        if covs is None:
            covs = self.covs

        responsibility = self.calculate_responsibility(wk=wk, centroids=centroids, covs=covs)

        plt.figure()
        plt.scatter(self.samples[0], self.samples[1], s=5, c=responsibility[0])

        # plot the centroids
        plt.scatter(centroids[0], centroids[1], color="black", s=50, alpha=0.5)  # plot the calculated centroids

        plt.annotate("centroid: ({:0.3f}, {:0.3f})".format(centroids[0][0], centroids[1][0]), xy=centroids[:, 0],
                     xytext=centroids[:, 0] - 3, arrowprops={"arrowstyle": "->"})
        plt.annotate("centroid: ({:0.3f}, {:0.3f})".format(centroids[0][1], centroids[1][1]), xy=centroids[:, 1],
                     xytext=centroids[:, 1] + 2, arrowprops={"arrowstyle": "->"})
        plt.axis('equal')

        # title
        plt.title("Responsibilities Generated by EM Algorithm")
        if show_plot:
            plt.show()

    def train(self, iterations=100):
        """Trains the model with EM algorithm"""
        for i in range(iterations):
            responsibility = self.e_step()
            self.m_step(responsibility)

            # saving the parameters of the model each iteration (for section b)
            self.mean_vectors_per_iteration[i + 1] = self.centroids
            self.covs_vectors_per_iteration[i + 1] = self.covs
            self.wk_vectors_per_iteration[i+ 1] = self.wk

    def print_parameters(self, iterations=(0, 2, 10, 100)):
        """Print the model parameters in certain iterations"""
        print("****************")
        print("Parameters of EM algorithm")
        for i in iterations:
            print("****************")
            print("Parameters after {} iterations:".format(i))
            print("Mean Vectors:\n{}\n".format(self.mean_vectors_per_iteration[i]))
            print("Covariances Vectors:\n{}\n".format(self.covs_vectors_per_iteration[i]))
            print("Wk vectors:\n{}\n".format(self.wk_vectors_per_iteration[i]))


def execute_part_b(kmeans):
    em = EM(k=kmeans.k, samples=kmeans.samples, wk=kmeans.wk, centroids=kmeans.centroids)
    em.train()
    em.print_parameters()
    em.plot_responsibilities(show_plot=False)

def main(show_plots=False):
    kmeans = execute_part_a(show_plots=False)
    execute_part_b(kmeans=kmeans)
    if show_plots:
        plt.show()

if __name__ == '__main__':

    # Entry point of the script
    main(show_plots=True)



