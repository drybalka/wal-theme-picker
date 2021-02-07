import numpy as np


def wkmean(k, data, weights=None, metric='Euclidian',
           MAX_ITERATIONS=1000, method='k++'):
    """
    Weighted k-means algorithm with population count (importance
    for each cluster. Returns the best centroids in the form
    positions, importances and fit errror.
    If no weights are provided all data points are assumed to be equal.
    If no metric is provided, the standard Euclidian metric is used.
    Possible methods are:
        'k++_pdf' for kmeans++ initialization with PDF distributed choice
        'k++_max' for kmeans++ initialization with farthest point choice
        'uniform' for uniform [-0.5, 0.5] initialization
    """

    def euclidian_metric(coords):
        return (coords**2).sum(axis=1)

    def kmeans_plus_plus_initialization(X, k, pdf_method=True):
        '''Initialize one point at random. Loop for k - 1 iterations:
        Next, calculate for each point the distance of the point from its
        nearest center. Sample a point with a probability proportional to
        the square of the distance of the point from its nearest center.'''
        def distance_to_centroids(data, centers):
            distance = np.sum((np.array(centers) - data[:, None, :])**2,
                              axis=2)
            return distance

        centers = []
        X = np.array(X)

        # Sample the first point
        initial_index = np.random.choice(range(X.shape[0]), )
        centers.append(X[initial_index, :].tolist())

        # Loop and select the remaining points
        for i in range(k - 1):
            distance = distance_to_centroids(X, np.array(centers))

            if i == 0:
                pdf = distance/np.sum(distance)
                centroid_new = X[np.random.choice(range(X.shape[0]),
                                                  replace=False,
                                                  p=pdf.flatten())]
            else:
                # Calculate the distance of each point to its nearest centroid
                dist_min = np.min(distance, axis=1)
                if pdf_method is True:
                    pdf = dist_min/np.sum(dist_min)
                    # Sample one point from the given distribution
                    centroid_new = X[np.random.choice(range(X.shape[0]),
                                                      replace=False,
                                                      p=pdf)]
                else:
                    index_max = np.argmax(dist_min, axis=0)
                    centroid_new = X[index_max, :]
            centers.append(centroid_new.tolist())

        return np.array(centers)

    # Uniform random initialization
    def random_coord_initialization(data, k):
        dim = np.array(data[1]).shape[0]
        centroids = [np.random.rand(dim) - 0.5 for _ in range(k)]
        return np.array(centroids)

    # Stop criteria for k-means update
    def should_stop(oldCentroids, centroids, iterations):
        if iterations > MAX_ITERATIONS:
            return True
        return (oldCentroids == centroids).all()

    # Returns array of indexes of closest centroid for each point
    def assign_centroids(data, centroids):
        distances = [0 for _ in range(k)]
        for ii in range(k):
            diff = data - centroids[ii]
            distances[ii] = metric(diff)
        distances = np.array(distances)
        closest_centroid = distances.argmin(axis=0)
        return closest_centroid

    # New centroid positions, their importance and error
    def update_centroids(data, weights, labels):
        neighborhood = np.array([labels == jj for jj in range(k)])
        neighborhood = neighborhood * weights
        # Sets unnormalized cumulative centroid
        centroids = neighborhood.dot(data)

        importances = np.array([0 for _ in range(k)]).astype(float)
        error = 0
        for jj in range(k):
            s = sum(neighborhood[jj])
            if s != 0:
                centroids[jj] = centroids[jj] / s
                diff = data - centroids[jj]
                distances = metric(diff)
                error += distances.dot(neighborhood[jj])
                importances[jj] = s
        return np.array(centroids), importances, error

    if weights is None:
        weights = np.array([1 for _ in data])
    if metric == 'Euclidian':
        metric = euclidian_metric
    if method == 'uniform':
        def initialize_centroids(data, k):
            return random_coord_initialization(data, k)
    elif method == 'k++_max':
        def initialize_centroids(data, k):
            return kmeans_plus_plus_initialization(data, k, pdf_method=False)
    elif method == 'k++_pdf':
        def initialize_centroids(data, k):
            return kmeans_plus_plus_initialization(data, k, pdf_method=True)
    else:
        raise NotImplementedError()

    # Initialize clusters
    centroids = initialize_centroids(data, k)
    importances = [0 for _ in range(k)]
    error = 10**15

    # Initialize book keeping vars
    iterations = 0
    oldCentroids = None

    # Repeat until centroids converge to a local minima
    while not should_stop(oldCentroids, centroids, iterations):
        oldCentroids = centroids
        iterations += 1

        # Find the closest centroid for each point
        labels = assign_centroids(data, centroids)

        # Update centroid positions
        centroids, importances, error = update_centroids(data,
                                                         weights, labels)

    return centroids, importances, error
