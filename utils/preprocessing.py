import numpy as np

def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    return x - np.mean(x, axis=1).reshape((x.shape[0], 1))
    
def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    return scale * x / np.sqrt(bias + np.mean(x ** 2, axis=1))[:,None]

def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Using training statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    mu = np.mean(x, axis=0)
    return x - mu, xtest - mu

def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Using training statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    covariance = np.dot(x.T, x) / x.shape[0]
    covariance += bias * np.eye(x.shape[1])
    U, S, _ = np.linalg.svd(covariance)
    pc = U @ np.diag(1. / np.sqrt(S)) @ U.T
    X = x @ pc
    Xtest = xtest @ pc
    return X, Xtest

def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW 
        NCHW is (samples, channels, height, width)
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    new_shape = (-1, 3, image_size, image_size)
    x, xtest = sample_zero_mean(x), sample_zero_mean(xtest)
    x, xtest = gcn(x), gcn(xtest)
    x, xtest = feature_zero_mean(x, xtest)
    # x, xtest = zca(x, xtest)
    x, xtest = x.reshape(new_shape), xtest.reshape(new_shape)
    return x, xtest