import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import os
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.preprocessing import MinMaxScaler


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

################################### Biexponential Transformation ###################################
"""
Biexponential scaling helps visualize data that is compressed against the low x- and y- axes. 
Squished data is easily viewed by adding a section of linear scale to log acquired data.

The general biexponential formula can be described as:

        


    where:



References:
Moore, Wayne A., and David R. Parks. "Update for the logicle data scale including operational code 
    implementations." Cytometry. Part A: the journal of the International Society for Analytical 
    Cytology 81.4 (2012): 273.

https://docs.flowjo.com/flowjo/graphs-and-gating/gw-transform-overview/gw-transform-procedure/

"""

####################################################################################################


###################################### Logicle Transformation ######################################
def _find_logicle_transform(T, W, M, A):
    # First, check parameter values to make certain they are valid
    parameter_limit_statement = "T > 0, M > 0, 0 <= W <= M/2"
    error_message = "Invalid input for parameter {}, please follow the convention: {}"
    if T <= 0:
        raise(InputError("Invalid input to logicle_transform() function", error_message.format("T", parameter_limit_statement)))
    if M <= 0:
        raise(InputError("Invalid input to logicle_transform() function", error_message.format("M", parameter_limit_statement)))
    if W < 0 or W > (M/2):
        raise(InputError("Invalid input to logicle_transform() function", error_message.format("W", parameter_limit_statement)))
    # Compute b and w
    b = (M + A) * np.log(10)
    w = W / (M + A)
    # Then, compute the points marking the lower end (x2), midpoint(x1), and upper end (x0) of the 
    # quasilinear region
    x2 = A / (M + A)
    x1 = x2 + w
    x0 = x2 + (2*w)
    # print("b is {}\nw is {}".format(b, w))
    # print("x2 (lower end) = {}\nx1 (midpoint) = {}\nx0 (upper end) = {}".format(x2, x1, x0))
    # Knowing b and w, we need to find the roots of the below equation to find d
    def find_d(d_input):
        return(w*(b + d_input) + 2*(np.log(d_input) - np.log(b)))
    d = fsolve(find_d, 1)[0]
    # print("d is {}".format(d))
    # The next step is to compute c, f, and a
    c_over_a = np.e**((b + d)*x0)
    f_over_a = -1 * ((np.e**(b*x1)) - (c_over_a * np.e**(-1 * d * x1)))
    a = T / ((np.e**(b)) - (c_over_a * np.e**(-1*d)) + (f_over_a))
    c = c_over_a * a
    f = f_over_a * a
    # print("c_over_a = {}\nf_over_a = {}\na = {}".format(c_over_a, f_over_a, a))
    # print("c = {}\nf = {}".format(c, f))
    # We create the modified bi-exponential function, which takes the result of the logicle transformation as input
    def mod_biex(logicle_output):
        return((a*np.e**(b * logicle_output)) - (c * np.e**(-1 * d * logicle_output)) + f)
    # Apply the modified biexponential function to find root and appply logicle transformation
    def find_logicle(raw_input_value):
        return(fsolve(lambda tmp_value: mod_biex(tmp_value) - raw_input_value, 1)[0])
    # Build a spline function to implement fast calculation of logicle normalized values
    # Use 1000 equally spaced values between an increased range to estimate
    logicle_domain = np.array(np.linspace(-1.1*T, 1.1*T, 1000))
    logicle_range = np.array([find_logicle(x) for x in logicle_domain])
    compute_logicle = InterpolatedUnivariateSpline(logicle_domain, logicle_range)
    compute_inverse_logicle = InterpolatedUnivariateSpline(logicle_range, logicle_domain)
    return({"compute_logicle": compute_logicle,
            "compute_inverse_logicle": compute_inverse_logicle})
    # # Return the more precise logicle  
    # return(np.array(list(map(find_logicle, np.atleast_1d(x)))))

def apply_logicle(x, T=3000000, W=0, M=3, A=1):
    """
    The logicle scale is the inverse of a modified biexponential function and has the same relation to 
    the modified biexponential function that a logarithmic scale has to its corresponding exponential 
    function. [1]

    The logicle uses the modified biexponential function B, according to:
        
        logicle(x, T, W, M, A) = root(B(y, T, W, M, A) - x)

    B is the modified biexponential function::

        B(y, T, W, M, A) = (ae^(by) - cd^(-dy)) - f

    where::
    
        w = W / (M + A)

        x2 = A / (M + A)

        x1 = x2 + w

        x0 = x2 + 2w

        b = (M + A) * ln(10)

    d is a constant so that::

        2(ln(d) - ln(b)) + w(d+b) = 0

    given b and w::

        ca = e^(x0(b+d))

        fa = (d^(b * x1)) - (ca / e^(d * x1))

        a = T / ((e^b) - f - (c / (e^d)))

        c = c * a

        f = f * a

    Arguments
    ---------
    x : list-like numeric vector
        The input vector to normalize with logicle transformation
    T : numeric
        The formal "Top of scale" value (Default=3000000)
    W : numeric
        (Width parameter) The number of decades in the approximately linear region 
        The choice of W = 0 gives essentially the hyperbolic sine function (sinh x)
    M : numeric
        The number of decades that the true logarithmic scale approached at the 
        high end of the logicle scale would cover in the plot range
    A : numeric
        Number of Additional decades of negative data values to be included

    Note
    ----
    Parameters should be chosen so that:
        - T > 0
        - M > 0
        - 0 <= W <= M/2

    Returns
    -------
    list-like numeric vector
        The input x after applying the logicle function

    References
    ----------
    .. [1] Moore, Wayne A., and David R. Parks. "Update for the logicle data 
        scale including operational code implementations," Cytometry. Part A: 
        the journal of the International Society for Analytical Cytology 81.4 
        (2012): 273.
    """
    logicle_obj = _find_logicle_transform(T, W, M, A)
    return(logicle_obj["compute_logicle"](x))

def apply_inverse_logicle(x, T=3000000, W=0, M=3, A=1):
    """This function inverts pyInfinityFlow.Transformations.apply_logicle

    Arguments
    ---------
    x : list-like numeric vector
        The input vector to invert the logicle transformation
    T : numeric
        The formal "Top of scale" value (Default=3000000)
    W : numeric
        (Width parameter) The number of decades in the approximately linear region 
        The choice of W = 0 gives essentially the hyperbolic sine function (sinh x)
    M : numeric
        The number of decades that the true logarithmic scale approached at the 
        high end of the logicle scale would cover in the plot range
    A : numeric
        Number of Additional decades of negative data values to be included

    Returns
    -------
    list-like numeric vector
        The input x after applying the inverse logicle function
    """
    logicle_obj = _find_logicle_transform(T, W, M, A)
    return(logicle_obj["compute_inverse_logicle"](x))



def scale_feature(input_array, min_threshold_percentile, max_threshold_percentile):
    """Removes outliers and applies MinMaxScaler

    This function is designed to remove outliers and fit the distribution into 
    the range (0,1)

    Arguments
    ---------
    input_array : list-like numeric vector
        The feature values to scale
    min_threshold_percentile : (number between 0 to 100 inclusive)
        The minimum value for the input domain to be accepted, outliers below \
        the percentile value given by this parameter will take on that minimum \
        value
    max_threshold_percentile : (number between 0 to 100 inclusive)
        The maximum value for the input domain to be accepted, outliers above \
        the percentile value given by this parameter will take on the maximum \
        value

    Returns
    -------
    list-like numeric vector
        The input_array after applying the thresholding and min-max scaling
    """
    input_array = np.array(list(input_array))
    min_threshold = np.percentile(input_array, min_threshold_percentile)
    max_threshold = np.percentile(input_array, max_threshold_percentile)
    input_array[input_array < min_threshold] = min_threshold
    input_array[input_array > max_threshold] = max_threshold
    return(MinMaxScaler().fit(input_array.reshape(-1,1)).transform(input_array.reshape(-1,1)).reshape(-1,))
