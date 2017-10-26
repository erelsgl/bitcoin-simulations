#!python3

"""
A function to draw a random number according to the power-law distribution,
  with PDF: f(z) = minValue/z^2 for z>=zmin.

Note that it is different than numpy.random.power and scipy.random.powerlaw.

@author Erel Segal-Halevi
@since  2017-10
"""

import numpy as np

def random_powerlaw(minValue, size=1):
	"""
	Draws a random value according to the power-law distribution, with:
	* PDF: f(z) = minValue/z^2 for z>=zmin
	* CDF: F(z) = 1-minValue/z for z>=zmin
	Uses inverse random sampling: https://en.wikipedia.org/wiki/Inverse_transform_sampling
	
	minValue: smallest value from which the PDF is nonzero.
	size:     number of random values to return (in a numpy array)
	"""
	
	u = np.random.uniform(0,1,size)
	# The inverse CDF is: F^{-1}(u) = minValue/(1-u)
	return minValue/(1-u)
	
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	minValue = 0.001
	values=random_powerlaw(minValue=minValue,size=10000)
	plt.hist(values) # All except 20 samples are in [0,500]
	maxValue = max(values)
	xValues = np.linspace(minValue,maxValue,10000) 
	plt.plot(xValues, minValue/xValues**2)  # Plot visible only by zooming to [0, 0.1]
	plt.show()
	
