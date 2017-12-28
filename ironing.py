#!python3

"""
A function that makes a numeric array monotonically-increasing.

@author Erel Segal-Halevi
@since  2017-10
"""

import numpy as np

def iron(x:list):
	"""
	"iron" the given array to make it weakly-increasing.
	Every value is increased to the maximum previous value.
	
	>>> x = [1,2,3,2]
	>>> iron(x)
	array([1, 2, 3, 3])
	>>> x
	[1, 2, 3, 2]
	>>> x = np.array([1,2,3,2])
	>>> iron(x)
	array([1, 2, 3, 3])
	>>> x
	array([1, 2, 3, 2])
	"""
	largestSoFar = -np.inf
	y = np.array(x)
	for i in range(len(x)):
		current = x[i]
		if current >= largestSoFar:
			largestSoFar = current
		else:
			y[i] = largestSoFar
	return y
	


if __name__ == "__main__":
	import doctest
	(failures,tests) = doctest.testmod(report=True)
	if not failures: print("{} tests passed!\n".format(tests))
