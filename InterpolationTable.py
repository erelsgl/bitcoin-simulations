#!python3

"""
A class for keeping a table of y-values vs. x-values and interpolating y-values for other x-values.

@author Erel
@since  2017-10
"""


import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import ironing
from log_progress import log_progress


class InterpolationTable:
	"""
	Contains a table of x-values and corresponding y-values.
	Can be used for interpolating y-values for other x-values.
	Can be saved/loaded to/from a file.
	NOTE: the class assumes that the function should be weakly-increasing.
	"""

	def __init__(self, xName:str, yName:str, fileName:str, valueCalculationFunction:Callable):
		self.xValues = None  # 1-dimensional array
		self.yValuesSamples = None  # 2-dimensional array. Every sample is a row; every x-value is a column.
		self.xName = xName
		self.yName = yName
		self.fileName = fileName
		self.valueCalculationFunction = valueCalculationFunction
		self.regressionFunction = None
		self.regressionString = ""

	def numOfSamples(self):
		return 0 if self.yValuesSamples is None else len(self.yValuesSamples)

	def calculateTable(self, xValues: list, numOfSamples:int = 1, recreateAllSamples:bool = False, numXValues:int = None, saveAfterEachSample:bool=False):
		"""
		Numerically calculate a table that gives, for each channel-capacity, its optimal reset-radius.
		This table is used for interpolation by self.getOptimalResetRadius.

		theFunction: gets an x-value and returns the corresponding y-value.
		numOfSamples: how many samples to run (the results will be averaged).
		recreateAllSamples: if True, all numOfSamples samples will be re-calculated. If False, only the missing will be re-calculated.
		"""
		if recreateAllSamples:
			self.yValuesSamples = np.zeros((0, len(xValues)))
			numOfExistingSamples = 0
		else:
			numOfExistingSamples = 0 if self.yValuesSamples is None else len(self.yValuesSamples)
			if numOfExistingSamples >= numOfSamples:
				return
		if numXValues is None:
			numXValues = len(xValues)
		self.xValues = xValues
		for iSample in range(numOfExistingSamples, numOfSamples):
			yValues = []
			for xValue in log_progress(xValues, every=1, name=self.xName, size=numXValues):
				yValue = self.valueCalculationFunction(xValue, iSample)
				yValues.append(yValue)
			yValuesArray = np.asarray(yValues)
			self.yValuesSamples = np.r_[self.yValuesSamples, [yValuesArray]] # add row; see https://stackoverflow.com/a/8505658/827927
			if saveAfterEachSample:
				self.saveTable()
		self.smoothTable()

	def calculateRegressionFunction(self, type:str):
		"""
		:param type: "linlin" or "linlin2" or "loglog"
		"""
		if self.xValues is None or self.yValuesSmoothed is None:
			raise Exception("run calculateTable first")
		xValues = self.xValues
		# yValues = self.yValuesSmoothed
		yValues = self.yValuesAverage
		# yValues = [item for sublist in self.yValuesSamples for item in sublist]  # flatten values in all samples
		if type=='linlin':
			regressionCoeffs = np.polyfit(xValues, yValues, 1)
			self.regressionString = "{} ~ {:.2e} {} + {:.2e}".format(self.yName, regressionCoeffs[0], self.xName, regressionCoeffs[1])
			self.regressionFunction = lambda x:  regressionCoeffs[0]*x + regressionCoeffs[1]
		elif type=='linlin2':
			regressionCoeffs = np.polyfit(xValues, yValues, 2)
			self.regressionString = "{} ~ {:.2e} {}^2 + {:.2e} {} + {:.2e}".format(self.yName, regressionCoeffs[0], self.xName, regressionCoeffs[1], self.xName, regressionCoeffs[2])
			self.regressionFunction = lambda x:  regressionCoeffs[0]*x**2 + regressionCoeffs[1]*x + regressionCoeffs[2]
		elif type=='loglog':
			regressionCoeffs = np.polyfit(np.log(xValues), np.log(yValues), 1)
			self.regressionString = "ln({}) ~ {:.2e} ln({}) + {:.2e}".format(self.yName, regressionCoeffs[0], self.xName, regressionCoeffs[1])
			self.regressionFunction = lambda x:  np.exp(regressionCoeffs[0]*np.log(x) + regressionCoeffs[1])
		else:
			raise Exception("type should be linlin or loglog")

	def smoothTable(self):
		if self.yValuesSamples is not None and len(self.yValuesSamples)>0:
			self.yValuesAverage = np.mean(self.yValuesSamples[:,:],axis=0)
			self.yValuesSmoothed = ironing.iron(self.yValuesAverage)

	def getYValue(self, xValue:float):
		if self.xValues is None or self.yValuesSmoothed is None:
			raise Exception("run calculateTable first")
		if self.regressionFunction is  None:
			return np.interp(xValue, self.xValues, self.yValuesSmoothed)
		else:
			return self.regressionFunction(xValue)

	def plotTable(self, xValues:list=None, numOfSamplesToShow:int=None):
		if self.xValues is None or self.yValuesSamples is None:
			raise Exception("run calculateTable first")
		if xValues is None:
			xValues = self.xValues
		if numOfSamplesToShow is None:
			numOfSamplesToShow = len(self.yValuesSamples)
		else:
			numOfSamplesToShow = min(numOfSamplesToShow, len(self.yValuesSamples))
		f, ax = plt.subplots(2, 1, sharex=True, figsize=(8,8))
		for i in range(0, numOfSamplesToShow):
			ax[0].plot(xValues, self.yValuesSamples[i], 'g--')
		ax[0].plot(xValues, self.yValuesAverage, 'b', label="Average of {} samples".format(len(self.yValuesSamples)))
		ax[0].set_ylabel("Exact "+self.yName)
		ax[0].legend(loc=0)
		ax[0].set_xlabel(self.xName)
		ax[1].plot(xValues,self.yValuesAverage,'b',label='Average')
		ax[1].plot(xValues,self.yValuesSmoothed,'r.',label='Monotone')
		ax[1].set_ylabel("Approximate "+self.yName)
		xValuesForInterpolation = np.concatenate( (xValues/2, xValues/2+xValues[-1]/2) )
		ax[1].plot(xValuesForInterpolation, [self.getYValue(x) for x in xValuesForInterpolation],'g',linewidth=3.0,label='Regression')
		ax[1].legend(loc=0)
		ax[1].set_xlabel(self.regressionString)
		return ax

	def plotTableLogLog(self):
		if self.xValues is None or self.yValuesSamples is None:
			raise Exception("run calculateTable first")
		f, ax = plt.subplots(3, 2, sharex='col', sharey=False, figsize=(12,12))
		for i in range(0,len(self.yValuesSamples)):
			ax[0][0].plot(self.xValues, self.yValuesSamples[i], 'g--')
		ax[0][0].plot(self.xValues, self.yValuesAverage, 'b', label="Average of {} samples".format(len(self.yValuesSamples)))
		ax[0][0].set_ylabel("Optimal "+self.yName)
		ax[0][0].legend(loc=0)
		ax[1][0].plot(self.xValues,self.yValuesSmoothed)
		ax[1][0].set_ylabel("Monotone "+self.yName)
		xValuesForInterpolation = np.concatenate( (self.xValues/2, self.xValues/2+self.xValues[-1]/2) )
		ax[2][0].plot(xValuesForInterpolation, [self.getYValue(x) for x in xValuesForInterpolation])
		ax[2][0].set_ylabel("Interpolated "+self.yName)
		ax[2][0].set_xlabel(self.xName)

		for i in range(0,len(self.yValuesSamples)):
			ax[0][1].loglog(self.xValues, self.yValuesSamples[i], 'g--')
		ax[0][1].loglog(self.xValues, self.yValuesAverage, 'b', label="Average of {} samples".format(len(self.yValuesSamples)))
		ax[0][1].set_ylabel("Optimal "+self.yName)
		ax[0][1].legend(loc=0)

		ax[1][1].loglog(self.xValues,self.yValuesSmoothed)
		loglogRegression = np.polyfit(np.log(self.xValues), np.log(self.yValuesSmoothed), 1)
		loglogRegressionString = "ln({}) = {:.2f} ln({}) + {:.2f}".format(self.yName, loglogRegression[0], self.xName, loglogRegression[1])
		ax[1][1].set_ylabel("Monotone "+self.yName)
		ax[1][1].set_xlabel(loglogRegressionString)

		xValuesForInterpolation = np.concatenate( (self.xValues/2, self.xValues/2+self.xValues[-1]/2) )
		ax[2][1].loglog(xValuesForInterpolation, [self.getYValue(x) for x in xValuesForInterpolation])
		ax[2][1].set_ylabel("Interpolated "+self.yName)
		ax[2][1].set_xlabel(self.xName)


	def saveTable(self):
		if self.xValues is None or self.yValuesSamples is None:
			print("WARNING: tables of "+self.yName+" vs "+self.xName+" not calculated yet")
			return
		np.savez(self.fileName,
			xValues=self.xValues,
			yValuesSamples=self.yValuesSamples,
			)

	def loadTable(self,regressionType=None):
		if not os.path.isfile(self.fileName):
			print("WARNING: file "+self.fileName+" does not exist")
			return
		data = np.load(self.fileName)
		arrays = {name:value for (name,value) in data.iteritems()}
		self.xValues = arrays.get('xValues')
		self.yValuesSamples = arrays.get('yValuesSamples')
		self.smoothTable()
		if regressionType:
			self.calculateRegressionFunction(type=regressionType)

	def setTable(self, xValues, yValuesSamples):
		self.xValues = xValues
		self.yValuesSamples = yValuesSamples
		self.smoothTable()

	def isTableCalculated(self):
		return self.xValues is not None and self.yValuesSamples is not None



if __name__ == "__main__":
	table = InterpolationTable(xName="x", yName="y", fileName="interpolation-tables/InterpolationTableDemo.npz",
		valueCalculationFunction=lambda x,iSample:np.sin(x)+np.random.random())
	filename="InterpolationTableDemo.npz"
	table.loadTable()
	print("current num of samples: ", table.numOfSamples())
	table.calculateTable(xValues=np.linspace(0,10,10000), numOfSamples=30, recreateAllSamples=False, saveAfterEachSample=True)
	table.plotTable(numOfSamplesToShow=100); plt.show()
	table.saveTable()
