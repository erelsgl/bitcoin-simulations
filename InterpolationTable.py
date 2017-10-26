#!python3

"""
A class for keeping a table of y-values vs. x-values and interpolating y-values for other x-values.

@author Erel
@since  2017-10
"""


from typing import Callable
import sys, os, numpy as np, scipy, matplotlib.pyplot as plt
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
		self.yValuesSamples = None  # 2-dimensional array: samples x y-Values-per-sample
		self.xName = xName
		self.yName = yName
		self.fileName = fileName
		self.valueCalculationFunction = valueCalculationFunction

	def calculateTable(self, xValues: list, numOfSamples:int = 1, recreateAllSamples:bool = False, numXValues:int = None, saveAfterEachSample:bool=False):
		"""
		Numerically calculate a table that gives, for each channel-capacity, its optimal reset-radius.
		This table is used for interpolation by self.getOptimalResetRadius.

		theFunction: gets an x-value and returns the corresponding y-value.
		numOfSamples: how many samples to run (the results will be averaged).
		recreateAllSamples: if True, all numOfSamples samples will be re-calculated. If False, only the missing will be re-calculated.
		"""
		if recreateAllSamples or self.yValuesSamples is None:
			numOfExistingSamples = 0
		else:
			numOfExistingSamples = len(self.yValuesSamples)
		if numOfExistingSamples >= numOfSamples:
			return
		if numXValues is None:
			numXValues = len(xValues)
		yValuesSamples = np.zeros((numOfSamples, len(xValues)))
		if numOfExistingSamples > 0:
			yValuesSamples[0:numOfExistingSamples, :] = self.yValuesSamples
		if saveAfterEachSample:
			self.xValues = xValues
			self.yValuesSamples = yValuesSamples
		for iSample in range(numOfExistingSamples, numOfSamples):
			yValues = []
			for xValue in log_progress(xValues, every=1, name=self.xName, size=numXValues):
				yValue = self.valueCalculationFunction(xValue, iSample)
				yValues.append(yValue)
			yValuesSamples[iSample, :] = yValues
			if saveAfterEachSample:
				self.saveTable()
		self.yValuesSamples = yValuesSamples
		self.xValues = xValues
		self.smoothTable()

	def smoothTable(self):
		if self.yValuesSamples is not None and len(self.yValuesSamples)>0:
			self.yValuesAverage = np.mean(self.yValuesSamples[:,:],axis=0)
			self.yValuesSmoothed = ironing.iron(self.yValuesAverage)

	def getYValue(self, xValue:float):
		if self.xValues is None or self.yValuesSmoothed is None:
			raise Exception("run calculateTable first")
		return np.interp(xValue, self.xValues, self.yValuesSmoothed)

	def plotTable(self):
		if self.xValues is None or self.yValuesSamples is None:
			raise Exception("run calculateTable first")
		f, ax = plt.subplots(3, 1, sharex=True, figsize=(8,12))
		for i in range(0,len(self.yValuesSamples)):
			ax[0].plot(self.xValues, self.yValuesSamples[i], 'g--')
		ax[0].plot(self.xValues, self.yValuesAverage, 'b', label="Average of {} samples".format(len(self.yValuesSamples)))
		ax[0].set_ylabel("Optimal "+self.yName)
		ax[0].legend(loc=0)
		ax[1].plot(self.xValues,self.yValuesSmoothed)
		ax[1].set_ylabel("Monotone "+self.yName)
		xValuesForInterpolation = np.concatenate( (self.xValues/2, self.xValues/2+self.xValues[-1]/2) )
		ax[2].plot(xValuesForInterpolation, [self.getYValue(x) for x in xValuesForInterpolation])
		ax[2].set_ylabel("Interpolated "+self.yName)
		ax[-1].set_xlabel(self.xName)

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

	def loadTable(self):
		if not os.path.isfile(self.fileName):
			print("WARNING: file "+self.fileName+" does not exist")
			return
		data = np.load(self.fileName)
		arrays = {name:value for (name,value) in data.iteritems()}
		self.xValues = arrays.get('xValues')
		self.yValuesSamples = arrays.get('yValuesSamples')
		self.smoothTable()

	def setTable(self, xValues, yValuesSamples):
		self.xValues = xValues
		self.yValuesSamples = yValuesSamples
		self.smoothTable()

	def isTableCalculated(self):
		return self.xValues is not None and self.yValuesSamples is not None



if __name__ == "__main__":
	table = InterpolationTable(xName="x", yName="y", fileName="interpolation-tables/InterpolationTableDemo.npz",
		valueCalculationFunction=lambda x:np.sin(x)+np.random.random())
	filename="InterpolationTableDemo.npz"
	table.loadTable()
	table.calculateTable(xValues=np.linspace(0,10,10000), numOfSamples=10, recreateAllSamples=False)
	table.plotTable(); plt.show()
	table.saveTable()
