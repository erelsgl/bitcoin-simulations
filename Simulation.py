#!python3

"""
Several classes for simulating a lightning channel.

@author Erel
@since  2017-10
"""


print("Simulation.py version 1.0")

import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from sympy import symbols

from InterpolationTable import InterpolationTable
from log_progress import log_progress
from powerlaw import random_powerlaw

np.random.seed(None)

resetSize,r,zmin,zmax,beta,D,L,Supply = symbols('a r  z_{\min} z_{\max} \\beta \\Delta \\ell \\tau', positive=True,finite=True,real=True)


class Simulation:
	"""
	Abstract class for simulation of a lightning channel.
	"""
	def __init__(self, params:dict, numOfDays:int, filenamePrefix:str):
		"""
		params: dictionary that matches parameter symbols to their numeric values.
		numOfDays: how many days to run the channel simulation.
		filenamePrefix: the prefix of the filenames of the interpolation tables.
		"""
		self.params = params
		self.numOfDays = numOfDays
		self.filenamePrefix = filenamePrefix
		self.optimalResetRadius = InterpolationTable(xName="channel capacity", yName="reset radius", fileName=filenamePrefix+"-optimalResetRadius.npz",
			valueCalculationFunction = None)
		self.optimalChannelCapacity = InterpolationTable(xName="blockchain fee", yName="channel capacity", fileName=filenamePrefix+"-optimalChannelCapacity.npz",
			valueCalculationFunction = None)
		self.equilibriumBlockchainFeeTables = {}

	def generateTransfers(self, numOfDays:int, generateTransferSize:Callable, probAliceToBob:float)->list:
		"""
		Generate a random list of transfer-sizes.
		Positive is from Alice to Bob; negative is from Bob to Alice.
		"""
		numOfTransfers = np.random.poisson(numOfDays*self.params[L])
		sizes      = generateTransferSize(numOfTransfers)
		probBobToAlice = 1-probAliceToBob
		directions = np.random.choice([1,-1], p=[probAliceToBob,probBobToAlice], size=numOfTransfers)
		return sizes*directions


	def simulateTransfersWithoutLightning(self, transfers:list, blockchainFee:float)->list:
		"""
		Simulate doing the given transfers when there are no lightning channels

		INPUTS:
		  transfers: list of transfer-sizes. Positive is from Alice to Bob; negative is from Bob to Alice.
		  blockchainFee: the fee for a blockchain transaction.

		OUTPUTS: (numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers, sumLightningTransfers, utilityFromTransfers):
		  numBlockchainHits: number of blockchain-records consumed, either for transfers or for resets.
		  numBlockchainTransfers: number of transfers done on the blockchain.
		  numLightningTransfers:  number of transfers done on the lightning channel.
		  sumBlockchainTransfers: total number of bitcoins transfered on the blockchain.
		  sumLightningTransfers:  total number of bitcoins transfered on the blockchain.
		  utilityFromTransfers:   net utility of the channel users, i.e:
		    total number of bitcoins transfered overall times beta (the value/size parameter),
		    minus
		    number of blockchain hits times the blockchain fee.
		    NOTE: the utility does not take into account the interest payments.
		"""
		relativeTransferValue = self.params[beta]
		numBlockchainHits = numBlockchainTransfers = numLightningTransfers = sumBlockchainTransfers = sumLightningTransfers =  utilityFromTransfers = 0
		for transferSize in transfers:  # transferSize is positive iff it is from Alice to Bob.
			absTransferSize = abs(transferSize)
			transferValue = absTransferSize*relativeTransferValue
			if transferValue < blockchainFee:
				# Don't make any transfer
				continue
			else:
				# Do a blockchain transfer
				numBlockchainHits += 1
				numBlockchainTransfers += 1
				sumBlockchainTransfers += absTransferSize
				utilityFromTransfers += transferValue - blockchainFee
		return (numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers, sumLightningTransfers, utilityFromTransfers)


	def simulateTransfers(self, transfers:list, channelCapacity:float, initialBalance:float, maxLowResetRange:float, minHighResetRange:float, blockchainFee:float, lightningFeePerCoin:float=0)->int:
		"""
		Simulate doing the given transfers in a lightning channel.

		INPUTS:
		  transfers: list of transfer-sizes. Positive is from Alice to Bob; negative is from Bob to Alice.
		  channelCapacity: number of bitcoins locked into the channel (=w).
		  initialBalance:  initial number of bitcoins in Alice's side.
		  maxLowResetRange, minHighResetRange:
		       determine when the channel is reset to its initial balance.
				The channel will reset if the new balance is below maxLowResetRange or above minHighResetRange.
		  resetSize: number of blockchain records used per reset transaction. Between 1 and  2.
		  blockchainFee: the fee for a blockchain transaction.
		  lightningFeePerCoin: the fee for transferring a coin in lightning (the lightning fee is this  umber times the transfer size).
		       CURRENTLY NOT USED

		OUTPUTS: (numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers, sumLightningTransfers, utilityFromTransfers):
		  numBlockchainHits: number of blockchain-records consumed, either for transfers or for resets.
		  numBlockchainTransfers: number of transfers done on the blockchain.
		  numLightningTransfers:  number of transfers done on the lightning channel.
		  sumBlockchainTransfers: total number of bitcoins transfered on the blockchain.
		  sumLightningTransfers:  total number of bitcoins transfered on the blockchain.
		  utilityFromTransfers:   net utility of the channel users, i.e:
		    total number of bitcoins transfered overall times beta (the value/size parameter),
		    minus
		    number of blockchain hits times the blockchain fee.
		    NOTE: the utility does not take into account the interest payments.
		"""
		relativeTransferValue = self.params[beta]
		recordsPerReset = self.params[resetSize]

		balance = initialBalance
		numBlockchainHits = numBlockchainTransfers = numLightningTransfers = sumBlockchainTransfers = sumLightningTransfers =  utilityFromTransfers = 0
		for transferSize in transfers:  # transferSize is positive if it is from Alice to Bob, negative if it is from  Bob to Alice.
			absTransferSize = abs(transferSize)
			transferValue = absTransferSize*relativeTransferValue
			newBalance = balance - transferSize
			if newBalance < 0 or newBalance > channelCapacity:
				if transferValue < blockchainFee:
					# Don't make any transfer
					continue
				else:
					# Do a blockchain transfer; do not change the balance.
					numBlockchainHits += 1
					numBlockchainTransfers += 1
					sumBlockchainTransfers += absTransferSize
					utilityFromTransfers += transferValue - blockchainFee
			elif newBalance <= maxLowResetRange or newBalance >= minHighResetRange:
				# Do a lightning transfer and reset to initial balance
				numBlockchainHits += recordsPerReset
				numLightningTransfers += 1
				sumLightningTransfers += absTransferSize
				utilityFromTransfers += transferValue - blockchainFee*recordsPerReset
				balance = initialBalance
			else:
				# Do a lightning transfer and do not reset
				balance = newBalance
				numLightningTransfers += 1
				sumLightningTransfers += absTransferSize
				utilityFromTransfers += transferValue
		return (numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers, sumLightningTransfers, utilityFromTransfers)



	def plotBlockchainHitsVsResetRadiuses(self, numOfDays:int, channelCapacity:float, resetRadiuses:list, blockchainFee:float,figsize=(8,12)):
		transfers = self.generateTransfers(numOfDays)
		numBlockchainHits = []
		numBlockchainTransfers = []
		numLightningTransfers = []
		utilityFromTransfers = []
		for resetRadius in log_progress(resetRadiuses, every=10, name="Radiuses"):
			simResults = self.simulateTransfers(transfers, channelCapacity, resetRadius, blockchainFee)
			numBlockchainHits.append(simResults[0])
			numBlockchainTransfers.append(simResults[1])
			numLightningTransfers.append(simResults[2])
			utilityFromTransfers.append(simResults[-1])
		f, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)
		ax[0].plot(resetRadiuses,numBlockchainHits)
		ax[0].set_ylabel("#Blockchain hits")
		ax[0].set_title("Channel with capacity {} simulated for {} days, fee={}".format(channelCapacity,numOfDays,blockchainFee))
		ax[1].plot(resetRadiuses,utilityFromTransfers)
		ax[1].set_ylabel("Utility from transfers")
		#ax[2].plot(resetRadiuses, numBlockchainTransfers, "r--", label="blockchain")
		#ax[2].plot(resetRadiuses, numLightningTransfers, "b.-", label="lightning")
		#ax[2].plot(resetRadiuses, np.array(numBlockchainTransfers) + np.array(numLightningTransfers), "k-", label="total")
		#ax[2].set_ylabel("#transfers")
		#ax[2].legend()
		ax[1].set_xlabel("Reset radius")

	def calculateOptimalResetRadius(self, transfers:list, channelCapacity:float, blockchainFee:float, optimizationBounds=None):
		if optimizationBounds is None:
			optimizationBounds = (0,channelCapacity/2)
		negativeUtilityFunction = lambda resetRadius: -self.simulateTransfers(transfers, channelCapacity, resetRadius[0], blockchainFee)[-1]
		opt = scipy.optimize.differential_evolution(negativeUtilityFunction, [optimizationBounds])
		if opt.success:
			return opt.x
		else:
			print("Optimization failed for channelCapacity={}:".format(channelCapacity), opt)
			return None

	def calculateOptimalResetRadiusTable(self, numOfDays:int, channelCapacities:list, blockchainFee:float, numOfSamples:int=1, recreateAllSamples:bool=False, optimizationBounds=None):
		""" 
		Numerically calculate a table that gives, for each channel-capacity, its optimal reset-radius.
		This table is used for interpolation by self.getOptimalResetRadius.
		
		numOfDays: how many days to simulate in a single sample.
		channelCapacities: an array of channel capacities for which the optimal radius is calculated.
		blockchainFee: the fee per blockchain record.
		numOfSamples: how many samples to run (the results will be averaged).
		recreateAllSamples: if True, all numOfSamples samples will be re-calculated. If False, only the missing will be re-calculated.
		"""
		transferss = [self.generateTransfers(numOfDays) for iSample in range(numOfSamples)]
		self.optimalResetRadius.valueCalculationFunction = \
			lambda channelCapacity,iSample:	self.calculateOptimalResetRadius(transferss[iSample], channelCapacity, blockchainFee, optimizationBounds)
		self.optimalResetRadius.calculateTable(channelCapacities, numOfSamples, recreateAllSamples, numXValues=len(channelCapacities), saveAfterEachSample=True)
		self.optimalResetRadius.calculateRegressionFunction(type="linlin")


	def plotOptimalResetRadiusTable(self):
		self.optimalResetRadius.plotTable()

	def getOptimalResetRadius(self, channelCapacity:float):
		return self.optimalResetRadius.getYValue(channelCapacity)

	def simulateTransfersWithOptimalResetRadius(self, transfers:list, channelCapacity:float, blockchainFee:float)->int:
		optimalResetRadius = self.getOptimalResetRadius(channelCapacity)
		return self.simulateTransfers(transfers, channelCapacity, optimalResetRadius, blockchainFee)

		
	####### COST OF CHANNEL MAINTENANCE ######	

	def economicCost(self, channelCapacity:float, numOfDays:int)->float:
		"""
		:return: the cost of interest-payments for locking channelCapacity coins in a channel.
		"""
		return channelCapacity * self.params[r] * numOfDays  # linear approximation - no compound interest
		# return channelCapacity * ( (1+self.params[r])**numOfDays - 1 )  # exact calculation - with compound interest

	def calculateCosts(self, numOfDays:int, transfers:list, channelCapacity:float, blockchainFee:float)->(float,float):
		"""
		Calculate the costs and the utility from maintaining a given channel for a given number of days, running the given transfers.
		
		Returns (blockchainCost,economicCost,utility)
		"""
		simulationResults = self.simulateTransfersWithOptimalResetRadius(transfers, channelCapacity, blockchainFee)
		numBlockchainHits = simulationResults[0]
		blockchainCost = numBlockchainHits*blockchainFee
		utilityFromTransfers = simulationResults[-1]  # value from transfers minus blockchain cost
		economicCost   = self.economicCost(channelCapacity,numOfDays)
		utility = utilityFromTransfers - economicCost
		return (blockchainCost,economicCost,utility)

	def plotCostsVsChannelCapacity(self, numOfDays:int, blockchainFee:float, channelCapacities:list, figsize=(8,8)):
		blockchainCosts = []
		economicCosts = []
		totalCosts = []
		utilities = []
		transfers = self.generateTransfers(numOfDays)
		for c in log_progress(channelCapacities, every=10, name="Capacities"):
			(blockchainCost, economicCost, utility) = self.calculateCosts(numOfDays,transfers, channelCapacity=c, blockchainFee=blockchainFee)
			blockchainCosts.append(blockchainCost)
			economicCosts.append(economicCost)
			totalCosts.append(blockchainCost+economicCost)
			utilities.append(utility)
		f, (ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=figsize)
		ax1.plot(channelCapacities,blockchainCosts,'g',label="blockchainCosts")
		ax1.plot(channelCapacities,economicCosts,'b',label="economicCosts")
		ax1.plot(channelCapacities,totalCosts,'r',label="totalCosts")
		ax1.set_ylabel("Cost [bitcoins]")
		ax1.legend()
		ax1.set_title("{} days, Blockchain fee = {}".format(numOfDays,blockchainFee))
		ax2.plot(channelCapacities,utilities,'g')
		ax2.set_ylabel("utility")
		ax2.set_xlabel("Channel capacity w [bitcoins]")


	def _calculateOptimalChannelCapacity(self, numOfDays:int, transfers:list, blockchainFee:float):
		negativeUtilityFunction = lambda channelCapacity: -self.calculateCosts(numOfDays, transfers, channelCapacity[0], blockchainFee)[-1]
		optimizationBounds = (self.optimalResetRadius.xValues[0], self.optimalResetRadius.xValues[-1])
		opt = scipy.optimize.differential_evolution(negativeUtilityFunction, [optimizationBounds])
		opt.x = opt.x[0]
		if opt.success:
			return opt.x
		else:
			print("Optimization failed for blockchainFee={}:".format(blockchainFee), opt)
			return None

	def calculateOptimalChannelCapacity(self, numOfDays:int, blockchainFee:float):
		transfers = self.generateTransfers(numOfDays)
		return self._calculateOptimalChannelCapacity(numOfDays, transfers, blockchainFee)
		
	def calculateOptimalChannelCapacityTable(self, numOfDays:int, blockchainFees:list, numOfSamples:int=1, recreateAllSamples:bool=False):
		"""
		Numerically calculate a table that gives, for each blockchain-fee, its optimal channel-capacity.
		This table is used for interpolation by self.getOptimalChannelCapacity.
		"""
		transferss = [self.generateTransfers(numOfDays) for iSample in range(numOfSamples)]
		self.optimalChannelCapacity.valueCalculationFunction = \
			lambda blockchainFee,iSample:	self._calculateOptimalChannelCapacity(numOfDays, transferss[iSample], blockchainFee)
		self.optimalChannelCapacity.calculateTable(blockchainFees, numOfSamples, recreateAllSamples, numXValues=len(blockchainFees), saveAfterEachSample=True)
		self.optimalChannelCapacity.calculateRegressionFunction(type="loglog")

	def plotOptimalChannelCapacityTable(self, xValues:list=None):
		self.optimalChannelCapacity.plotTable(xValues)

	def getOptimalChannelCapacity(self, blockchainFee:float):
		return self.optimalChannelCapacity.getYValue(blockchainFee)

	def simulateTransfersWithOptimalChannelCapacity(self, transfers:list, blockchainFee:float)->int:
		optimalChannelCapacity = self.getOptimalChannelCapacity(blockchainFee)
		return self.simulateTransfersWithOptimalResetRadius(transfers, optimalChannelCapacity, blockchainFee)

	def simulateTransfersWithOrWithoutLightning(self, transfers:list, blockchainFee:float, withLightning:bool=True)->int:
		if withLightning:
			return self.simulateTransfersWithOptimalChannelCapacity(transfers, blockchainFee)
		else:
			return self.simulateTransfersWithoutLightning(transfers, blockchainFee)

	def demandForBlockchainRecords(self, numOfDays:int, transfers:list, blockchainFee:float, withLightning:bool=True)->int:
		"""
		Calculate the number of blockchain records demanded when the given transfers are executed with the given blockchain fee.
		"""
		if withLightning:
			optimalChannelCapacity = self.getOptimalChannelCapacity(blockchainFee)
			(numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers,
			 sumLightningTransfers, utilityFromTransfers) =	self.simulateTransfersWithOptimalResetRadius(transfers, optimalChannelCapacity, blockchainFee)
			economicCost = self.economicCost(optimalChannelCapacity, numOfDays)
			utility = utilityFromTransfers - economicCost
			# print("utility={}".format(utility))
			if utility>0:
				return numBlockchainHits
			else:
				print("utility={} - lightning will not be used".format(utility))
		# If the utility is negative,
		#    then the lightning fee for EVERY transfer is smaller than its value
		#    (since both the lightning fee and the value are  proportional to the transfer size).
		#    Therefore, lightning will not be used at all.
		result = self.simulateTransfersWithoutLightning(transfers, blockchainFee)
		return result[0]

	###### DEMAND SUPPLY and EQUILIBRIUM PRICE ######

	def plotDailyDemandVsBlockchainFee(self, numOfDays:int, blockchainFees:list, withLightning:bool=True, figsize=(8,4)):
		transfers = self.generateTransfers(numOfDays)
		dailyDemands = []
		for blockchainFee in log_progress(blockchainFees, every=10, name="Fees"):
			dailyDemands.append(self.demandForBlockchainRecords(numOfDays,transfers,blockchainFee,withLightning) / numOfDays)
		f, ax = plt.subplots(1, 2, sharey=False, figsize=figsize)
		ax[0].plot(blockchainFees,dailyDemands)
		ax[0].set_title("Daily demand per pair averaged over {} days".format(numOfDays))
		ax[0].set_xlabel("Blockchain fee")
		ax[0].set_ylabel("Daily demand for blockchain records")
		ax[1].loglog(blockchainFees,dailyDemands)
		loglogRegression = np.polyfit(np.log(blockchainFees), np.log(dailyDemands), 1)
		loglogRegressionLatex = '\ln({}) \approx {:.2f} \ln({}) + {:.2f}'.format("Daily demand", loglogRegression[0], "Blockchain fee", loglogRegression[1])
		loglogRegressionString = 'ln({}) ~ {:.2f} ln({}) + {:.2f}'.format("Daily demand", loglogRegression[0], "Blockchain fee", loglogRegression[1])
		ax[1].set_xlabel(loglogRegressionString)


	def _findEquilibriumFee(self, numOfDays:int, transfers:list, numOfPairs:int, supply:int, withLightning:bool=True)->float:
		surplusFunction = lambda blockchainFee: \
			self.demandForBlockchainRecords(numOfDays,transfers,blockchainFee,withLightning)*numOfPairs/numOfDays - \
			supply
		minFee = self.optimalChannelCapacity.xValues[0]
		if surplusFunction(minFee) < 0:
			return minFee   # If the demand is less than the supply even for a price of 0, then the price will be 0

		maxFee = self.optimalChannelCapacity.xValues[-1]
		while surplusFunction(maxFee) > 0:
			maxFee = maxFee*2
		fee = scipy.optimize.brentq(surplusFunction, minFee, maxFee)
		return fee
	
	def findEquilibriumFee(self, numOfDays:int, numOfPairs:int, supply:int, withLightning:bool=True):
		transfers = self.generateTransfers(numOfDays)
		return self._findEquilibriumFee(numOfDays, transfers, numOfPairs, supply, withLightning)

	def _plotNetworkPerformanceVsNumOfUsers(self, transfers:list, numOfDays:int, numsOfUsers:list, supply:int, ax:list, withLightning:bool=True):
		"""
		:param numOfDays:    how many days to run the simulation.
		:param numsOfUsers:  a list of numbers of users; run the simulation for each num-of-users separately.
		:param supply:       number of records per day. Currently 288000.
		:param ax:           list of 6 plot-axes.
		"""
		if supply is None:
			supply = self.params[Supply]

		equilibirumFees, numBlockchainHitss, minersRevenues, numBlockchainTransferss, numLightningTransferss, sumBlockchainTransferss, sumLightningTransferss, netUtilities = ([] for i in range(8))
		for numOfUsers in log_progress(numsOfUsers, every=1, name="Num of users"):
			numOfPairs = numOfUsers / 2

			# The approximation below results in over-supply of records, which is impossible.
			# Equilibrium fee must be calcualted per-transfer, not on average!
			# equilibriumFee = self.getEquilibriumBlockchainFee(numOfUsers)

			equilibriumFee = self._findEquilibriumFee(numOfDays, transfers, numOfPairs, supply, withLightning)
			equilibirumFees.append(equilibriumFee)
			if withLightning:
				optimalChannelCapacity = self.getOptimalChannelCapacity(equilibriumFee)
				(numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers, sumLightningTransfers, utilityFromTransfers) = \
					self.simulateTransfersWithOptimalResetRadius(transfers, optimalChannelCapacity, equilibriumFee)
				netUtility = utilityFromTransfers - self.economicCost(optimalChannelCapacity, numOfDays)  # utility per user
			else:
				(numBlockchainHits, numBlockchainTransfers, numLightningTransfers, sumBlockchainTransfers,
				 sumLightningTransfers, utilityFromTransfers) = \
					self.simulateTransfersWithoutLightning(transfers, equilibriumFee)
				netUtility = utilityFromTransfers
			pairsPerDay = numOfPairs / numOfDays
			numBlockchainHitss.append(numBlockchainHits * pairsPerDay)
			minersRevenues.append(numBlockchainHits * pairsPerDay * equilibriumFee)
			numBlockchainTransferss.append(numBlockchainTransfers * pairsPerDay)
			numLightningTransferss.append(numLightningTransfers * pairsPerDay)
			sumBlockchainTransferss.append(sumBlockchainTransfers * pairsPerDay)
			sumLightningTransferss.append(sumLightningTransfers * pairsPerDay)
			netUtilities.append(netUtility / numOfDays)  # utility per user per day

		ax[0].plot(numsOfUsers, equilibirumFees)
		ax[0].set_ylabel("Equilibrium fee")
		ax[0].set_title("Network stats averaged on {} days with {} recs/day".format(numOfDays,supply))
		ax[1].plot(numsOfUsers, numBlockchainHitss)
		ax[1].set_ylabel("#Daily Blockchain hits")
		ax[2].plot(numsOfUsers, minersRevenues)
		ax[2].set_ylabel("Daily miners' revenue")
		ax[3].plot(numsOfUsers, numBlockchainTransferss, "r--", label="blockchain")
		ax[3].plot(numsOfUsers, numLightningTransferss, "b.-", label="lightning")
		ax[3].plot(numsOfUsers, np.array(numBlockchainTransferss) + np.array(numLightningTransferss), "k-", label="total")
		ax[3].legend()
		ax[3].set_ylabel("Daily transfer count")
		ax[4].plot(numsOfUsers, sumBlockchainTransferss, "r--", label="blockchain")
		ax[4].plot(numsOfUsers, sumLightningTransferss, "b.-", label="lightning")
		ax[4].plot(numsOfUsers, np.array(sumBlockchainTransferss) + np.array(sumLightningTransferss), "k-", label="total")
		ax[4].legend()
		ax[4].set_ylabel("Daily transfer volume")
		ax[5].plot(numsOfUsers, netUtilities)
		ax[5].set_ylabel("Net utility per user")
		ax[-1].set_xlabel("Num of users")

	def plotNetworkPerformanceVsNumOfUsers(self, numOfDays:int, numsOfUsers:list, supply:int=None,figsize=(6,15), withLightning:bool=True):
		transfers = self.generateTransfers(numOfDays)
		f, ax = plt.subplots(6, 1, sharex=True, figsize=figsize)
		self._plotNetworkPerformanceVsNumOfUsers(transfers, numOfDays, numsOfUsers, supply, ax, withLightning)

	def plotNetworkPerformanceVsNumOfUsers2(self, numOfDays:int, numsOfUsers:list, supplies:list, figsize=(6,15), withLightning:bool=True):
		transfers = self.generateTransfers(numOfDays)
		f, ax = plt.subplots(6, len(supplies), sharex=True, figsize=figsize)
		for i in range(len(supplies)):
			self._plotNetworkPerformanceVsNumOfUsers(transfers, numOfDays, numsOfUsers, supplies[i], ax[:,i], withLightning)


	### Create an interpolation table for the equilibrium blockchain fee. NOT USED

	def calculateEquilibriumBlockchainFeeTable(self, numOfDays: int, numsOfUsers: list, supply:int=None, withLightning:bool=True, numOfSamples: int = 1, recreateAllSamples: bool = False):
		"""
		Numerically calculate a table that gives, for each number of users in the system,
		its equilibrium blockchain-fee.

		:param supply -- the number of blockchain records produced each day.

		NOTE: a different table is calculated for each value of supply.
		"""
		if supply is None:
			supply = self.params[Supply]
		if not supply in self.equilibriumBlockchainFeeTables:
			self.equilibriumBlockchainFeeTables[supply] = InterpolationTable(xName="num of users", yName="equilibrium fee",
			fileName=self.filenamePrefix+"-equilibriumBlockchainFee-"+str(supply)+".npz",
			valueCalculationFunction = None)

		transferss = [self.generateTransfers(numOfDays) for iSample in range(numOfSamples)]
		self.equilibriumBlockchainFeeTables[supply].valueCalculationFunction = \
			lambda numOfUsers, iSample: self._findEquilibriumFee(numOfDays, transferss[iSample], numOfPairs = numOfUsers/2, supply=supply, withLightning=withLightning)
		self.equilibriumBlockchainFeeTables[supply].calculateTable(numsOfUsers, numOfSamples, recreateAllSamples, saveAfterEachSample=True)
		self.equilibriumBlockchainFeeTables[supply].calculateRegressionFunction(type='linlin2')

	def plotEquilibriumBlockchainFeeTable(self, supply):
		ax=self.equilibriumBlockchainFeeTables[supply].plotTable()
		ax[0].set_title("Equilibrium fee for supply {}".format(supply))

	def getEquilibriumBlockchainFee(self, numOfUsers: float):
		return self.equilibriumBlockchainFee.getYValue(numOfUsers)

	def simulateTransfersWithEquilibiriumBlockchainFee(self, transfers: list, numOfUsers: int) -> int:
		equilibriumFee = self.getEquilibriumBlockchainFee(numOfUsers)
		return self.simulateTransfersWithOptimalChannelCapacity(transfers, equilibriumFee)



	###### SAVE AND LOAD THE INTERPOLATION TABLES ######
	
	def saveTables(self):
		if self.optimalResetRadius.isTableCalculated():
			self.optimalResetRadius.saveTable()
		if self.optimalChannelCapacity.isTableCalculated():
			self.optimalChannelCapacity.saveTable()
		supplyValues = np.array(list(self.equilibriumBlockchainFeeTables.keys()))
		np.savez(self.filenamePrefix+"-supplyValues.npz", supplyValues=supplyValues)
		for supply,table in self.equilibriumBlockchainFeeTables.items():
			if table.isTableCalculated():
				table.saveTable()

	def loadTables(self):
		self.optimalResetRadius.loadTable(regressionType="linlin")
		self.optimalChannelCapacity.loadTable(regressionType="loglog")
		try:
			supplyValues = np.load(self.filenamePrefix+"-supplyValues.npz")['supplyValues']
			for supply in supplyValues:
				self.equilibriumBlockchainFeeTables[supply] = InterpolationTable(xName="num of users",
					 yName="equilibrium fee",
					 fileName=self.filenamePrefix + "-equilibriumBlockchainFee-" + str(
						 supply) + ".npz",
					 valueCalculationFunction=None)
				self.equilibriumBlockchainFeeTables[supply].loadTable(regressionType="linlin2")
		except:
			print("WARNING: cannot read "+self.filenamePrefix+"-supplyValues.npz")
			print(sys.exc_info()[0])


class SymmetricSimulation(Simulation):
	"""
	Abstract class for simulating a lightning chjannel assuming transfer-rate is symmetric
	"""
	def simulateTransfers(self, transfers:list, channelCapacity:float, resetRadius:float, blockchainFee:float)->int:
		"""
		Simulate doing the given transfers in a lightning channel and return the number of blockchain hits.
		
		transfers: the list of transfer-sizes. Positive is Alice to Bob; negative is Bob tgo Alice.
		channelCapacity: number of bitcoins locked into the channel (=w).
		resetRadius:     a paramter that determines in what balance to reset the channel.
		blockchainFee:   price of a blockchain record. Deteremines what transfers will not take place.

		Returns the number of times the transfers hit the blockchain (either for a direct transfer or for a reset).
		"""
		return Simulation.simulateTransfers(self, transfers, channelCapacity, 
			initialBalance = channelCapacity/2,   # symmetric initialization
			maxLowResetRange = resetRadius,      # symmetric reset range
			minHighResetRange=channelCapacity-resetRadius,
			blockchainFee=blockchainFee,
			)
		
		
class AsymmetricSimulation(Simulation):
	"""
	Abstract class for simulating a lightning chjannel assuming transfer-rate is asymmetric.
	"""
	def simulateTransfers(self, transfers:list, channelCapacity:float, resetRadius:float, blockchainFee:float)->int:
		"""
		Simulate doing the given transfers in a lightning channel and return the number of blockchain hits.
		
		transfers: the list of transfer-sizes. Positive is Alice to Bob; negative is Bob tgo Alice.
		channelCapacity: number of bitcoins locked into the channel (=w).
		resetRadius:     a paramter that determines in what balance to reset the channel.

		Returns the number of times the transfers hit the blockchain (either for a direct transfer or for a reset).
		"""
		return Simulation.simulateTransfers(self, transfers, channelCapacity, 
			initialBalance = 0.99 * channelCapacity,   # Alice transfers more to Bob, so initially we put almost all funds at Alice's side.
			maxLowResetRange =2*resetRadius,           # We reset only when Alice's balance is low;
			minHighResetRange=np.inf,                 #    not when it is high.
			blockchainFee=blockchainFee,
			)
	
class UniformSymmetricSimulation(SymmetricSimulation):
	def generateTransfers(self, numOfDays:int)->list:
		"""
		Draw random transfers assuming Alice and Bob have the same transfer-rate (L/2),
		and the transfer-size is distributed uniformly in [0,zmax].
		"""
		return Simulation.generateTransfers(self,
			numOfDays, 
			generateTransferSize=lambda numOfTransfers: np.random.uniform(low=0,high=self.params[zmax],size=numOfTransfers),
			probAliceToBob = 0.5)

	def calculateOptimalResetRadius(self, transfers:list, channelCapacity:float, blockchainFee:float, optimizationBounds=None):
		if optimizationBounds is None:
			optimizationBounds = [0,self.params[zmax]]
		return SymmetricSimulation.calculateOptimalResetRadius(transfers, channelCapacity, blockchainFee, optimizationBounds=[0,self.params[zmax]])

	
class UniformAsymmetricSimulation(AsymmetricSimulation):
	def generateTransfers(self, numOfDays:int)->list:
		"""
		Draw random transfers assuming Alice and Bob have different transfer-rates ((L+D)/2, (L-D)/2),
		and the transfer-size is distributed uniformly in [0,zmax].
		"""
		return Simulation.generateTransfers(self,
			numOfDays, 
			generateTransferSize=lambda numOfTransfers: np.random.uniform(low=0,high=self.params[zmax],size=numOfTransfers),
			probAliceToBob = (self.params[L]+self.params[D])/2/self.params[L])

	def calculateOptimalResetRadius(self, transfers:list, channelCapacity:float, blockchainFee:float, optimizationBounds=None):
		if optimizationBounds is None:
			optimizationBounds = [0,self.params[zmax]]
		return AsymmetricSimulation.calculateOptimalResetRadius(transfers, channelCapacity, blockchainFee, optimizationBounds=optimizationBounds)

	
class PowerlawSymmetricSimulation(SymmetricSimulation):
	power = 2

	def setPower(self, power):
		if power<=1:
			raise Error("power should be more than 1")
		self.power = power

	def generateTransfers(self, numOfDays:int)->list:
		"""
		Draw random transfers assuming Alice and Bob have the same transfer-rate (L/2),
		and the transfer-size is distributed uniformly in [0,zmax].
		"""
		return Simulation.generateTransfers(self,
			numOfDays, 
			generateTransferSize=lambda numOfTransfers: random_powerlaw(minValue=0.5,size=numOfTransfers,power=self.power),
			probAliceToBob = 0.5)
	
class PowerlawAsymmetricSimulation(AsymmetricSimulation):
	power = 2
	def setPower(self, power):
		if power<=1:
			raise Error("power should be more than 1")
		self.power = power

	def generateTransfers(self, numOfDays:int)->list:
		"""
		Draw random transfers assuming Alice and Bob have different transfer-rates ((L+D)/2, (L-D)/2),
		and the transfer-size is distributed uniformly in [0,zmax].
		"""
		return Simulation.generateTransfers(self,
			numOfDays, 
			generateTransferSize=lambda numOfTransfers: random_powerlaw(minValue=0.5,size=numOfTransfers,power=self.power),
			probAliceToBob = (self.params[L]+self.params[D])/2/self.params[L])
	

if __name__ == "__main__":
	print("Start demo")
	params = {
		L: 10,  # total transfers per pair per day.
		D: 6,  # delta transfers per day (Alice-to-Bob minus Bob-to-Alice) in the asymmetric case.
		beta: 0.01,  # value / transfer-size
		r: 4 / 100 / 365,  # interest rate per day
		resetSize: 1.1,  # records per reset tx
		Supply: 288000,  # records per day
		zmin: 0.001,  # min transfer size (for power law distribution)
		zmax: 1,  # max transfer size (for uniform distribution)
	}
	#sim = UniformSymmetricSimulation(params, numOfDays=100, filenamePrefix="interpolation-tables/uniform-symmetric-100days")
	sim = PowerlawSymmetricSimulation(params, numOfDays=100, filenamePrefix="interpolation-tables/uniform-symmetric-100days")
	sim.power = 10
	supply = params[Supply]
	sim.loadTables()

	test = 1
	if test==1:
		sim.plotBlockchainHitsVsResetRadiuses(
			numOfDays=100,
			channelCapacity=20,
			resetRadiuses=np.linspace(-1, 3, 100),
			blockchainFee=0.1)
	elif test==2:
		sim.calculateOptimalResetRadiusTable(numOfDays=100, channelCapacities=np.linspace(1,50,50), blockchainFee=0, numOfSamples=5, recreateAllSamples=False)
		sim.saveTables()
		sim.plotOptimalResetRadiusTable(); plt.show()
	elif test==3:
		sim.plotCostsVsChannelCapacity(
			numOfDays=1000,
			blockchainFee=0.01,
			channelCapacities=np.linspace(1 * params[zmax], 50 * params[zmax], 1000))
	elif test==4:
		sim.calculateOptimalChannelCapacityTable(numOfDays=100, blockchainFees=np.linspace(0.001,0.1,50), numOfSamples=5, recreateAllSamples=False)
		sim.saveTables()
		sim.plotOptimalChannelCapacityTable(); plt.show()
	elif test==5:
		sim.plotDailyDemandVsBlockchainFee(numOfDays=100, blockchainFees=np.linspace(0.001,0.1,100))
	elif test==6:
		sim.calculateEquilibriumBlockchainFeeTable(numOfDays=100, numsOfUsers=np.linspace(100000,10000000,50), supply=supply, numOfSamples=4, recreateAllSamples=False)
		sim.saveTables()
		sim.plotEquilibriumBlockchainFeeTable(supply); plt.show()
		sim.calculateEquilibriumBlockchainFeeTable(numOfDays=100, numsOfUsers=np.linspace(100000,10000000,50), supply=supply*2, numOfSamples=4, recreateAllSamples=False)
		sim.saveTables()
		sim.plotEquilibriumBlockchainFeeTable(supply*2); plt.show()
	elif test==7:
		sim.plotNetworkPerformanceVsNumOfUsers2(numOfDays=100, numsOfUsers=np.linspace(100000,10000000,50), supplies=[supply,supply*2], figsize=(20,30))
		sim.plotNetworkPerformanceVsNumOfUsers2(numOfDays=100, numsOfUsers=np.linspace(100000,10000000,50), supplies=[supply,supply*2], figsize=(20,30), withLightning=False)
	plt.show()
	print("End demo")
