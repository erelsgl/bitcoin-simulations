#!python3

"""
A wallet that uses MDP to make optimal (?) payment decisions.

Alice is the decision-maker. Each time, with probability p she has to send A coins to Bob.
Then her wallet has to decide between:
* Sending directly through the blockchain and paying the mining fee;
* Sending through the channel (if her balance is at least A) and paying 0;
* Pouring 1 or more coins into the channel (if her balance is less than A) and sending through the channel, paying the channel reset cost.

We assume that Bob never locks money - i.e, he always sends through the channel if possible, otherwise he sends through the blockchain.

The state of the channel is composed of two numbers: (capacity,balance) where 0<=balance<=capacity and capacity<numCapacities

Author: Erel Segal-Halevi
Since : 2017-07
"""

import mdptoolbox
import mdptoolbox.mdp as mdp
import doctest
import numpy as np

INFINITE_COST = 1e30       # used to denote illegal actions (np.inf does not work)

# The number of states depends on the maximum channel capacity - the maximum number of coins Alice holds initially.
# To keep the number of states at a reasonable level, not all capacities are allowed - 
#    the only allowed capacites are multiples of capacityMultiplier.
def setGlobalCapacity(newCapacityMultiplier:int, newNumCapacities:int):
    global capacityMultiplier, numCapacities, maxCapacity, numStates
    capacityMultiplier = newCapacityMultiplier
    numCapacities = newNumCapacities
    maxCapacity = (numCapacities-1)*capacityMultiplier+1 
    numStates = maxCapacity*numCapacities
    print("MDP Wallet has {} states".format(numStates))

setGlobalCapacity(newCapacityMultiplier = 2, newNumCapacities = 10)

### states:

def toState(capacity:int,balance:int)->int:
    """
    INPUT: capacity - total capacity of channel, in units of capacityMultiplier:   capacity<numCapacities
           balance  - balance of Alice in the channel: 0<=balance<=capacity*capacityMultiplier
    OUTPUT: a number representing the state of the MDP
    
    >>> setGlobalCapacity(newCapacityMultiplier=1, newNumCapacities=10)  # 0,10,...,90
    MDP Wallet has 100 states
    >>> toState(5,3)
    53
    >>> setGlobalCapacity(newCapacityMultiplier=10, newNumCapacities=11)  # 0,10,...,100
    MDP Wallet has 1111 states
    >>> toState(5,3)
    508
    """
    return capacity*maxCapacity + balance

def fromState(state:int)->(int,int):
    """
    INPUT:  a number representing the state of the MDP
    OUTPUT: capacity - total capacity of channel, in units of capacityMultiplier:   capacity<numCapacities
            balance  - balance of Alice in the channel: 0<=balance<=capacity*capacityMultiplier

    >>> setGlobalCapacity(newCapacityMultiplier=1, newNumCapacities=10)  # 0,10,...,90
    MDP Wallet has 100 states
    >>> fromState(53)
    (5, 3)
    >>> setGlobalCapacity(newCapacityMultiplier=10, newNumCapacities=11)  # 0,10,...,100
    MDP Wallet has 1111 states
    >>> fromState(508)
    (5, 3)
    """
    capacity = state // maxCapacity
    balance = state % maxCapacity
    return (capacity,balance)

def states():
    """
    Generates all possible states as triples: (capacity, balance, stateID)
    capacity is given in units of capacityMultiplier:   capacity<numCapacities
    balance is given in coins:    0<=balance<=capacity*capacityMultiplier
    """
    for capacity in range(numCapacities):
        for balance in range(capacity*capacityMultiplier+1):
            yield (capacity, balance, toState(capacity, balance))

####### 

def setTransitionAndReward(fromState,toState,curTransitions,transition,curRewards,reward):
    curTransitions[fromState,toState] = transition
    curRewards[fromState,toState] = reward

def addTransitionAndReward(fromState,toState,curTransitions,transition,curRewards,reward):
    curRewards[fromState,toState] = \
        curRewards[fromState,toState] * curTransitions[fromState,toState] + \
        reward * (1-curTransitions[fromState,toState])
    curTransitions[fromState,toState] = curTransitions[fromState,toState] + transition


def findPolicy(
    p = 0.1,    # probability that the next send is from Alice to Bob
    A = 3,      # amount Alice sends to Bob
    B = 1,      # amount Bob sends to Alicec
    txCost = 1, # Cost of a blockchain transaction
    txsPerReset = 1, # Number of blockchain transactions required for a channel reset
    interest = 0.001,
    discount = None
    ):
    
    if not discount: discount = 1 / (1+interest)
    rtCost = txCost*txsPerReset
    
    # STEP A: construct the matrices of transitions and rewards.
    # Each list below will have one matrix per action; each matrix is numStates x numStates.
    transitions = []
    rewards = []

    # ACTION 0: ALICE SENDS THROUGH BLOCKCHAIN
    curTransitions = np.identity(numStates)
    curRewards     = np.zeros((numStates,numStates))
    for (capacity,balance,state) in states():
        capacityCoins = capacity*capacityMultiplier
        channelCost = interest*capacityCoins
        # Case #1: Alice sends through the blockchain - channel does not change:
        setTransitionAndReward(state,state,                       curTransitions,p, curRewards,-txCost-channelCost)
        # Case #2: Bob sends:
        if capacityCoins-balance < B:    # Bob's balance is too low - will always use blockchain - channel unchanged
            addTransitionAndReward(state,state,                   curTransitions,1-p, curRewards,-channelCost)
        else:    # Bob will use the channel if he wants to send
            addTransitionAndReward(state,toState(capacity,balance+B), curTransitions,1-p, curRewards,-channelCost)
    #print(curTransitions.sum(axis=1))  # this should be all ones
    transitions.append(curTransitions)
    rewards.append(curRewards)

    # ACTION 1: ALICE SENDS THROUGH CHANNEL - NO RESET
    curTransitions = np.identity(numStates)
    curRewards     = np.zeros((numStates,numStates))
    for (capacity,balance,state) in states():
        capacityCoins = capacity*capacityMultiplier
        channelCost = interest*capacityCoins
        if balance < A:    # Alice's balance is too low - state unchanged and Alice goes to hell
            setTransitionAndReward(state,state,                       curTransitions,1.0, curRewards,-INFINITE_COST)
        else:    
            # Case #1:  Alice sends through channel:
            setTransitionAndReward(state,state,                        curTransitions,0, curRewards,-channelCost)
            setTransitionAndReward(state, toState(capacity,balance-A), curTransitions,p, curRewards,-channelCost)
            # Case #2: Bob sends:
            if capacityCoins-balance < B:    # Bob's balance is too low - will always use blockchain - channel unchanged
                addTransitionAndReward(state,state,                   curTransitions,1-p, curRewards,-channelCost)
            else:    # Bob will use the channel if he wants to send
                addTransitionAndReward(state,toState(capacity,balance+B), curTransitions,1-p, curRewards,-channelCost)
    #print(curTransitions.sum(axis=1))  # this should be all ones
    transitions.append(curTransitions)
    rewards.append(curRewards)
    
    # ACTIONS 2...numCapacities: ALICE SENDS THROUGH CHANNEL AFTER RESETTING IT TO "(action,0)"
    #                          NOTE: RESET OCCURS ONLY IF ALICE SENDS!
    for action in range(2,numCapacities):
        curTransitions = np.identity(numStates)
        curRewards     = np.zeros((numStates,numStates))
        for (capacity,balance,state) in states():
            capacityCoins = capacity*capacityMultiplier
            channelCost = interest*capacityCoins
            capacityAfterReset = action 
            balanceAfterReset = capacityAfterReset*capacityMultiplier
            if balanceAfterReset < A:    # Alice's balance after pouring is too low for sending - state unchanged and Alice goes to hell
                setTransitionAndReward(state,state, curTransitions,1.0, curRewards,-INFINITE_COST)
            elif capacityAfterReset >= numCapacities:   # Channel capacity after reset is too high - state unchanged and Alice goes to hell
                setTransitionAndReward(state,state, curTransitions,1.0, curRewards,-INFINITE_COST)
            else:     
                # Case #1: Alice resets and sends through the channel:
                setTransitionAndReward(state,state,                       curTransitions,0,   curRewards,-channelCost)
                setTransitionAndReward(state, toState(capacityAfterReset, balanceAfterReset-A), curTransitions,p, curRewards,-rtCost-channelCost)
                # Case #2: Bob sends:
                if capacityCoins-balance < B:    # Bob's balance is too low - will always use blockchain - channel unchanged
                    addTransitionAndReward(state,state,                   curTransitions,1-p, curRewards,-channelCost)
                else:    # Bob will use the channel if he wants to send
                    addTransitionAndReward(state,toState(capacity,balance+B), curTransitions,1-p, curRewards,-channelCost)
        #print(curTransitions.sum(axis=1))  # this should be all ones
        transitions.append(curTransitions)
        rewards.append(curRewards)

    print("transitions",len(transitions),"x",transitions[0].shape, "\n")
    print("rewards",len(rewards),"x",rewards[0].shape)
    
    
    # STEP B: solve the MDP:
    solver = mdp.ValueIteration(transitions, rewards, discount)
    solver.run()
    return (solver.policy,solver.V)


def actionToString(action):
    if action==0:
        return "blockchain"
    elif action==1:
        return "channel"
    else:
        return "reset "+str(action*capacityMultiplier)

def policyToHTML(policy,value):
    htmlHeading = (
        #"<h2>Alice's policy for next send [expected value]</h2>"
        "<h2>Alice's policy for next send</h2>\n<p>Discounted cost = {:0.2f}</p>\n".format(value[toState(0,0)])
        )
    htmlTable = "<table border='1' padding='1'>\n"
    
    htmlHeaderRow = " <tr><th>&nbsp;Alice's balance &rarr;<br/>Channel capacity &darr;</th>\n"
    for balance in range(maxCapacity+1):
        htmlHeaderRow += "  <td>"+str(balance)+"</td>\n"
    htmlHeaderRow += " </tr>\n"
    htmlTable += htmlHeaderRow
    
    for capacity in range(numCapacities):
        capacityCoins = capacity*capacityMultiplier
        htmlRow = " <tr><th>"+str(capacityCoins)+"</th>\n"
        for balance in range(capacityCoins+1):
            state = toState(capacity,balance)
            action = policy[state]
            #htmlRow += "  <td>{} [{:0.2f}]</td>\n".format(actionToString(action), value[state])
            htmlRow += "  <td>{}</td>\n".format(actionToString(action))
        htmlRow += " </tr>\n"
        htmlTable += htmlRow
        
    htmlTable += ' </table>'
    return htmlHeading + htmlTable
	


if __name__=="__main__":
    import doctest
    doctest.testmod()
    print("Doctest OK!")

    setGlobalCapacity(
        newNumCapacities = 10,      # Number of different capacities.
        newCapacityMultiplier = 2   # The multiplier of the capacities. E.g, with multiplier 10, the capacities are 0,10,20,...
        )
    
    (policy,value) = findPolicy(
        p = 0.1,    # probability that the next send is from Alice to Bob
        A = 4,     # amount Alice sends to Bob
        B = 1,      # amount Bob sends to Alicec
        txCost = 2, # Cost of a blockchain transaction
        txsPerReset = 1, # num of blockchain transactions required for channel reset
        interest = 0.001,   # interest rate - cost of locking money in a channel
        discount = 0.999    # discount factor for MDP calculations. 
        )
    
    print(policyToHTML(policy,value))
