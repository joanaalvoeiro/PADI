#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model
# 
# Consider once again the "Doom" domain, described in the Homework which you modeled using a Markov decision process. In this environment, 
# 
# * The agent can move in any of the four directions: up, down, left, and right. 
# * Movement across a grey cell division succeeds with a 0.8 probability and fails with a 0.2 probability. 
# * Movements across colored cell divisions (blue or red) succeed with a 0.8 probability (and fail with a probability 0.2) only if the agent has the corresponding colored key. Otherwise, they fail with probability 1. To get a colored key, the agent simply needs to stand in the corresponding cell.
# * When the movement fails, the agent remains in the same cell. 
# * The action that takes the agent through the exit always succeeds.  
# 
# In this lab you will interact with larger version of the same problem. You will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="maze.png" width="400px">
# 
# We consider that the agent is never in a cell $c\geq 17$ without a red key, and is never in a cell $c\geq28$ without a blue key. **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `X` that contains all the states in the MDP. There is a total of 73 states describing the possible positions of the agent in the environment and whether or not the agent has each of the two keys. Those states are represented as strings taking one of the forms `"N"`, indicating that the agent is in cell `N`, `"NR"`, indicating that the agent is in cell `N` with the red key, `"NRB"`, indicating that the agent is in cell `N` with both keys, or `"E"`, indicating that the agent has reached the exit.
# * An array `A` that contains all the actions in the MDP. Each action is represented as a string `"u"`, `"d"`, `"l"` or `"r"`.
# * An array `P` containing 4 $73\times 73$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., 3), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 4 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[16]:


import numpy as np
import numpy.random as rand
import time 


# In[17]:


# Add your code here.
def load_mdp(file, discountFactor):
    data = np.load(file)
    #['X', 'A', 'P', 'c'] - states, actions, transitions, costs
    return (data['X'],data['A'],data['P'],data['c'],discountFactor)


# We provide below an example of application of the function with the file `maze.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('maze.npz', 0.9)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0][s])
# 
# # Final state
# print('Final state:', M[0][-1])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[2][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[3][s, a])
# 
# # Discount
# print('Discount:', M[4])
# ```
# 
# Output:
# 
# ```
# Number of states: 73
# Random state: 9RB
# Final state: E
# Number of actions: 4
# Random action: u
# Transition probabilities for the selected state/action:
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0.]
# Cost for the selected state/action:
# 1.0
# Discount: 0.9
# ```

# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# Write a function `noisy_policy` that builds a noisy policy "around" a provided action. Your function should receive, as input, an MDP described as a tuple like that of **Activity 1**, an integer `a`, corresponding to the index of an action in the MDP, and a real number `eps`. The function should return, as output, a policy for the provided MDP that selects action with index `a` with a probability `1-eps` and, with probability `eps`, selects another action uniformly at random. The policy should be a `numpy` array with as many rows as states and as many columns as actions, where the element in position `[s,a]` should contain the probability of action `a` in state `s` according to the desired policy. 
# 
# ---

# In[18]:


# Add your code here.
def noisy_policy(mdp, a, eps):
    n_states = len(mdp[0])
    n_actions = len(mdp[1])
    
    action_prob = eps / (n_actions - 1)
    policy_row = np.zeros((1, n_actions))
    
    for i in range(n_actions):
        policy_row[0][i] = action_prob
    
    policy_row[0][a] = 1 - eps 
    
    policy = np.tile(policy_row, (n_states, 1))
    
    return policy


# We provide below an example of application of the function with MDP from the example in **Activity 1**, that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ. Note also that your function should work with **any** MDP that is specified as a tuple like the one from **Activity 1**.
# 
# ```python
# # Noiseless policy for action "Left" (action index: 2)
# pol_noiseless = noisy_policy(M, 2, 0.)
# 
# # Random state
# s = rand.randint(len(M[0]))
# 
# # Policy at selected state
# print('Random state:', M[0][s])
# print('Noiseless policy at selected state:', pol_noiseless[s, :])
# 
# # Noisy policy for action "Left" (action index: 2)
# pol_noisy = noisy_policy(M, 2, 0.1)
# 
# # Policy at selected state
# print('Noisy policy at selected state:', pol_noisy[s, :])
# 
# # Random policy for action "Left" (action index: 2)
# pol_random = noisy_policy(M, 2, 0.75)
# 
# # Policy at selected state
# print('Random policy at selected state:', pol_random[s, :])
# ```
# 
# Output:
# 
# ```
# Random state: 15
# Noiseless policy at selected state: [0. 0. 1. 0.]
# Noisy policy at selected state: [0.03 0.03 0.9  0.03]
# Random policy at selected state: [0.25 0.25 0.25 0.25]
# ```

# ---
# ---
# ---
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., 3), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 4 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.

# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# ---

# In[19]:


def get_expected_cost(pol, costF, nStates):
    expectedCost = np.zeros((nStates, 1))

    for r in range(nStates):
        actionCosts = costF[r]
        policy = pol[r]

        expectedCost[r] = np.dot(policy, np.transpose(actionCosts))
    
    return expectedCost


# In[20]:


def get_state_process(pol, costF, pMat, nActions, nStates):
    stateProcessAux = np.zeros((nActions, nStates, nStates))
    stateProcess = np.zeros((nStates, nStates))

    for r in range(nStates):
        actionCosts = costF[r]
        policy = pol[r]
        
        for a in range(nActions):
            stateProcessAux[a][r] = policy[a] * pMat[a][r]
    
    for a in range(nActions):
        stateProcess = np.add(stateProcess, stateProcessAux[a])
        
    return stateProcess


# In[21]:


# Add your code here.
def evaluate_pol(mdp, pol):
    nStates = len(mdp[0])
    nActions = len(mdp[1])
    pMat = mdp[2]
    costF = mdp[3]
    gamma = mdp[4]

    expectedCost = get_expected_cost(pol, costF, nStates)
    stateProcess = get_state_process(pol, costF, pMat, nActions, nStates)

    identity = np.eye(nStates, nStates)
    auxCalc = np.subtract(identity, (gamma * stateProcess))
    Jpi = np.dot(np.linalg.inv(auxCalc), expectedCost)
        
    return Jpi


# As an example, you can evaluate the random policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jpi = evaluate_pol(M, pol_noisy)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# ```
# 
# Output: 
# ```
# Cost to go at state 9RB: [10.]
# Cost to go at state 15: [10.]
# Cost to go at state 29RB: [9.6]
# ```

# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# ---

# In[22]:


def value_iteration(mdp):
    tInit = time.time()
    nStates = len(mdp[0])
    nActions = len(mdp[1])
    pMat = mdp[2]
    costF = mdp[3]
    gamma = mdp[4]

    J = np.zeros((nStates, 1))
    Q = np.zeros((nActions, nStates,1))
    err = 1
    i = 0
    
    while(err > 1e-8):
        for a in range(nActions):
            Ca = costF[:, a, None]
            Pa = pMat[a]
            Q[a] = Ca + (gamma * np.dot(pMat[a], J))
        Jnew = np.min(Q, axis=0)
        err = np.linalg.norm(Jnew - J)
        J = Jnew
        i += 1
    
    tEnd = time.time()
    tElapsed = tEnd - tInit

    print("Execution time: {0:.{1}f} seconds".format(tElapsed,3))
    print("N. iterations: {}".format(i))

    return J


# For example, the optimal cost-to-go for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# Jopt = value_iteration(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.006 seconds
# N. iterations: 73
# Cost to go at state 9RB: [6.82]
# Cost to go at state 15: [9.79]
# Cost to go at state 29RB: [1.]
# 
# Is the policy from Activity 2 optimal? False
# ```

# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Your function should print the time it takes to run before returning, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[23]:


def policy_iteration(mdp):
    nStates = len(mdp[0])
    nActions = len(mdp[1])
    pMat = mdp[2]
    costF = mdp[3]
    gamma = mdp[4]

    tInit = time.time()
    i = 0
    quitting = False
    pi = np.ones((nStates, nActions)) * 1/nActions

    while not quitting:
        J = evaluate_pol(mdp,pi)
        Q = []
        piNew = np.zeros((nStates, nActions))

        for aI in range(nActions):
            Q.append(costF[:, aI, None] + gamma * pMat[aI].dot(J))

        for aI in range(nActions):
            piNew[:, aI, None] = np.isclose(Q[aI], np.min(Q, axis=0), atol=1e-10,  rtol=1e-10).astype(int)

        piNew = piNew / np.sum(piNew, axis=1, keepdims=True)
        quitting = (pi==piNew).all()
        pi = piNew
        i += 1
    
    tEnd = time.time()
    
    print("Execution time: {0:.{1}f} seconds".format(tEnd-tInit,3))
    print("N. iterations: {}".format(i))

    return pi


# For example, the optimal policy for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# popt = policy_iteration(M)
# 
# rand.seed(42)
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# ```
# 
# Output:
# ```
# Execution time: 0.005 seconds
# N. iterations: 3
# Policy at state 9RB: l
# Policy at state 29RB: r
# Policy at state 8R: u
# ```

# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, `x0`, corresponding to a state index
# * A second integer, `length`
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **`NRUNS`** trajectories of `length` steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---

# In[24]:


NRUNS = 100

# Add your code here.
def simulate(mdp, pol, initState, length):
    states = mdp[0]
    actions = mdp[1]
    pMat = mdp[2]
    costF = mdp[3]
    gamma = mdp[4]

    empiricalCosts = []

    for runI in range(NRUNS):
        cost = 0
        currentState = initState

        for stepI in range(length):
            a = rand.choice(actions,p=pol[currentState])
            aI = list(actions).index(a)

            nextState = rand.choice(states,p=pMat[aI][currentState])
            nextStateI = list(states).index(nextState)

            cost += np.power(gamma,stepI)*costF[currentState][aI]
            currentState = nextStateI
        empiricalCosts.append(cost)
    
    return sum(empiricalCosts) / len(empiricalCosts)


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 100))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 1000))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s, 10000))
# ```
# 
# Output:
# ````
# Cost-to-go for state 9RB:
# 	Theoretical: [6.82]
# 	Empirical: 6.864862326263111
# Cost-to-go for state 4:
# 	Theoretical: [9.94]
# 	Empirical: 9.944922179980777
# Cost-to-go for state 5:
# 	Theoretical: [9.94]
# 	Empirical: 9.937132279574923
# ```
