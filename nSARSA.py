import gym   # install this by "pip install gym"
import itertools
import matplotlib.style
import sys
import numpy as np
import plotting

matplotlib.style.use('ggplot')

if "../" not in sys.path:
  sys.path.append("../") 

from collections import defaultdict
from frozen_lake import FrozenLakeEnv

env = FrozenLakeEnv()


def make_epsilon_greedy_policy(Q, epsilon, num_actions):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.

    Returns a function that takes the state
    as an input and returns the probabilities
    for each action in the form of a numpy array
    of length of the action space(set of possible actions).
    """

    def policyFunction(state):
        Action_probabilities = np.ones(num_actions,
                                      dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


def n_step_SARSA(env, n, num_episodes, discount_factor=0.9,
              alpha=0.01, epsilon=0.1):
    """
    SARSA algorithm: on-policy TD control
    You may change alpha, epsilon, discount_factor for your purpose
    """

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    # Record the length and the reward of each episode
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))


    # For every episode
    for ith_episode in range(num_episodes):
        if ith_episode % 1000 == 1:
            print(ith_episode)

        # Reset the environment and pick the first action
        state = env.reset()

        ## [Your task1] ## Define policy using epsilon_greedy policy
        ####################################################################
        policy = make_epsilon_greedy_policy(Q, epsilon, n)
        ####################################################################

        # Choose action according to the probability distribution
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        ## [Your task2] ## Save state, action, reward in your way
        ####################################################################
        state_memory, action_memory, reward_memory = [state], [action], [0]
        ####################################################################
        
        # Define T
        T = float("inf")
        
        for t in itertools.count():
            if t < T:         
                # Take one step and store it
                # If you save memory in a different way, replace 'append' according to your way
                next_state, reward, done, _ = env.step(action)
                
                state_memory.append(next_state) 
                reward_memory.append(reward) 

                if done:  
                    T = t + 1
                else:
                    # Select and store an action A_{t+1}
                    next_action_probs = policy(next_state)
                    next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
    
                    action_memory.append(next_action)
                    
                # Update statistics
                stats.episode_rewards[ith_episode] += reward
                stats.episode_lengths[ith_episode] = t+1
            
            ## [Your task3] ## Complete the code (REF: Lecture Note 6: 7p)
            ####################################################################
            # Tau is the time whose estimate is being updated
            # (1) tau = /
            tau = t - n + 1

            if tau >= 0:
                # (2) Return G
                try:
                    G = np.sum([discount_factor ** (i - tau - 1) * reward_memory[i] for i in np.arange(tau, np.minimum(tau + n, int(T) + 1), dtype=int)])
                except OverflowError:
                    T = sys.maxsize
                    G = np.sum([discount_factor ** (i - tau - 1) * reward_memory[i] for i in np.arange(tau, np.minimum(tau + n, T + 1), dtype=int)])
                                
                # (3) If tau + n >= T, G_{t:t+n} = G_{t}
                if tau + n < T:
                    G = G + (discount_factor ** n) * Q[next_state][next_action]

                # (4) TD update
                td_ = G - Q[state_memory[tau]][action_memory[tau]]
                Q[state_memory[tau]][action_memory[tau]] += alpha * td_

            ##################################################################### 
            if tau == T-1 :
                break
            
            if t > 50:
                break

            # Update state and action
            state = next_state
            action = next_action
            
    return Q, stats


Q, stats = n_step_SARSA(env, n=4, num_episodes=20000)

plotting.plot_episode_stats(stats)
