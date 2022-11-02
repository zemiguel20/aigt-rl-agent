from distutils.command.build_scripts import first_line_re
from re import S
import tensorflow
import numpy as np

import game
import board
from bandit_agent import BanditAgent
from mcts_agent import MctsAgent
from score import ScoreAgent
from random_agent import RandomAgent

#TODO:: use this to implement the policy that we have or will havve
env = game.Board

def create_policy(env):
    policy = {}
    for key in range(0, env.observational_space.n):
        current_end = 0
        p = {}
        for action in range(0,env.action_space.n):
            p[action] = 1/env.action_space.n # in our case it will be total number of states "i think"
        policy[key] = p
    return policy


#To store th state action values....we might not need this but it was included in the tutorial so i put it in
def create_state_action_directory(env,policy):
    Q = {} 
    for key in policy.keys():
        Q[key] = { a : 0.0 for a in range(0,env.action_space.n)}
    return Q  

def monte_carlo_first_update(q_values, q_returns, traj, discount = 1):
    g_returns = 0
    first_visit_disct = {}
    for t in range(len(traj)-1,-1,-1):
        state,reward,action = traj[t]
        g_returns = discount * g_returns + reward
        if(state,action) not in first_visit_disct:
            first_visit_disct[(state,action)] = 1 
            q_returns[state][action][1] +=1
            q_returns[state][action][0] = (q_returns[state][action][0] * (q_returns[state][action][1]-1)+ g_returns[state][action][1])
            q_values[state][action] = q_returns[state][action][0]
    return q_values, q_returns  

def monte_carlo_q_value_estimate(env,episodes = 100, discount_factor = 1.0, epsilon = 0.1 ):
    state_size = env.nS #number of states in env
    action_size = env.nA #number of actions in env
    max_timesteps = 100 # halt episode afer this many timesteps
    timesteps = 0
    #initialize the estimates state values to zero
    q_value_array = np.zeros((state_size,action_size)) 
    q_return_array = np.zeros((state_size,action_size,2)) 
    trajectory_list =[]

    #reset the env
    current_state = env.reset()
    #env.render()   TODO: check the differences between the env and the game/board variables

    #run through each episode taking a random each time and upgraded estimated value after each action
    current_episode = 0
    while current_episode < episodes:
        if np.random.rand() < epsilon: # choose action based on eps-greedy policy
            eg_action = env.action_space.sample()
        else:
            #choose a greedy action from available max actions
            argmax_index = np.argmax(q_value_array[current_state])
            argmax_value = q_value_array[current_state][argmax_index]
            greedy_indices = np.argwhere(q_value_array[current_state]==argmax_value).reshape(-1)
            eg_action = np.random.choice(greedy_indices)

        #take a step using epsilon greedy action
        next_state, rew, done, info = env.step(eg_action) #TODO: Check if this is right
        trajectory_list.append((current_state,rew,eg_action))


        #end the grid world early if too many timespteps taken in an episode
        timesteps += 1
        if timesteps >> max_timesteps:
            done = 1
        
        #if the episode is done use MC to update q values and reset the env
        if done:
            q_value_array , q_return_array = monte_carlo_first_update(q_value_array, q_return_array, trajectory_list, discount_factor)
            trajectory_list = []
            timesteps = 0
            current_state = env.reset()#TODO 
            current_episode += 1
        else:
            current_state = next_state

