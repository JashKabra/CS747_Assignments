import argparse
from re import X
import numpy as np
from pulp import *

def value_iteration(num_states, num_actions, gamma, transitions, rewards):
    V0=np.zeros(num_states,dtype=np.float64)
    V1=np.zeros(num_states,dtype=np.float64)
    pi=np.zeros(num_states,dtype=int)
    while True:
        V1=np.max(np.sum(transitions*(rewards+gamma*V0),axis=-1),axis=-1)
        if(np.all(np.abs(V1-V0)<1e-10)):
            break
        V0=V1
    pi=np.argmax(np.sum(transitions*(rewards+gamma*V0),axis=-1),axis=-1)
    return V1,pi    

def policy_evaluation(num_states, num_actions, gamma, transitions, rewards, pi):
    A=np.eye(num_states,dtype=np.float64)-gamma*transitions[np.arange(num_states),pi]
    b=np.sum(transitions*rewards,axis=-1,dtype=np.float64)[np.arange(num_states),pi]
    x=np.linalg.solve(A,b)  
    return x

def howard_policy_iteration(num_states, num_actions, gamma, transitions, rewards):
    V=np.zeros(num_states,dtype=np.float64)
    pi=np.zeros(num_states,dtype=int)
    while True:
        V=policy_evaluation(num_states, num_actions, gamma, transitions, rewards, pi) 
        IA=np.argmax(np.sum(transitions*(rewards+gamma*V),axis=-1),axis=-1) #taking argmax for each state like in value iteration (policy guaranteed to change for all states in IS)
        if(np.all(IA==pi)): 
            break
        pi=IA
    return V,pi

def linear_programming(num_states, num_actions, gamma, transitions, rewards):
    problem=LpProblem("Linear_Programming_MDP",LpMaximize)
    V=LpVariable.dicts("V",range(num_states),cat="Continuous")
    for s1 in range(num_states):
        for a in range(num_actions):
            problem+=V[s1]>=lpSum([transitions[s1,a,s2]*(rewards[s1,a,s2]+gamma*V[s2]) for s2 in range(num_states)])
    problem+=-lpSum(V[s] for s in range(num_states))
    problem.solve(PULP_CBC_CMD(msg=0))
    val_func=np.zeros(num_states,dtype=np.float64)
    for s in range(num_states):
        val_func[s]=V[s].value()
    pi=np.argmax(np.sum(transitions*(rewards+gamma*val_func),axis=-1),axis=-1)
    return val_func,pi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str, required=True, help='The path to the input MDP file')
    parser.add_argument('--algorithm', type=str, required=False, help='The algo to run. Valid values are: vi, hpi, and lp')
    parser.add_argument('--policy', type=str, required=False, help='The path to policy file')
    args = parser.parse_args()

    if args.algorithm is None:
        args.algorithm='vi'
    elif args.algorithm.lower() not in ['vi', 'hpi', 'lp']:
        print('Invalid algorithm')
        exit(1)
    
    if args.mdp is None:
        print('Please specify an MDP file')
        exit(1)
    
    with open(args.mdp, 'r') as f:
        lines=f.readlines()
        num_states=int(lines[0].split()[1])
        num_actions=int(lines[1].split()[1])
        end_states=list(map(int, lines[2].split()[1:]))
        gamma=float(lines[-1].split()[1])
        mdptype=lines[-2].split()[1]
        lines=lines[3:-2]
        transitions=np.zeros((num_states, num_actions, num_states),dtype=np.float64)
        rewards=np.zeros((num_states, num_actions, num_states),dtype=np.float64)
        for line in lines:
            s1=int(line.split()[1])
            a=int(line.split()[2])
            s2=int(line.split()[3])
            r=float(line.split()[4])
            p=float(line.split()[5])
            transitions[s1,a,s2]=p
            rewards[s1,a,s2]=r

    if args.policy is not None:
        with open(args.policy, 'r') as f:
            lines=f.readlines()
            pi=np.zeros(num_states,dtype=int)
            for i in range(num_states):
                pi[i]=int(lines[i].strip())
            V=policy_evaluation(num_states, num_actions, gamma, transitions, rewards, pi)

    else:        
        if args.algorithm.lower()=='vi':
            V,pi=value_iteration(num_states, num_actions, gamma, transitions, rewards)
        elif args.algorithm.lower()=='hpi':
            V,pi=howard_policy_iteration(num_states, num_actions, gamma, transitions, rewards)
        elif args.algorithm.lower()=='lp':
            V,pi=linear_programming(num_states, num_actions, gamma, transitions, rewards)

    for s in range(num_states):
        print(V[s], pi[s])    
    
    



