import argparse
import numpy as np

def get_state(runs,balls,player,total_runs,total_balls):
    return balls*(total_runs+1)*2+runs*2+player

def get_rbp(total_runs,total_balls,state):
    p=state%2
    state=state//2
    r=state%(total_runs+1)
    state=state//(total_runs+1)
    b=state
    return r,b,p

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help='The path to the input states file')
    parser.add_argument('--parameters', type=str, required=True, help='The path to the player parameters file')
    parser.add_argument('--q', type=str, required=True, help='Value of q')
    args = parser.parse_args()

    if args.states is None or args.parameters is None or args.q is None:
        print("Enter valid arguments")
        exit(1)
    
    with open(args.states,'r') as f:
        lines=f.readlines()
        total_runs=int(lines[0].strip()[2:])
        total_balls=int(lines[0].strip()[:2])
    
    with open(args.parameters, 'r') as f:
        lines=f.readlines()
        prob_matrix=np.zeros((5,7),dtype=np.float64)
        lines=lines[1:]
        for i in range(5):
            line=lines[i].split()
            for j in range(7):
                prob_matrix[i,j]=float(line[j+1])
    q=float(args.q)

    num_states=(total_balls+1)*(total_runs+1)*2
    num_actions=6 #5 for P1, 1 for P2
    transitions=np.zeros((num_states, num_actions, num_states),dtype=np.float64)
    rewards=np.zeros((num_states, num_actions, num_states),dtype=np.float64)
    gamma=1
    runs_scored=[-1,0,1,2,3,4,6]
    runs_scored_2=[-1,0,1]

    for i in range(num_states):
        runs,balls,player=get_rbp(total_runs,total_balls,i)
        if runs==0 or balls==0:
            continue
        if player==0:
            for j in range(5):
                for k in range(7):
                    if k>0 and runs-runs_scored[k]>0:
                        runs_new=runs-runs_scored[k]
                        balls_new=balls-1
                        player_new=player
                        if balls_new%6==0:
                            player_new+=1
                        if runs_scored[k]%2==1:
                            player_new+=1
                        player_new=player_new%2
                        state_new=get_state(runs_new,balls_new,player_new,total_runs,total_balls)
                        transitions[i,j,state_new]+=prob_matrix[j,k]
                    elif k>0 and runs-runs_scored[k]<=0:
                        runs_new=0
                        balls_new=balls-1
                        player_new=player
                        state_new=get_state(runs_new,balls_new,player_new,total_runs,total_balls)
                        transitions[i,j,state_new]+=prob_matrix[j,k]
                        rewards[i,j,state_new]=1
                    elif k==0:
                        runs_new=1
                        balls_new=0
                        player_new=player
                        state_new=get_state(runs_new,balls_new,player_new,total_runs,total_balls)
                        transitions[i,j,state_new]+=prob_matrix[j,k]
        else: #player=1
            for j in range(3):
                if j>0:
                    runs_new=runs-runs_scored_2[j]
                    balls_new=balls-1
                    player_new=player
                    if balls_new%6==0:
                        player_new+=1
                    if runs_scored_2[j]%2==1:
                        player_new+=1
                    player_new=player_new%2
                    state_new=get_state(runs_new,balls_new,player_new,total_runs,total_balls)
                    transitions[i,5,state_new]+=(1-q)/2
                    if runs_new==0:
                        rewards[i,5,state_new]=1
                elif j==0:
                    runs_new=1
                    balls_new=0
                    player_new=player
                    state_new=get_state(runs_new,balls_new,player_new,total_runs,total_balls)
                    transitions[i,5,state_new]+=q
    
    print("numStates",num_states)
    print("numActions",num_actions)
    print("end",end=" ")
    for i in range(num_states):
        runs,balls,player=get_rbp(total_runs,total_balls,i)
        if runs==0 or balls==0:
            print(i,end=" ")
    print("")
    for i in range(num_states):
        for j in range(num_actions):
            for k in range(num_states):
                if transitions[i,j,k]!=0:
                    print("transition",i,j,k,rewards[i,j,k],transitions[i,j,k])
    print("mdptype episodic")
    print("discount",gamma)
    