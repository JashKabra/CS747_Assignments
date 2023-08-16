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


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', type=str, required=True, help='The path to the value-policy file')
    parser.add_argument('--states', type=str, required=True, help='The path to the input states file')
    args = parser.parse_args()

    if args.value_policy is None or args.states is None:
        print("Enter valid arguments")
        exit(1)
    
    with open(args.states,'r') as f:
        lines=f.readlines()
        total_runs=int(lines[0].strip()[2:])
        total_balls=int(lines[0].strip()[:2])
        num_states=(total_balls+1)*(total_runs+1)*2

    with open(args.value_policy,'r') as f:
        lines=f.readlines()
        value_func=np.zeros((total_balls,total_runs),dtype=np.float64)
        policy=np.zeros((total_balls,total_runs),dtype=np.int64)
        for i in range(num_states):
            runs,balls,player=get_rbp(total_runs,total_balls,i)
            if runs==0 or balls==0 or player==1:
                continue
            value_func[balls-1,runs-1]=float(lines[i].split()[0])
            policy[balls-1,runs-1]=int(lines[i].split()[1])

    runs_targetted=[0,1,2,4,6]
    with open(args.states,'r') as f:
        lines=f.readlines()
        for line in lines:
            runs=int(line.strip()[2:])
            balls=int(line.strip()[:2])
            print(line.strip(),runs_targetted[policy[balls-1,runs-1]],value_func[balls-1,runs-1])
    