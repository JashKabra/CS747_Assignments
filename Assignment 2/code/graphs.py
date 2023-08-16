import matplotlib.pyplot as plt
import numpy as np
import random,argparse,sys,subprocess,os

pol=np.zeros((16,31,2),dtype=int)
for i in range(16):
    for j in range(31):
        for k in range(2):
            if k%2==1:
                pol[i][j][k]=5

diction={'0':0,'1':1,'2':2,'4':3,'6':4}
with open('data/cricket/rand_pol.txt','r') as file:
    lines = file.readlines()
    for line in lines:
        rb=line.strip().split()[0]
        pols=diction[line.strip().split()[1]]
        runs=int(rb[2:])
        balls=int(rb[:2])
        pol[balls][runs][0]=pols

def get_rbp(total_runs,total_balls,state):
    p=state%2
    state=state//2
    r=state%(total_runs+1)
    state=state//(total_runs+1)
    b=state
    return r,b,p

def create_policy_file(runs,balls,pols):
    f=open('data/cricket/rand_pol_1.txt','w')
    num_states=(runs+1)*(balls+1)*2
    for i in range(num_states):
        r,b,p=get_rbp(runs,balls,i)
        #print(b,r,p,i)
        f.write(str(pols[b,r,p])+"\n")
    f.close()

def run(states, p1_parameter, q):
    cmd_encoder = "python","encoder.py","--parameters", p1_parameter, "--q", q, "--states",states
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","verify_attt_mdp"
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","verify_attt_planner","--states",states 
    print("\n","Generating the decoded policy file using decoder.py")
    f=open('verify_attt_decoder','a')
    subprocess.call(cmd_decoder,stdout=f)
    f.close()

    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')

def run2(states, p1_parameter, q,policyfile=""):
    cmd_encoder = "python","encoder.py","--parameters", p1_parameter, "--q", q, "--states",states
    print("\n","Generating the MDP encoding using encoder.py")
    f = open('verify_attt_mdp','w')
    subprocess.call(cmd_encoder,stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","verify_attt_mdp","--policy",policyfile
    print("\n","Generating the value policy file using planner.py using default algorithm")
    f = open('verify_attt_planner','w')
    subprocess.call(cmd_planner,stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","verify_attt_planner","--states",states 
    print("\n","Generating the decoded policy file using decoder.py")
    f=open('verify_attt_decoder','a')
    subprocess.call(cmd_decoder,stdout=f)
    f.close()

    os.remove('verify_attt_mdp')
    os.remove('verify_attt_planner')

def analysis_1():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    for i in range(11):
        run(states,parameters,str(i*0.1)) 
        
def analysis_1_2():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    policy_file='data/cricket/rand_pol_1.txt'
    for i in range(11):
        run2(states,parameters,str(i*0.1),policy_file)

def analysis_2():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    balls=10
    q=0.25
    policy_file='data/cricket/rand_pol_1.txt'
    for runs in range(20,0,-1):
        cmd_cric = "python","cricket_states.py","--runs", str(runs), "--balls", str(balls)
        print("\n","Generating the states")
        f = open(states,'w')
        subprocess.call(cmd_cric,stdout=f)
        f.close()
        create_policy_file(runs,balls,pol)
        run2(states,parameters,str(q),policy_file)

def analysis_2_2():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    balls=10
    q=0.25
    for runs in range(20,0,-1):
        cmd_cric = "python","cricket_states.py","--runs", str(runs), "--balls", str(balls)
        print("\n","Generating the states")
        f = open(states,'w')
        subprocess.call(cmd_cric,stdout=f)
        f.close()
        run(states,parameters,str(q)) 

def analysis_3():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    runs=10
    q=0.25
    policy_file='data/cricket/rand_pol_1.txt'
    for balls in range(15,0,-1):
        cmd_cric = "python","cricket_states.py","--runs", str(runs), "--balls", str(balls)
        print("\n","Generating the states")
        f = open(states,'w')
        subprocess.call(cmd_cric,stdout=f)
        f.close()
        create_policy_file(runs,balls,pol)
        run2(states,parameters,str(q),policy_file)

def analysis_3_2():
    states = 'data/cricket/cricket_state_list.txt'
    parameters='data/cricket/sample-p1.txt'
    runs=10
    q=0.25
    for balls in range(15,0,-1):
        cmd_cric = "python","cricket_states.py","--runs", str(runs), "--balls", str(balls)
        print("\n","Generating the states")
        f = open(states,'w')
        subprocess.call(cmd_cric,stdout=f)
        f.close()
        run(states,parameters,str(q)) 

analysis_3_2()
                

