from unicodedata import decimal
from unittest.mock import patch
import gym
from matplotlib.pyplot import axis
from numpy import argmax
import numpy as np
np.set_printoptions(precision=4)
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as st
torch.random.manual_seed(torch.e)
env = gym.make("CartPole-v0")

class QNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1=nn.Linear(4,16)
        self.l2=nn.Linear(16,32)
        self.l3=nn.Linear(32,2)

    def forward(self,x):
        x=torch.tensor(x)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=torch.sigmoid(self.l3(x))
        return x
    
    def sample_episode(self,):
       observation= env.reset().reshape(-1,4)
       complete=False
       observations=[]
       actions=[]
       rewards=[]
       while not complete:
           action=self.forward(torch.tensor(observation))
           action_prob=action.reshape(-1)/torch.sum(action.reshape(-1))
           action_pmf=st.rv_discrete(values=(np.array([0,1]),action_prob.detach().numpy()))
           action_sample=action_pmf.rvs()
           observation,reward,done,_=env.step(action_sample)
           observations.append(torch.tensor(observation.reshape(1,-1)))
           actions.append(action_sample)
           rewards.append(reward)
           complete=done
       env.reset()
       return (torch.cat(observations,dim=0),rewards,actions)

def calc_value(a,gamma):
    n = len(a)
    value=0
    for i in range(n):
        value+=gamma**i*a[i]
    return value 

net=QNET()
observation=torch.tensor(env.reset().reshape(-1,4))
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
epochs=100
num_episodes=10
batch_size=30
for _ in range(epochs):
    loss=0
    for b in range(batch_size):
        for num_pi in range(num_episodes):
            obs,rewards,actions=net.sample_episode()
            label=torch.zeros(len(actions),2)
            for i in range(len(actions)):
                label[i,actions[i]]=1
            out=net(obs)
            loss+=sum(rewards)*loss_function(out,label)
    loss=-loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"loss:{loss}")


torch.save(net.state_dict(),"policy")
with torch.no_grad():
    o=env.reset()
    env.render()
    finished = False
    while not finished:
        action=net(o)
        action_prob=action.reshape(-1)/torch.sum(action.reshape(-1))
        action_pmf=st.rv_discrete(values=(np.array([0,1]),action_prob.detach().numpy()))
        action_sample=action_pmf.rvs()
        o,r,done,_=env.step(action_sample)
        finished= done
    print("done")