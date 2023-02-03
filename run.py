#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pm_env import PM
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from numpy import *


# In[2]:


def run_pm():
    rwd=[]
    stage_his=[]
    inv_level=[]
    s_check=[]
    stage_inv=[]
    stage_inv_sub=[]
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()
        t = 0
        reward_total = 0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)
            
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            
#             print("??????", observation ,action)
            stage = 1 + int(observation[1]* 40)  
            inven = int(observation[0]* 40)
            # swap observation
            observation = observation_
            
            
            reward_total += reward
            print('present reward',reward,'total',reward_total)
            
            stage_his.append(stage)
            inv_level.append(inven)
            s_check.append([stage,action])
            stage_inv.append([stage,inven])
            stage_inv_sub.append([stage,inven,action])
            t += 1
            # break while loop when end of this episode
            #             if done:
            if t > 99:
                rwd.append(reward_total)
                break                
            step += 1
    
    import json
    with open('dataN32Reward', 'w') as filehandle:
        filehandle.truncate()
        json.dump(rwd,filehandle)
    
    
    # end of game and plot
    print('game over')
    
    stage_his=stage_his[:5000]
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(stage_his)), stage_his,color='b')
    plt.ylabel('Stage',fontdict={'size':14})
    plt.xlabel('Every steps',fontdict={'size':14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    
    plt.plot(np.arange(len(rwd)), rwd)
    plt.ylabel('Cost_total')
    plt.xlabel('Every traning episode')
    plt.show()
    print(mean(rwd))
    
    plt.plot(np.arange(len(inv_level)), inv_level)
    plt.ylabel('Inventory level')
    plt.xlabel('Every steps ')
    plt.show()
    
    a=[]
    for x in range(1,41):
        for y in range(0,4):
            b= s_check.count([x,y])
            a.append([x,b])
    r=[]
    for x in range(40):
        tempsum=a[4*x][1] + a[4*x+1][1] + a[4*x+2][1] + a[4*x+3][1]
        r.append([x+1,a[4*x][1]/tempsum,a[4*x+1][1]/tempsum,a[4*x+2][1]/tempsum,a[4*x+3][1]/tempsum])

#     plt.title('Action to states of system') 
    df=pd.DataFrame(r,columns=["stage","PM","Producing","Outsourcing","CM"])
    df.plot(x="stage", y=["PM","Producing","Outsourcing","CM"], kind="bar",figsize=(12,6))
    plt.ylabel('proportion of action',fontdict={'size':16})
    plt.xlabel('stage',fontdict={'size':16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    b=stage_inv[-10000:]
    xt=[]
    for j in range(1,41):
        t=[]
        for i in range(len(b)):        
            if b[i][0] == j:
                t.append([b[i][1]])
                tm=mean(t)
        xt.append([tm])
    plt.plot((1+np.arange(40)),xt)
    plt.ylabel('Inventory level')
    plt.xlabel('States ')
    plt.show()
    
#     print(stage_inv_sub)
    stage_inv_sub=stage_inv_sub[-8000:]
    xx=[]
    yy=[]
    zz=[]
    for i in range(len(stage_inv_sub)):
        xx.append(stage_inv_sub[i][0])
        yy.append(stage_inv_sub[i][1])
        zz.append(stage_inv_sub[i][2])
    for i in range(len(zz)):
        if zz[i] == 2:
            zz[i]=1
        else:
            zz[i]=0
    print("Count for sub : ",zz.count(1)/8000) 
    fig = plt.figure(figsize=(12, 8),facecolor='lightyellow')  # 
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(xx,yy,zz,s=70,cmap='b',marker='.')    # 
    plt.xlabel('stage ',fontdict={'size':16})
    plt.ylabel('Inventory level',fontdict={'size':16})
    ax1.set_zlabel('outsourcing', fontsize=16, rotation=60)
    plt.show()
    
    xg=yg=np.arange(41)
    zgrid= np.full((len(xg), len(yg)), 0, dtype=int)
    for i in range(len(stage_inv_sub)):
        if stage_inv_sub[i][2]==2:
            if stage_inv_sub[i][1] < 41:
                zgrid[stage_inv_sub[i][1]][stage_inv_sub[i][0]] =1    
    xg,yg = np.meshgrid(xg,yg)
    fig = plt.figure(figsize=(12, 8),facecolor='lightyellow')   
    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(xg,yg,zgrid,cmap='plasma')    
    plt.xlabel('stage ',fontdict={'size':16})
    plt.ylabel('Inventory level',fontdict={'size':16})
    ax2.set_zlabel('outsourcing', fontsize=16, rotation=60)
    mytick=np.arange(0,41,5)
    plt.xticks(mytick)
    plt.show()
     
    
    
    env.destroy()



# In[3]:


if __name__ == "__main__":
    # maze game
    env = PM()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_pm)
    env.mainloop()
    RL.plot_cost()
    RL.plot_reward()


# In[4]:


inv_level


# In[ ]:




