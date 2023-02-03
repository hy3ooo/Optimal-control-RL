#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys
import random
import math

# In[2]:


if sys.version_info.major == 2: # import canvas
    import Tkinter as tk
else:
    import tkinter as tk


# In[3]:


UNIT = 2   # pixels
STAGE = 40  # Number of stage
INVENTORY = 400  # N inventory
order_num = 40
p_num = 42 # production number in new stage
b_cost = 5 #backorder cost
p_cost = 2 #per 
Reset_day = 1
Reset_cost = 800
Sub_rate = 0.8        #0.6
Sub_cost = 3       #3
PM_cost = 300      #300
Det_lambda = 0.3
inv_cost = 0.2 #inventory cost


# In[ ]:


class PM(tk.Tk, object):
    def __init__(self):
        super(PM, self).__init__()
        self.action_space = ['P', 'PM','SUB','CM']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('pm')
        self.geometry('{0}x{1}'.format(INVENTORY * UNIT, STAGE * UNIT))
        self._build_pm()

    def _build_pm(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=STAGE * UNIT,
                           width=INVENTORY * UNIT)

        # create grids
        for c in range(0, INVENTORY * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, STAGE * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, STAGE * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, INVENTORY * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

#         create origin
        origin = np.array([1, 1])
        origin_center = origin 
        self.origin = self.canvas.create_rectangle(
            origin_center[0] - 1, origin_center[1] - 1,
            origin_center[0] + 1, origin_center[1] + 1,
            fill='cyan')


#         create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 1, origin[1] - 1,
            origin[0] + 1, origin[1] + 1,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([1, 1])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 1, origin[1] - 1,
            origin[0] + 1, origin[1] + 1,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]))/(STAGE*UNIT)

    def step(self, action):
#         time.sleep(0.1)
        random_rate = random.random()    
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        ac=0
         
        current_stage = int(self.canvas.coords(self.rect)[1])/2 + 1
        
        D_rate= 1-np.exp(- Det_lambda * (int(current_stage/5)+2)) 
        pr = float('%.2f'% (1-0.01*(3.26**(current_stage/10)-1)))
        print("""
        .....................
        """,'production rate:',pr)
        

        
        if action == 0:   # PM
            ac+=1
            print("action: preventive maintenance","current stage=",current_stage)
            if s[1] >= UNIT:
                if s[0] >= UNIT * 10:
                    base_action[1] -= UNIT
                    base_action[0] -= UNIT * order_num
                else :
                    base_action[1] -= UNIT
                    base_action[0] -= s[0]
            else:
                if s[0] >= UNIT * 10:
                    base_action[1] = base_action[1]
                    base_action[0] -= UNIT * order_num
                else :
                    base_action[1] = base_action[1]
                    base_action[0] -= s[0] 
                    
        elif action == 1:   # P 
            ac +=2
            print("action = producing","current stage=",current_stage)
            if s[1] < (STAGE - 1) * UNIT:
                if random_rate < D_rate: # deterioration rate
                    base_action[1] += UNIT
                    if s[0]/2 + p_num * pr - order_num >= 0:
                        base_action[0] += UNIT *(p_num * pr - order_num)
                    else: 
                        base_action[0] -= s[0]
                else:
                    base_action[1] = base_action[1]
                    if s[0]/2 + p_num * pr - order_num >= 0:
                        base_action[0] += UNIT *(p_num * pr - order_num)
                    else: 
                        base_action[0] -= s[0]          
#             else:
#                 if random_rate < 0.5:
#                     base_action[1] += base_action[1]
#                     if s[0]/2 + p_num * pr - order_num >= 0:
#                         base_action[0] += UNIT *(p_num * pr - order_num)
#                     else: 
#                         base_action[0] -= s[0]   
        elif action == 2: # SUBCONTRACT
            ac += 3
            print("action = subcontracting","current stage=",current_stage)
            if random_rate < D_rate: # 5
                base_action[1] += UNIT
                if s[0]/2 + p_num * pr - order_num*(1-Sub_rate) >= 0:
                    base_action[0] += UNIT *(p_num * pr - order_num*(1-Sub_rate))
                else: 
                    base_action[0] -= s[0]
            else:
                base_action[1] = base_action[1]
                if s[0]/2 + p_num * pr - order_num*(1-Sub_rate) >= 0:
                    base_action[0] += UNIT *(p_num * pr - order_num*(1-Sub_rate))
                else: 
                    base_action[0] -= s[0]
            
        elif action == 3: #CM
            ac+=4
            print("action = corrective maintenance","current stage=",current_stage)            
            base_action[1] -= UNIT * (current_stage - 1)
            if s[0]/2 - order_num * Reset_day >= 0:
                base_action[0] -= UNIT * order_num * Reset_day 
            else: 
                base_action[0] -= s[0]



        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state
        
        inventory_=next_coords[0]/2
        print('inventory=',inventory_)
        
        # reward function

        if ac ==1:   #pM          
            if (next_coords[0])/2 == 0:
                if s[0] >= UNIT * order_num: 
                    reward = -PM_cost- inv_cost*inventory_
                    done = False
                else:
                    reward = -((-s[0]/2 + order_num)* b_cost)-PM_cost - inv_cost*inventory_
                    done= False
            else:
                reward = -PM_cost - inv_cost*inventory_   
                done= False
        elif ac == 2:   #P
            if (next_coords[0])/2 == 0:
                if p_num * pr - order_num >= 0:
                    reward = -(order_num * p_cost)-inv_cost*inventory_
                    done = False
                else:
                    reward = -((p_num * pr) * p_cost + (-p_num * pr + order_num)* b_cost)-inv_cost*inventory_
                    done= False
            else:
                reward = -(order_num * p_cost)-inv_cost*inventory_    
                done= False 
        elif ac == 3:   #Sub
            if (next_coords[0])/2 == 0:
                if p_num * pr - order_num*(1-Sub_rate) >= 0:
                    reward = -(p_num * pr * p_cost + order_num*Sub_rate*Sub_cost)-inv_cost*inventory_
                    done = False
                else:
                    reward = -((p_num * pr) * p_cost + order_num*Sub_rate*Sub_cost + (-p_num * pr + order_num*(1-Sub_rate))* b_cost)-inv_cost*inventory_
                    done= False
            else:
                reward = -(p_num * pr * p_cost + order_num*Sub_rate*Sub_cost)-inv_cost*inventory_    
                done= False         
        elif ac == 4: #cm
            if s[0] >= UNIT * order_num * Reset_day:
                reward = -(Reset_cost)-inv_cost*inventory_
                done = False
            else:
                reward = -((-s[0]/2 + order_num * Reset_day)* b_cost + Reset_cost)-inv_cost*inventory_
                done=False
            
        s_ = (np.array(next_coords[:2]))/(STAGE*UNIT)
        
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()

