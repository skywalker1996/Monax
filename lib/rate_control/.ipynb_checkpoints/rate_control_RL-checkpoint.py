import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from lib.rate_control.model_RC import *

class rate_control_module_RL(object):
	def __init__(self, initial_sending_bw):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.BATCH_SIZE = 32
		self.GAMMA = 0.999
		self.EPS_START = 0.9
		self.EPS_END = 0.05
		self.EPS_DECAY = 200
		self.TARGET_UPDATE = 10


		# Get number of actions from gym action space
		self.n_actions = 2
		self.seq_len = 10

		self.policy_net = DQN(self.seq_len, self.n_actions).to(self.device)
		self.target_net = DQN(self.seq_len, self.n_actions).to(self.device)
		self.target_net.load_state_dict(self.policy_net.state_dict())
		self.target_net.eval()

		self.optimizer = optim.RMSprop(self.policy_net.parameters())
		self.memory = ReplayMemory(10000)
		self.steps_done = 0
		

		self.episode_durations = []

		self.i_episode = 0

		self.last_state = None
		self.last_action = None

		## Rate Control 
		self.rec_window = 30

		self.cold_start = True

		self.network_delay_rec = [0,0]   #(min, max)
		self.packet_loss_rec = [0,0]
# 		self.send_delay_rec = [0,0]

		self.network_delay_g_rec = [0]*self.rec_window  # stroe previous gradients 
# 		self.send_delay_g_rec = [0]*self.rec_window  # stroe previous gradients  
		self.packet_loss_g_rec = [0]*self.rec_window


		self.network_delay_g2_rec = [0]*self.rec_window  # stroe previous gradients  
		self.packet_loss_g2_rec = [0]*self.rec_window

		self.utility_rec = [0]*self.rec_window

		state_list = ['step_1', 'step_2', 'action']
		self.state = 'step_1'

		self.ideal_delay = 100
		self.sending_bw_max = 40
		self.sending_bw_min = 5
		self.server_sending_bandwidth = initial_sending_bw
		self.initial_sending_bw = initial_sending_bw
		
		self.initial_step_size = 0.2
		
		self.step_size_increase = 0.2
		self.step_size_decrease = 0.2
		self.step_size_try = 0.2
		
		self.global_RTT_min = 0.0
		self.buffer_bloating = False
		self.bloating_count = 0
		
		self.direction_count = 0
# 		self.decrease_count = 0

		self.last_step = 0     # 0 for decrease / 1 for increase
		self.speedup_thres = 2
		
		self.utility_part_rec = [0,0]

	def compute_utility(self, monitor_params):
		
		# monitor_params = {'network_delay':[], 'packet_loss', 'send_queue_delay', 'server_sending_delay'}
		network_delay = monitor_params['network_delay']
		packet_loss = monitor_params["packet_loss"] 
# 		send_queue_delay = monitor_params["send_queue_delay"]
		# server_sending_delay = monitor_params["server_sending_delay"]

		
		network_delay_min = min(network_delay) 
		network_delay_max = max(network_delay)
# 		send_delay_min = min(send_queue_delay) 
# 		send_delay_max = max(send_queue_delay)
		packet_loss_min = min(packet_loss)
		packet_loss_max = max(packet_loss)

		if(network_delay[-1]>network_delay[-2] and network_delay[-2]>network_delay[-3] and network_delay[-3]>network_delay[-4]):
			self.buffer_bloating = True
		else:
			self.buffer_bloating = False
			
		if(self.cold_start):
			self.network_delay_rec = [network_delay_min, network_delay_max]
# 			self.send_delay_rec = [send_delay_min, send_delay_max]
			self.packet_loss_rec = [packet_loss_min, packet_loss_max]
			self.cold_start = False
			self.global_RTT_min = network_delay_min
			return 0.1, [0.1,0.1,0.1]
		else:
			if(network_delay_min<self.global_RTT_min):
				self.global_RTT_min = network_delay_min

		## network delay 
		network_delay_min_g = network_delay_min - self.network_delay_rec[0]
		network_delay_max_g = network_delay_max - self.network_delay_rec[1]

		network_delay_g = 0.7*network_delay_min_g/self.network_delay_rec[0] + 0.3*network_delay_max_g/self.network_delay_rec[1]

		self.network_delay_g_rec = self.network_delay_g_rec[1:] + [network_delay_g]

		## send delay
# 		send_delay_min_g = send_delay_min - self.send_delay_rec[0]
# 		send_delay_max_g = send_delay_max - self.send_delay_rec[1]

		# send_delay_g = 0.7*send_delay_min_g/self.send_delay_rec[0] + 0.3*send_delay_max_g/self.send_delay_rec[1]

		# self.send_delay_g_rec = self.send_delay_g_rec[1:] + [send_delay_g]


		packet_loss_min_g = packet_loss_min - self.packet_loss_rec[0]
		packet_loss_max_g = packet_loss_max - self.packet_loss_rec[1]
		if(self.packet_loss_rec[0] > 0.001 and self.packet_loss_rec[1] > 0.001):
			packet_loss_g = 0.7*packet_loss_min_g/self.packet_loss_rec[0] + 0.3*packet_loss_max_g/self.packet_loss_rec[1]
		else:
			packet_loss_g = 0.7*packet_loss_min_g + 0.3*packet_loss_max_g

		self.packet_loss_g_rec = self.packet_loss_g_rec[1:] + [packet_loss_g]


		self.network_delay_rec = [network_delay_min, network_delay_max]
		self.packet_loss_rec = [packet_loss_min, packet_loss_max]
# 		self.send_delay_rec = [send_delay_min, send_delay_max]

		self.network_delay_g2_rec = self.network_delay_g2_rec[1:] + [self.network_delay_g_rec[-1] - self.network_delay_g_rec[-2]]
		self.packet_loss_g2_rec = self.packet_loss_g2_rec[1:] + [self.packet_loss_g_rec[-1] - self.packet_loss_g_rec[-2]]

		send_rate_part = 1 * self.server_sending_bandwidth
		network_part = -1 * network_delay_g - 1 * packet_loss_g - (network_delay[-1]-self.ideal_delay)
# 		send_queue_part = -send_queue_delay[-1] if send_queue_delay[-1] > 30 else 0 
		
		#### Test Method
		if((network_delay[-3]+network_delay[-2]+network_delay[-1])/3 > 100):	
			utility =  -1.0
		else:
			utility = self.server_sending_bandwidth
			
		utility_part = [send_rate_part, network_part]

		self.utility_rec = self.utility_rec[1:] + [utility]
		# print('compute the utility = ', utility)
		return utility, utility_part

	def select_action(self,state):
		
	    sample = random.random()
	    eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
	        math.exp(-1. * self.steps_done / self.EPS_DECAY)
	    self.steps_done += 1
	    if sample > eps_threshold:
	        with torch.no_grad():
	            # t.max(1) will return largest column value of each row.
	            # second column on max result is index of where max element was
	            # found, so we pick action with the larger expected reward.
	            #print('input is ', state)
	            res = self.policy_net(state).cpu().numpy()
	            #print('res is ', np.argmax(res))
	            return torch.tensor(np.argmax(res)).view(1,1).cuda()
	    else:
	        return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

	def optimize_model(self):
	    if len(self.memory) < self.BATCH_SIZE:
	        return
	    transitions = self.memory.sample(self.BATCH_SIZE)
	    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
	    # detailed explanation). This converts batch-array of Transitions
	    # to Transition of batch-arrays.
	    batch = Transition(*zip(*transitions))

	    # Compute a mask of non-final states and concatenate the batch elements
	    # (a final state would've been the one after which simulation ended)
	    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
	                                          batch.next_state)), device=self.device, dtype=torch.bool)
	    non_final_next_states = torch.cat([s for s in batch.next_state
	                                                if s is not None])
	    #print('reward is ',batch.reward)

	    state_batch = torch.cat(batch.state)
	    action_batch = torch.cat(batch.action)
	    reward_batch = torch.cat(batch.reward)

	    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	    # columns of actions taken. These are the actions which would've been taken
	    # for each batch state according to policy_net

	    #The output of policy_net is (batch_size, n_actions)
	    state_action_values = self.policy_net(state_batch).gather(1,action_batch)
	    # print("state_action_values:",state_action_values.size())
	    # state_action_values: (batch_size, 1)

	    # Compute V(s_{t+1}) for all next states.
	    # Expected values of actions for non_final_next_states are computed based
	    # on the "older" target_net; selecting their best reward with max(1)[0].
	    # This is merged based on the mask, such that we'll have either the expected
	    # state value or 0 in case the state was final.
	    next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
	    # print("next_state_values:",next_state_values.size())
	    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1)[0].detach()
	    # print("next_state_values_new:",next_state_values.size())
	    # Compute the expected Q values
	    # print("reward_batch:",reward_batch.size())
	    expected_state_action_values = (next_state_values.unsqueeze(1) * self.GAMMA) + reward_batch
	    # print("expected_state_action_values:",expected_state_action_values.size())

	    # Compute Huber loss
	    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
	    # Optimize the model
	    self.optimizer.zero_grad()
	    loss.backward()
	    for param in self.policy_net.parameters():
	        param.grad.data.clamp_(-1, 1)
	    self.optimizer.step()

	def move(self, state, reward, done):

		if(self.last_state is None):
			self.last_state = state
			self.current_state = state

			action = self.select_action(self.current_state)
			self.last_state = self.current_state
			self.last_action = action
			return action
		else:
			self.current_state = state
			if(not done):
				self.current_state = self.current_state
			else:
				self.current_state = None
			
			simple_reward = 0.0
			if(self.buffer_bloating):
				if(self.last_action == 0):
					simple_reward = 1.0
				else:
					simple_reward = -1.0
			
				
			self.memory.push(self.last_state, self.last_action, self.current_state, reward)

			self.optimize_model()
			self.i_episode += 1

			# Update the target network, copying all weights and biases in DQN
			if self.i_episode % self.TARGET_UPDATE == 0:
				self.target_net.load_state_dict(self.policy_net.state_dict())

			if done:
				self.episode_durations.append(t + 1)
				# self.plot_durations()

			# print(self.current_state)
			action = self.select_action(self.current_state)

			self.last_state = self.current_state
			self.last_action = action

			return action

	def action(self, monitor_params):
		#increase sending rate
		state = monitor_params['network_delay'][-10:]
		utility,utility_part = self.compute_utility(monitor_params)
		done = False

		action = self.move(torch.tensor(state,dtype=torch.float).unsqueeze(0).cuda(), torch.tensor(utility, dtype=torch.float).view(1,1).cuda(),done)
		

# 		if(self.buffer_bloating):
# 			if(self.last_step == 0):
# 				self.direction_count += 1
# 				if(self.direction_count>=self.speedup_thres):
# 					self.step_size_decrease += self.initial_step_size
# 			else:
# 				self.direction_count = 0
# 				self.step_size_decrease = self.initial_step_size
				
# 			if(self.server_sending_bandwidth-self.step_size_decrease>=self.sending_bw_min):				
# 				self.server_sending_bandwidth -= self.step_size_decrease
				
# 			self.buffer_bloating = False
# 			self.last_step = 0

# 			return self.server_sending_bandwidth, utility
			
		#减小发送速率
		if(action == 0):
			if(self.server_sending_bandwidth<=self.sending_bw_min):
# 				self.server_sending_bandwidth = self.sending_bw_min
# 				self.server_sending_bandwidth += 3*self.step_size_increase
				self.server_sending_bandwidth = self.initial_sending_bw
# 					print('exceed min value, change server sending rate to {}!'.format(self.server_sending_bandwidth))
			else:
				if(self.last_step == 0):
					self.direction_count += 1
					if(self.direction_count>=self.speedup_thres):
						self.step_size_decrease += self.initial_step_size
				else:
					self.direction_count = 0
					self.step_size_decrease = self.initial_step_size
					
# 					self.decrease_count+=1
				self.server_sending_bandwidth -= self.step_size_decrease
				self.last_step = 0
# 					self.decrease_count+=1
# 					print('Decrease server sending rate to {}!'.format(self.server_sending_bandwidth))
		#增加发送速率
		elif(action == 1):
			if(self.server_sending_bandwidth>=self.sending_bw_max):
# 				self.server_sending_bandwidth = self.sending_bw_max
# 				self.server_sending_bandwidth -= 3*self.step_size_decrease
				self.server_sending_bandwidth = self.initial_sending_bw
# 					print('exceed max value, change server sending rate to {}!'.format(self.server_sending_bandwidth))
			else:
				if(self.last_step == 1):
					self.direction_count += 1
					if(self.direction_count>=self.speedup_thres):
						self.step_size_increase+=self.initial_step_size
						
				else:
					self.direction_count = 0
					self.step_size_increase = self.initial_step_size
				self.server_sending_bandwidth += self.step_size_increase
# 					print('Increase server sending rate to {}!'.format(self.server_sending_bandwidth))
				self.last_step = 1

		return self.server_sending_bandwidth, utility


# sending_bandwidth = 20
# RL_Module = rate_control_module_RL(sending_bandwidth)

# for i in range(100):
# 	network_delay_record = [1,2,3,4,5,6,7,8,9,10]
# 	packet_loss_record = [1,2,3,4,5,6,7,8,9,10]
# 	sending_bandwidth, utility = RL_Module.action({'network_delay': network_delay_record[-10:],
# 														  'packet_loss': packet_loss_record[-10:]})

# 	print(sending_bandwidth, utility)