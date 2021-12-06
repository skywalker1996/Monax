# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/zhijian/workspace_fast/Monax")
import os
import time
import pickle
import numpy as np
import sklearn.datasets as dt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import math
import random
from utils.utils import generator_nonzero, cal_loss, evaluate_model, logistic, MSE, predict, check_accuracy
from lib.rate_control.ftrl_module.aggregation import Aggregation
# from model.RTT_prediction import get_reg_cls_result
# from model.models_RTT_prediction import LSTMTestNet
from lib.rate_control.ftrl_module.utility import utility_function
import torch
'''logistic regression model'''


class rate_control_module_FTRL:

	def __init__(self, dim, alpha, beta, lambda1, lambda2, initial_cwnd):
		"""
		the constructor of LR class
		:param dim: the dimension of input features
		:param alpha: the alpha parameters for learning rate in the update of weights
		:param beta: the beta parameters for learning rate in the update of weights
		:param lambda1: L1 regularization
		:param lambda2: L2 regularization
		"""

		self.CWND_MIN = 30
		self.CWND_MAX = 1000
		self.HISTORY_LEN = 100

		self.dim = dim
		self.alpha = alpha
		self.beta = beta
		self.lambda1 = lambda1
		self.lambda2 = lambda2

		# initialize the zis, nis, gradient, weights
		self.model_num = 8
		self.modelPool = []
		self.current_model = 0
		self.model_changing_flag = False

		for i in range(self.model_num):
			buf = {}
			buf['_zs'] = np.zeros(self.dim + 1)
			buf['_ns'] = np.zeros(self.dim + 1)
			buf['weights'] = np.zeros(self.dim + 1)
			self.modelPool.append(buf)

		self.utility_record = []
		self.cold_start = True  
		self.cwnd = initial_cwnd
		self.mid=15
		self.high=20

		self.RTT_threshold = 30

		self.history_prediction = []
		self.history_decision = []

		self.history_label = []

		self.increase_step = 10
		self.decrease_step = 10
		self.initial_step_size = 10
		self.increase_step_change_rate = 10
		self.decrease_step_change_rate = 10
		self.stepsize_max = 30

		self.intervention_prob = 0.0
		self.accuracy=1

		self.ensemble=True
		self.bagging=False
		self.RTT_prediction=False

		if (self.ensemble==True):
			self.used_model=0
			self.model_change=1
			self.modelAccuracy=[]
			self.bagging_prediction=[1]*self.model_num
			self.model_history_prediction=[[] for _ in range(self.model_num)]
			#self.model_history_label=[[0] * 100 for _ in range(self.model_num)]

		if (self.RTT_prediction==True):
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			self.model=LSTMTestNet(name='TSP_1',seq_len=10,forecast_len=2,input_size=1, hidden_size=16, num_layers=1).to(device)
			path='/home/hanying/monax/model/weights/weights.pth'
			pretrained_dict=torch.load(path)['state_dict']
			try:
				self.model.load_state_dict(pretrained_dict)
			except IOError:
				raise IOError("netG weights not found")


	def action(self, monitor_para):
		"""
		update the parameters: weights, zs, ns, gradients
		:param sample: the feature vector                -array vector
		:param label: the ground truth label             -value
		:param nonzero_index: the nonzero index list     -list
		"""
		self.current_monitor_para = monitor_para

		self.sending_rate = monitor_para['sending_rate'] 
		self.network_delay_gradient = monitor_para["network_delay"][-1] - monitor_para["network_delay"][-2]
		self.current_loss = monitor_para['packet_loss'][-1]
		self.max_network_delay = max(monitor_para['network_delay'][-3:])
		self.throughput_error = monitor_para['throughput_error']
		self.cwnd = monitor_para['cwnd']

		sample=[self.sending_rate,self.network_delay_gradient,monitor_para['network_delay'][-1],self.current_loss, self.throughput_error]

		current_utility = utility_function(sample,self.RTT_threshold,self.RTT_prediction)
		if(self.cold_start):

			self.history_label.append(0)
			if(self.ensemble):
				self.model_history_prediction[self.current_model].append(0)
			self.history_prediction.append(0)
			self.history_decision.append(0)
			self.utility_record.append(current_utility)#initial history_utility_record if necessary
			self.utility_record.append(current_utility)#initial history_utility_record if necessary
			self.cold_start = False
		else:
			label = self.history_decision[-1] if current_utility - self.utility_record[-1] > 0 else (1-self.history_decision[-1])
			if(len(self.history_label)>=self.HISTORY_LEN):
				del self.history_label[0]
			self.history_label.append(label)

			if(len(self.utility_record)>self.HISTORY_LEN):
				del self.utility_record[0]
			self.utility_record.append(current_utility)

		## determine self.current_model
		para_zs = self.modelPool[self.current_model]['_zs']
		para_ns = self.modelPool[self.current_model]['_ns']
		para_weights = self.modelPool[self.current_model]['weights']

		if np.abs(para_zs[-1]) > self.lambda1:
			fore = (self.beta + np.sqrt(para_ns[-1])) / self.alpha + self.lambda2
			para_weights[-1] = -1. / fore * (para_zs[-1] - np.sign(para_zs[-1]) * self.lambda1)
		else:
			para_weights[-1] = 0.0

		# update weights
		for index in generator_nonzero(sample):
			if np.abs(para_zs[index]) > self.lambda1:
				fore = (self.beta + np.sqrt(para_ns[index])) / self.alpha + self.lambda2
				para_weights[index] = -1. / fore * (para_zs[index] - np.sign(para_zs[index]) * self.lambda1)
			else:
				para_weights[index] = 0 


		base_grad=self.history_prediction[-1]-self.history_label[-1]

		# update the zs, ns of current model
		for j in generator_nonzero(sample):
			gradient = base_grad * sample[j]
			sigma = (np.sqrt(para_ns[j] + gradient ** 2) - np.sqrt(para_ns[j])) / self.alpha
			para_zs[j] += gradient - sigma * para_weights[j]
			para_ns[j] += gradient ** 2
		sigma = (np.sqrt(para_ns[-1] + base_grad ** 2) - np.sqrt(para_ns[-1])) / self.alpha
		para_zs[-1] += base_grad - sigma * para_weights[-1]
		para_ns[-1] += base_grad ** 2

		if(self.ensemble==True):
			for i in range(self.model_num):
				para_weight_buf = self.modelPool[i]['weights']		
				prediction, prob = predict(sample, para_weight_buf)

				if(len(self.model_history_prediction[i])>self.HISTORY_LEN):
					del self.model_history_prediction[i][0]
				self.model_history_prediction[i].append(prediction)

				if(i==self.current_model):
					if(len(self.history_prediction)>self.HISTORY_LEN):
						del self.history_prediction[0]
					self.history_prediction.append(prediction)
		else:
			prediction, prob = predict(sample, para_weights)
			if(len(self.history_prediction)>self.HISTORY_LEN):
				del self.history_prediction[0]
			self.history_prediction.append(prediction)



		para_zsm=[0]*self.model_num
		para_nsm=[0]*self.model_num
		para_weightsm=[0]*self.model_num
		for numbers in range(self.model_num):
			para_zsm[numbers] = self.modelPool[numbers]['_zs']
			para_nsm[numbers] = self.modelPool[numbers]['_ns']
			para_weightsm[numbers] = self.modelPool[numbers]['weights']

			if np.abs(para_zsm[numbers][-1]) > self.lambda1:
				fore = (self.beta + np.sqrt(para_nsm[numbers][-1])) / self.alpha + self.lambda2
				para_weightsm[numbers][-1] = -1. / fore * (para_zsm[numbers][-1] - np.sign(para_zsm[numbers][-1]) * self.lambda1)
			else:
				para_weightsm[numbers][-1] = 0.0        

		if(self.ensemble==True):
			#self.model_history_prediction[self.current_model]=self.model_history_prediction[self.current_model][1:]+[prediction]
			if (self.current_monitor_para['network_delay'][-4]>self.current_monitor_para['network_delay'][-3]\
			 and self.current_monitor_para['network_delay'][-3]>self.current_monitor_para['network_delay'][-2]\
			 and self.current_monitor_para['network_delay'][-2]>self.current_monitor_para['network_delay'][-1]):
				if(self.used_model<self.model_num-1):			
					self.modelAccuracy.append(check_accuracy(self.model_history_prediction[self.current_model],self.history_label))
					self.used_model+=1
					self.current_model+=1
# 					print(f"***** model num = {len(self.modelAccuracy)} and current model = {self.current_model}")
				else:
					if(len(self.modelAccuracy) < self.model_num):
						self.modelAccuracy.append(check_accuracy(self.model_history_prediction[self.current_model],self.history_label))
					for model_id in range(self.model_num):
						self.modelAccuracy[model_id]=check_accuracy(self.model_history_prediction[model_id],self.history_label)
						lead_model = np.argmax(self.modelAccuracy)
						if(self.current_model!=lead_model):	
							self.current_model=lead_model
							self.model_changing_flag = True
# 							print(f'switch to model # {lead_model}')
					#print(self.model_history_label[self.current_model])  #intervention times can also be used


		## Policy Aggregation
		aggregation=Aggregation(self.current_monitor_para,self.history_prediction,self.history_decision,\
								self.utility_record,self.RTT_threshold)
		decision, prob = aggregation.Policy_aggregation(prob)

		if(len(self.history_decision)>self.HISTORY_LEN):
			del self.history_decision[0]
		self.history_decision.append(decision)

		self.sending_rate = self.ratecontrol(prob)

# 		if (self.ensemble==True):
# 			print(f'current model is {self.current_model},each model accuracy is {self.modelAccuracy}')
		log_info = {}
		log_info['intervention_prob'] = self.intervention_prob
		if (self.bagging==True):
			print(f'The model accuracy is {self.bagging_weight,self.bagging_prediction}')
		return self.cwnd, current_utility, log_info


	def ratecontrol(self, res):
		network_delay_record = self.current_monitor_para['network_delay']
		current_network_delay = network_delay_record[-1]
#         print(f"**********current RTT = {current_network_delay}")
		if(res<0.5):
			##decrease sending_rate

			if(self.history_decision[-1]==0 and self.history_decision[-2]==0 and self.decrease_step<self.stepsize_max):
				self.decrease_step += self.decrease_step_change_rate
			else:
				self.decrease_step = int(self.decrease_step/2)

			if(self.cwnd-self.decrease_step>self.CWND_MIN):
				self.cwnd-=self.decrease_step
			else:
				self.cwnd += 3*self.decrease_step
				#print(f"decrease sending rate to {self.sending_rate} using decrease_step = {self.decrease_step}")

		elif(res>=0.5):
			##increase sending_rate
			if(self.history_decision[-1]==1 and self.history_decision[-2]==1 and self.increase_step<self.stepsize_max):
				self.increase_step += self.increase_step_change_rate
			else:
				self.increase_step = self.initial_step_size
			if(self.cwnd+self.increase_step<self.CWND_MAX):
				self.cwnd += self.increase_step
			else:
				self.cwnd -= 3*self.increase_step
			#print("increase sending rate to = ", self.sending_rate)
		return self.cwnd