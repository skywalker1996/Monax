# -*- coding: utf-8 -*-
import sys
sys.path.append("/home/zhijian/workspace_fast/Monax")
from configs.Common import *
import os
import time
import pickle
import numpy as np
import datetime as dt
from river import datasets
from river import linear_model
from river import compose
from river import preprocessing
from river import metrics
from river import neural_net as nn
from river import optim
from river import evaluate
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

class rate_control_module_Monax:

	def __init__(self, model_num, initial_cwnd):
		"""
		the constructor of LR class
		:param dim: the dimension of input features
		:param alpha: the alpha parameters for learning rate in the update of weights
		:param beta: the beta parameters for learning rate in the update of weights
		:param lambda1: L1 regularization
		:param lambda2: L2 regularization
		"""

		self.CWND_MIN = 100
		self.CWND_MAX = 500
		self.HISTORY_LEN = 100

		self.ACC_LEN = 10

		self.model_num = model_num
		self.model_pool = []
		self.current_model = 0
		self.model_changing_flag = False

		for i in range(self.model_num):
			model_dict = {}
			model_dict["metric"] = metrics.Accuracy()
			model_dict["optimizer"] = optim.FTRLProximal()
			model = compose.Pipeline(
						preprocessing.StandardScaler(),
						linear_model.LogisticRegression(model_dict["optimizer"])
					)
			model_dict["model"] = model
			self.model_pool.append(model_dict)

		self.utility_record = []
		self.cold_start = True
		self.cwnd = initial_cwnd
		self.mid = 15
		self.high = 20

		self.e2e_delay_threshold = 50
		self.queue_delay_threshold = 50

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
			self.modelAccuracy=[0.0 for i in range(model_num)]
			self.bagging_prediction=[1]*self.model_num
			self.model_history_prediction=[[] for _ in range(self.model_num)]
			#self.model_history_label=[[0] * 100 for _ in range(self.model_num)]

		# if (self.RTT_prediction==True):
		# 	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# 	self.model=LSTMTestNet(name='TSP_1',seq_len=10,forecast_len=2,input_size=1, hidden_size=16, num_layers=1).to(device)
		# 	path='/home/hanying/monax/model/weights/weights.pth'
		# 	pretrained_dict=torch.load(path)['state_dict']
		# 	try:
		# 		self.model.load_state_dict(pretrained_dict)
		# 	except IOError:
		# 		raise IOError("netG weights not found")


	def action(self, monitor_para):
		"""
		update the parameters: weights, zs, ns, gradients
		:param sample: the feature vector                -array vector
		:param label: the ground truth label             -value
		:param nonzero_index: the nonzero index list     -list
		"""
		start_time = time.time()


		self.current_monitor_para = monitor_para
		self.cwnd = monitor_para['cwnd']
		
		rtt_list = monitor_para[STATE_RTT]
		queue_delay_list = monitor_para[STATE_QUEUE_DELAY]
		e2e_delay_list = [rtt_list[i]+queue_delay_list[i] for i in range(-10,0)]
		ack_t_list = monitor_para[STATE_ACK_TIMESTAMP]
		thr_err_list = monitor_para[STATE_THROUGHPUT_ERROR]

		# self.LR_state = {
		# 	"001":(rtt_list[-3] - rtt_list[-4])/(ack_t_list[-3]-ack_t_list[-4]),
		# 	"002":(rtt_list[-2] - rtt_list[-3])/(ack_t_list[-2]-ack_t_list[-3]),
		# 	"003":(rtt_list[-1] - rtt_list[-2])/(ack_t_list[-1]-ack_t_list[-2]),
		# 	"004":rtt_list[-1]-self.RTT_threshold,
		# 	"005":thr_err_list[-1]
		# }
		self.LR_state = {
			"001":(rtt_list[-3] - rtt_list[-4]),
			"002":(rtt_list[-2] - rtt_list[-3]),
			"003":(rtt_list[-1] - rtt_list[-2]),
			"004":(queue_delay_list[-3] - queue_delay_list[-4]),
			"005":(queue_delay_list[-2] - queue_delay_list[-3]),
			"006":(queue_delay_list[-1] - queue_delay_list[-2]),
			"007":thr_err_list[-1]
		}

		self.state = {
			STATE_SENDING_RATE: monitor_para[STATE_SENDING_RATE],
			STATE_RTT: monitor_para[STATE_RTT],
			STATE_QUEUE_DELAY: monitor_para[STATE_QUEUE_DELAY],
			STATE_E2E_DELAY:e2e_delay_list, 
			STATE_LOSS: monitor_para[STATE_LOSS],
			STATE_THROUGHPUT_ERROR:monitor_para[STATE_THROUGHPUT_ERROR],
			STATE_ACK_TIMESTAMP: monitor_para[STATE_ACK_TIMESTAMP]
		}

		# print("Monax 001 cost time = ", (time.time()-start_time)*(10**6))

		current_utility = utility_function(self.state,self.e2e_delay_threshold)

		# print("Monax 002 cost time = ", (time.time()-start_time)*(10**6))
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
			label = self.history_decision[-1] if current_utility > self.utility_record[-1] else (1-self.history_decision[-1])
			
			pre_queue_delay_avg = sum(queue_delay_list[-4:-1])/3
			pre_rtt_delay_avg = sum(rtt_list[-4:-1])/3
			pre_e2e_delay_avg = sum(e2e_delay_list[-4:-1])/3
			# label = self.history_decision[-1] if monitor_para[STATE_RTT][-1]<pre_rtt_avg else 1- self.history_decision[-1]
		
			queue_delay_part = queue_delay_list[-1] - pre_queue_delay_avg
			rtt_part = rtt_list[-1]-pre_rtt_delay_avg
			# print(f">>>> queue delay part = {queue_delay_part} while rtt part = {rtt_part}")
			if queue_delay_part > rtt_part or queue_delay_list[-1]>rtt_list[-1] :
				label = 1
			else:
				label = 0

			# print(f"last decision is {self.history_decision[-1]} | rtt = {monitor_para[STATE_RTT][-1]}")
			# print(f"label = {label}")
			if(len(self.history_label)>=self.HISTORY_LEN):
				del self.history_label[0]
			self.history_label.append(label)

			if(len(self.utility_record)>self.HISTORY_LEN):
				del self.utility_record[0]
			self.utility_record.append(current_utility)
			self.model_train(self.previous_LR_state, self.history_prediction[-1], label, self.current_model)
		# print("Monax 0021 cost time = ", (time.time()-start_time)*(10**6))

		## determine self.current_model

		if(self.ensemble==True):
			if (self.current_monitor_para[STATE_RTT][-4]<self.current_monitor_para[STATE_RTT][-3]\
			and self.current_monitor_para[STATE_RTT][-3]<self.current_monitor_para[STATE_RTT][-2]\
			and self.current_monitor_para[STATE_RTT][-2]<self.current_monitor_para[STATE_RTT][-1]):
				for model_id in range(self.model_num):
					self.modelAccuracy[model_id]=check_accuracy(self.model_history_prediction[model_id],self.history_label, self.ACC_LEN)
					# print(f"model {model_id} recent acc: {self.modelAccuracy[model_id]}")
					lead_model = np.argmax(self.modelAccuracy)
					if(self.current_model!=lead_model):	
						self.current_model=lead_model
						self.model_changing_flag = True
						# print(f'################ switch to model # {lead_model}')
					#print(self.model_history_label[self.current_model])  #intervention times can also be used

		# print("Monax 003 cost time = ", (time.time()-start_time)*(10**6))

		if(self.ensemble==True):
			for i in range(self.model_num):
				prediction, prob = self.model_predict(self.LR_state, model_id=i)
				
				if(len(self.model_history_prediction[i])>self.HISTORY_LEN):
					self.model_history_prediction[i] = self.model_history_prediction[i][1:] + [prediction]
				else:
					self.model_history_prediction[i].append(prediction)
				if(i==self.current_model):
					if(len(self.history_prediction)>self.HISTORY_LEN):
						self.history_prediction = self.history_prediction[1:] + [prediction]
					else:
						self.history_prediction.append(prediction)
		else:
			prediction, prob = self.model_predict(self.LR_state, model_i=0)
			if(len(self.history_prediction)>self.HISTORY_LEN):
				self.history_prediction[0] = self.history_prediction[0][1:] + [prediction]
			else:
				self.history_prediction.append(prediction)

		# print("Monax 004 cost time = ", (time.time()-start_time)*(10**6))

	
		# print("Monax 005 cost time = ", (time.time()-start_time)*(10**6))

		## Policy Aggregation
		# aggregation=Aggregation(self.current_monitor_para,self.history_prediction,self.history_decision,\
		# 						self.utility_record,self.RTT_threshold)
		# decision, prob = aggregation.Policy_aggregation(prob)

		decision, prob = prediction, prob
		# print(f"decision: {decision} and its prob is {prob}")

		if(len(self.history_decision)>self.HISTORY_LEN):
			self.history_decision = self.history_decision[1:] + [decision]
		else:
			self.history_decision.append(decision)
		# print(f"the decision is {decision} and the prob = {prob}")
		self.cwnd = self.ratecontrol(decision, self.cwnd)
		# print(f"the cwnd is {self.cwnd}")

# 		if (self.ensemble==True):
# 			print(f'current model is {self.current_model},each model accuracy is {self.modelAccuracy}')
		log_info = {"test":"001"}

		# print("Monax 006 cost time = ", (time.time()-start_time)*(10**6))

		self.previous_LR_state = self.LR_state
		# log_info[""]
		return self.cwnd, current_utility, log_info


	def model_predict(self, state, model_id):
		model = self.model_pool[model_id]["model"]
		prediction = model.predict_one(state)
		proba = model.predict_proba_one(state)[prediction]
		return int(prediction), proba

	def model_train(self, state, pred, label, model_id):
		model = self.model_pool[model_id]["model"]
		self.model_pool[model_id]["metric"] = self.model_pool[model_id]["metric"].update(pred, label)
		model.learn_one(state, label)

	def ratecontrol(self, res, cwnd):
		rtt_record = self.current_monitor_para[STATE_RTT]
		current_rtt = rtt_record[-1]
#         print(f"**********current RTT = {current_rtt}")
		if(res==0):
			##decrease sending_rate

			if(self.history_decision[-1]==0 and self.history_decision[-2]==0 and self.decrease_step<self.stepsize_max):
				self.decrease_step += self.decrease_step_change_rate
			else:
				self.decrease_step = self.initial_step_size

			if(cwnd-self.decrease_step>self.CWND_MIN):
				cwnd-=self.decrease_step
			else:
				cwnd += 3*self.decrease_step
				#print(f"decrease sending rate to {self.sending_rate} using decrease_step = {self.decrease_step}")

		elif(res==1):
			##increase sending_rate
			if(self.history_decision[-1]==1 and self.history_decision[-2]==1 and self.increase_step<self.stepsize_max):
				self.increase_step += self.increase_step_change_rate
			else:
				self.increase_step = self.initial_step_size
			if(cwnd+self.increase_step<self.CWND_MAX):
				cwnd += self.increase_step
			else:
				cwnd -= 3*self.increase_step
			#print("increase sending rate to = ", self.sending_rate)

		return cwnd




