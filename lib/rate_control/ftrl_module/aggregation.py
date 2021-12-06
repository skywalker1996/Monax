import random

class Aggregation(object):
	def __init__(self,current_monitor_para,history_prediction,history_decision,utility_record,RTT_threshold):
		self.current_monitor_para=current_monitor_para
		self.history_prediction=history_prediction
		self.history_decision=history_decision
		self.utility_record=utility_record
		self.RTT_threshold=RTT_threshold

	def Policy_aggregation(self, prob):

	#         print("input prob = ", prob)
		### PCC
		delta_utility = self.utility_record[-1] - self.utility_record[-2]
		self.intervention_prob = self.compute_intervention_prob(self.history_prediction,self.history_decision)

		if(self.current_monitor_para['network_delay'][-1] > self.RTT_threshold):
			aggregate_decision = 0
			#print("force slowing down with prob = ", prob if self.history_prediction[-1]==aggregate_decision else 1-prob)
			return aggregate_decision, prob if self.history_prediction[-1]==aggregate_decision else 1-prob

		if(delta_utility>0):
			### right direction
			return self.history_prediction[-1], prob
		elif(delta_utility<=0):
			### false direction
			aggregate_decision = 1-self.history_decision[-1]

#         div_threshold_scale = (self.RTT_threshold - (sum(self.current_monitor_para['network_delay'][-4:-1])/3))/self.RTT_threshold
#         if(div_threshold_scale>0):
#             ### speed up mode
#             aggregate_decision = 1 if random.random()<div_threshold_scale else aggregate_decision
#         elif(div_threshold_scale<0):
#             ### slow down mode
#             aggregate_decision = 0 if random.random()<(-1*div_threshold_scale) else aggregate_decision
#         else:
#             aggregate_decision = aggregate_decision


		prob = prob if self.history_prediction[-1]==aggregate_decision else 1-prob


		return aggregate_decision, prob

	def compute_intervention_prob(self,history_prediction,history_decision):
		intervention_count = 0
		rec_count = min(len(self.history_prediction), len(self.history_decision))
		for i in range(rec_count):
			if(self.history_prediction[i] != self.history_decision[i]):
				intervention_count+=1

		return intervention_count/rec_count
