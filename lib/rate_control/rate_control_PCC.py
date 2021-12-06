class rate_control_module_PCC(object):
	def __init__(self, initial_sending_bw):
 
# 		self.frame_pieces = frame_pieces
		# self.sending_delay_max = 1 / (60*frame_pieces)
		# self.server_sending_delay = self.sending_delay_max

		self.rec_window = 30

		self.cold_start = True

		self.network_delay_rec = [0,0]   #(min, max)
		self.packet_loss_rec = [0,0]
# 		self.send_delay_rec = [0,0]

		self.network_delay_g_rec = [0]*self.rec_window  # store previous gradients 
# 		self.send_delay_g_rec = [0]*self.rec_window  # store previous gradients  
		self.packet_loss_g_rec = [0]*self.rec_window


		self.network_delay_g2_rec = [0]*self.rec_window  # store previous gradients  
		self.packet_loss_g2_rec = [0]*self.rec_window
  
		self.utility_rec = [0]*self.rec_window

		state_list = ['step_1', 'step_2', 'action']
		self.state = 'step_1'

		self.ideal_delay = 100
		self.sending_bw_max = 40
		self.sending_bw_min = 5
		self.server_sending_bandwidth = initial_sending_bw
		
		self.initial_step_size = 0.2
		
		self.step_size_increase = 0.2
		self.step_size_decrease = 0.2
		self.step_size_try = 0.2
		
		self.global_RTT_min = 0.0
		self.buffer_bloating = False
		self.bloating_count = 0
		
		self.direction_count = 0
# 		self.decrease_count = 0
		
		self.throughput_error_threshold = 1
		
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
		
		utility =  send_rate_part + network_part
		utility_part = [send_rate_part, network_part]

		self.utility_rec = self.utility_rec[1:] + [utility]
		# print('compute the utility = ', utility)
		return utility, utility_part

	def action(self, monitor_params):
		#increase sending rate
		
# 		throughput_error = monitor_params['throughput_error']
# 		rate_delta = throughput_error - self.throughput_error_threshold
# 		if(rate_delta>0):
# 			print("rate_delta = ",rate_delta)
			
# 			self.state = 'step_1'
# 			utility,utility_part = self.compute_utility(monitor_params)
# 			if(self.server_sending_bandwidth-rate_delta*1.1<=self.sending_bw_min):
# 				return self.server_sending_bandwidth, utility
# 			else:
# 				self.server_sending_bandwidth-=rate_delta*1.1
# 				return self.server_sending_bandwidth, utility
				
		if(self.state == 'step_1'):

			self.state = 'step_2'
			utility,utility_part = self.compute_utility(monitor_params)
			if(self.server_sending_bandwidth+self.step_size_try>=self.sending_bw_max):
				return self.server_sending_bandwidth, utility
			return self.server_sending_bandwidth+self.step_size_try, utility
			

		#decrease sending rate
		elif(self.state == 'step_2'):
			self.state = 'action'
			utility,utility_part = self.compute_utility(monitor_params)
			self.utility_part_rec[0] = utility_part  #utility for increase sending rate

			if(self.server_sending_bandwidth-self.step_size_try<=self.sending_bw_min):
				return self.server_sending_bandwidth, utility
			return self.server_sending_bandwidth-self.step_size_try, utility

		elif(self.state == 'action'):
			self.state = 'step_1'
			#take actions
			utility,utility_part = self.compute_utility(monitor_params)
			self.utility_part_rec[1] = utility_part #utility for decrease sending rate

			delta = self.utility_rec[-1] - self.utility_rec[-2]
			delta_part = [self.utility_part_rec[1][i]-self.utility_part_rec[0][i] for i in range(2)]  #decrease - increase
            
			delta = sum(delta_part)
			# print()
			# print()
# 			print('******** Get the utility value = {}'.format(utility))
# 			print('******** Get the delta_part value = {}'.format(delta_part))
			
			if(self.buffer_bloating):
				if(self.last_step == 0):
					self.direction_count += 1
					if(self.direction_count>=self.speedup_thres):
						self.step_size_decrease += self.initial_step_size
				else:
					self.direction_count = 0
					self.step_size_decrease = self.initial_step_size
					
				self.server_sending_bandwidth -= self.step_size_decrease
				self.buffer_bloating = False
				self.last_step = 0
				return self.server_sending_bandwidth, utility
				
			if(delta > 0):

				if(self.last_step == 0):
					self.direction_count += 1
					if(self.direction_count>=self.speedup_thres):
						self.step_size_decrease += self.initial_step_size
				else:
					self.direction_count = 0
					self.step_size_decrease = self.initial_step_size

# 					self.decrease_count+=1
				if(self.server_sending_bandwidth-self.step_size_decrease<=self.sending_bw_min):
					return self.server_sending_bandwidth, utility
				else:
					self.server_sending_bandwidth -= self.step_size_decrease
					self.last_step = 0
					return self.server_sending_bandwidth, utility
# 					self.decrease_count+=1
# 					print('Decrease server sending rate to {}!'.format(self.server_sending_bandwidth))
			
			if(delta <= 0):

				if(self.last_step == 1):
					self.direction_count += 1
					if(self.direction_count>=self.speedup_thres):
						self.step_size_increase+=self.initial_step_size
				else:
					self.direction_count = 0
					self.step_size_increase = self.initial_step_size
				if(self.server_sending_bandwidth+self.step_size_increase>=self.sending_bw_max):
					return self.server_sending_bandwidth, utility
				else:
					self.server_sending_bandwidth += self.step_size_increase
	# 					print('Increase server sending rate to {}!'.format(self.server_sending_bandwidth))
					self.last_step = 1
				return self.server_sending_bandwidth, utility


	def get_non_zero(self, List):
		return [List[i] for i in range(len(List)) if List[i]>0]
