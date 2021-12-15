import matplotlib
from torch._C import _create_function_from_trace 
matplotlib.use('Agg') 
from configs.config import Config
import json
import time
# from queue import Queue  
from multiprocessing import Manager,Process, Queue
import numpy 
import socket
import ctypes
import sys
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from threading import Thread, Lock
from lib.monitorFlux import MonitorFlux
import ipaddress
import multiprocessing
import selectors
import argparse
import logging
from lib.rate_control.rate_control_PCC import rate_control_module_PCC
from lib.rate_control.rate_control_RL import rate_control_module_RL
from lib.rate_control.rate_control_Monax import rate_control_module_Monax
# from lib.rate_control.test_indigo import Test
from subprocess import Popen

from lib.H264_Stream import H264_Stream
from traces.trace_tool import getTrace

from utils.helpers import (
	curr_ts_ms, apply_op, get_open_udp_port,
	READ_FLAGS, WRITE_FLAGS, ALL_FLAGS)

from configs.Common import *
import pandas as pd

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

DATA_LOAD = ('1'*1280).encode()


if not os.path.exists('./logs'):
	os.mkdir('./logs')
if not os.path.exists('./record'):
	os.mkdir('./record')

logging.basicConfig(filename='./logs/server.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, filemode='w')


class Server(object): 
	def __init__(self, global_cfg, multiflow_cfg, sender_id, time_range, time_mark, epoch):

		logging.debug("init start")
		self.config = Config(global_cfg)
		self.recv_buf_size = int(self.config.get("server", "recv_buf_size"))
		self.send_buf_size = int(self.config.get("server", "send_buf_size"))
		self.ip = self.config.get("server", "ip")

		self.stream_type = self.config.get("experiment", "STREAM_TYPE")
		self.video_path = self.config.get("experiment", "VIDEO")
		
		self.multi_flow = True if int(self.config.get("TC", "Multiflow"))==1 else False

		self.recv_buffer = Queue(self.recv_buf_size)
		self.send_buffer= Queue(self.send_buf_size)

		

		self.env = self.config.get("experiment", "ENV")

		### time mark for record file directory
		self.time_mark = time_mark
		self.epoch = epoch

		### sending parameters
		self.cwnd = 100
		self.using_rate_control = True
		self.next_ack = 0
		
		### Multiflow Setting
		self.sender_id = sender_id

		if(self.multi_flow):
			self.multiflow_config = Config(multiflow_cfg)
			self.client_ip = self.multiflow_config.get("client-"+str(self.sender_id), "ip")
			self.client_port = int(self.multiflow_config.get("client-"+str(self.sender_id), "port"))
			self.flow_start_time = int(time_range.split(',')[0])
			self.flow_end_time = int(time_range.split(',')[1]) if time_range.split(',')[1]!='end' else 'end'
			
			self.cc = self.multiflow_config.get("server-"+str(self.sender_id), "CC")
			self.protocol = PROTOCOL_MAP[self.cc]

			print(f"[Flow {self.sender_id} ({self.protocol})]: server side starts!")

			# if(sender_id==0):
			# 	self.protocol = PROTOCOL_UDP
			# 	print("********** Monax primary flow")
			# else:
			# 	self.protocol = PROTOCOL_TCP
			# 	print("********** TCP flow")
			
		else:
			self.client_ip = self.config.get("client", "ip")
			self.client_port = int(self.config.get("client", "port"))
			self.cc = self.config.get("experiment", "CC")
			self.protocol = PROTOCOL_MAP[self.cc]
			
		
		self.primary_flow = True if self.sender_id==0 else False


		### data info
		self.pkt_id = 0  # divided into packet batches
		self.batch_id = 0
		self.slice_per_frame = 50
		self.frame_id = 0
		self.slice_id = 0
		self.FPS = 30

		### monitored sending throughput
		self.sending_rate = 0.0

		self.record_window = 30
		self.network_delay_record = [0] * self.record_window    #单位ms
		self.queue_delay_record = [0] * self.record_window
		self.packet_loss_record = [0] * self.record_window
		
		trace_path = self.config.get("experiment", "TRACE")
		trace_portion = float(self.config.get("experiment", "TRACE_PORTION"))
		exp_round = int(self.config.get("experiment", "TRACE_EPOCH"))
		
		
		self.trace = getTrace(trace_path)
		
		if(len(self.trace)==0):
			bd = float(trace_path.split('/')[-1].split('mbps')[0])
			self.trace = [bd]*600
		self.trace = self.trace[:int(trace_portion*len(self.trace))] 
		
# 		print(self.trace[50:70])
		self.trace_point = 2
		self.exp_round = exp_round 
		self.trace_count = 0
# 		self.bw_limit = self.getNextBandwidth()
		self.bw_limit = 0.0
		

		self.rtt_ewma = None
		self.sending_rate_ewma = None
		self.delivery_rate_ewma = None

		self.min_rtt = None
		self.current_rtt = 0.0

		self.current_queue_delay = 0.0
		self.current_loss = 0.0

		self.record_count = 0   # for cold start

		### Estimate Setting 
		self.estimate_start_time = time.time()
		self.last_send_time = time.time()
		self.estimate_interval = 1
		self.send_count = 0

		self.record_start_time = time.time()
		self.record_save_interval = 0.05


		if(self.cc == CC_PCC):
			self.RC_Module = rate_control_module_PCC(self.sending_rate)
		elif(self.cc == CC_MONAX):
			self.RC_Module = rate_control_module_Monax(5, 0.2, 1, 0.2, 0.2,self.cwnd)
		elif(self.cc == CC_RL):
			self.RC_Module = rate_control_module_RL(self.sending_rate)
		# elif(self.cc == CC_INDIGO):
		# 	self.RC_Module = Test("lib/rate_control/model/model")

		
		### Record Setting 
		self.record_type = self.config.get("experiment", "RECORD_TYPE")

		if(self.record_type == RECORD_TYPE_DATABASE):		
			db_para = {}
			db_para['url'] = self.config.get('database', 'url')
			db_para['token'] = self.config.get('database', 'token')
			db_para['org'] = self.config.get('database', 'org')
			db_para['bucket'] = self.config.get('database', 'bucket')
			self.Monitor = MonitorFlux(db_para=db_para, plot_number=5, monitor_targets=None)

		elif(self.record_type == RECORD_TYPE_CSV):
			self.records = {}
			self.records["time"] = []
			self.records["rtt"] = []
			self.records["queue_delay"] = []
			self.records["packet_loss"] = []
			self.records["server_sending_rate"]	= []
			self.records["delivery_rate"] = []
			
		# print('connecting database successfully!')
		

		self.poller = selectors.DefaultSelector()
# 		time.sleep(1)
# 		avail_port = self.handshake()
		### send socke	t
		if(self.protocol==PROTOCOL_UDP):
			self.server_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
			self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			# self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
			self.server_sock.connect((self.client_ip, self.client_port))
# 			self.server_sock.setblocking(False)
			logging.debug('connect client addr: {}, {}'.format(self.client_ip, self.client_port))

		elif(self.protocol==PROTOCOL_TCP):
# 			TCP_CONGESTION = getattr(socket, 'TCP_CONGESTION', 13)
			self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
			self.server_sock.connect((self.client_ip, self.client_port))

		self.source_addr = self.server_sock.getsockname()
		self.poller.register(self.server_sock, ALL_FLAGS)

		self.running = True
		logging.debug("init successfully")


	def start(self):

		self.mainThread = Thread(target=self.run, args=())
		self.mainThread.daemon = True

		if(self.stream_type==STREAM_TYPE_VIDEO):
			self.Encoder_thread = Process(target=self.Encoder, args=(self.send_buffer,self.video_path))
			self.Encoder_thread.daemon = True 
			self.Encoder_thread.start() 

		self.mainThread.start()

		self.mainThread.join()
		if(self.stream_type==STREAM_TYPE_VIDEO):
			self.Encoder_thread.join()
		sys.exit(0)


	def recv(self, sock):
# 		print("recv ack")
		data = sock.recv(2000)
		data = self.Unpack_ackFrame(data)

		### update state
		self.current_rtt = (int((time.time()%(10**6))*1000) - data['create_time'])/1.0 #ms
		
# 		print(f"current time = {int((time.time()%(10**6))*1000)} while recv create_time = {data['create_time']}", )

		if(self.min_rtt is None):
			self.min_rtt = self.current_rtt
		else:	
			self.min_rtt = min(self.min_rtt, self.current_rtt)

		if(self.rtt_ewma is None):
			self.rtt_ewma = self.current_rtt
		else:
			self.rtt_ewma = (0.875*self.rtt_ewma + 0.125*self.current_rtt)
		
		if(self.delivery_rate_ewma is None):
			self.delivery_rate_ewma = data['recv_throughput']
		else:
			self.delivery_rate_ewma = (0.875*self.delivery_rate_ewma + 0.125*data['recv_throughput'])
			
		if(self.sending_rate_ewma is None):
			self.sending_rate_ewma = self.sending_rate
		else:
			self.sending_rate_ewma = (0.875*self.sending_rate_ewma + 0.125*self.sending_rate)
		

		pkt_id = data['pkt_id']	
		self.next_ack = max(self.next_ack, pkt_id+1)
# 		print(f"expect the next akc pkt_id = {self.next_ack} and timestamp = {time.time()%1000}")

		self.current_loss = data['loss']

		self.network_delay_record = self.network_delay_record[1:]+[self.rtt_ewma]
		self.packet_loss_record = self.packet_loss_record[1:]+[self.current_loss]
		self.record_count += 1
		utility = 0.0
		self.throughput_error = self.sending_rate - data['recv_throughput']

		log_info = {}
		if(self.using_rate_control):
			if(self.protocol==PROTOCOL_UDP):
				### cold start
				if(self.record_count>11):
					state = {'network_delay': self.network_delay_record[-10:],
							  'packet_loss': self.packet_loss_record[-10:],
							  'sending_rate': self.sending_rate, 
							  'throughput_error': self.throughput_error, 
							  'queue_length': self.send_buffer.qsize(),
							  'cwnd': self.cwnd}
					if(self.cc==CC_PCC):
						self.sending_rate, utility = self.RC_Module.action(state)

					elif(self.cc==CC_MONAX):
						self.cwnd, utility, log_info=self.RC_Module.action(state)
						utility = 1.0
					elif(self.cc==CC_INDIGO):
						state = [self.rtt_ewma,
							self.delivery_rate_ewma, #self.delivery_rate_ewma
							self.sending_rate_ewma, #self.send_rate_ewma
							self.cwnd]
						action = self.RC_Module.sample_action(state)
						self.cwnd=self.RC_Module.take_action(action)
						print("cwnd,",self.cwnd)
	# 					print('####### cwnd: ', self.cwnd)
	# 					print('sending buffer size: ', self.send_buffer.qsize())

		current_time = time.time()
		if(current_time - self.record_start_time >= self.record_save_interval):
			if(self.record_type == RECORD_TYPE_DATABASE):
# 				print(f"upload data: RTT :{self.rtt_ewma}  loss: {self.current_loss}")
				record = self.construct_data(self.current_rtt, self.current_queue_delay, self.current_loss, self.sending_rate, data['recv_throughput'], utility, self.cwnd, self.bw_limit, log_info)
				self.Monitor.pushData(measurement = 'monax-server-'+str(self.sender_id), datapoints = [record], tags = {'version': 0.1} )
			elif(self.record_type == RECORD_TYPE_CSV):
				self.records["time"].append(int(time.time()))
				self.records["rtt"].append(self.current_rtt)
				self.records["queue_delay"].append(self.current_queue_delay)
				self.records["packet_loss"].append(self.current_loss)
				self.records["server_sending_rate"].append(self.sending_rate)
				self.records["delivery_rate"].append(data['recv_throughput'])

			self.record_start_time = time.time()

			# logging.debug("cannot upload data because RTT = ", self.current_rtt)


	def send(self, sock):

		data = None
# 		if(self.current_send_time-self.last_send_time>=self.sending_interval):
		
		if(self.check_flow() and (self.protocol==PROTOCOL_TCP or self.window_is_open())):
			if(self.stream_type==STREAM_TYPE_VIDEO):
				try:
					(data, self.frame_id, priority, put_time) = self.send_buffer.get_nowait()
				except:
					return
				self.current_queue_delay = (time.time()-put_time)*1000  #ms
				self.queue_delay_record = self.queue_delay_record[1:]+[self.current_queue_delay]
				self.pkt_id += 1
				data = self.Pack_H264Packet(self.client_ip, self.client_port, self.frame_id, data)
				sock.send(data)
			else:
				data = self.getNextData()	
				sock.send(data)
# 		else:
# 			print("window full")

			# print('send data index', send_index)
		self.current_send_time = time.time()
		current_time = time.time()
		if(current_time - self.estimate_start_time >= self.estimate_interval):
			self.sending_rate = (self.send_count*8)/(self.estimate_interval*(10**6))
			self.estimate_start_time = time.time()
			self.send_count = 0
# 				if(DEBUG):
# 					self.Log_send('server send throughput = {}'.format(self.sending_rate))
			# logging.debug('******server send throughput = {}'.format(self.sending_rate))
			# if(self.primary_flow):
			# 	print('******server send throughput = {}'.format(self.sending_rate))
			self.bw_limit = self.getNextBandwidth()
		else:
			if(data is not None):
				self.send_count+=len(data)

		self.last_send_time = self.current_send_time


	def run(self):
		while(self.running):
			events = self.poller.select(timeout=None)
			for fd, flag in events:
				if(flag&READ_FLAGS):
# 					print('recv func')
					self.recv(fd.fileobj)
				if(flag&WRITE_FLAGS):
# 					print("send func")
					self.send(fd.fileobj) 
		
		if(self.record_type==RECORD_TYPE_CSV):
			self.save_records()
		
		# self.exit()

	def save_records(self):
		
		save_dir = f"./record/{self.env}/{self.time_mark}"
		if(not os.path.exists(save_dir)):
			os.makedirs(save_dir)
		record_file = f"server_{self.sender_id}_{self.cc}_epoch_{self.epoch}.csv"
		dataframe = pd.DataFrame.from_dict(self.records)
		dataframe.to_csv(os.path.join(save_dir, record_file))
		print("************* save records in file: ", os.path.join(save_dir, record_file))
		time.sleep(10)


	def Encoder(self, send_buffer, video_path):
		H264_stream=H264_Stream(video_path=video_path)
		
		while(True):
			res = H264_stream.getNextPacket()
			if(res is None):
# 				print('111') 	
				logging.debug(f'sending queue length = {send_buffer.qsize()}')
				continue
			else:
				while(True):
					(data, frame_id, priority) = res
					try:
						send_buffer.put_nowait((data, frame_id, priority, time.time()))
						break
					except:
						continue
					


	def getNextData(self):

		if(self.slice_id<self.slice_per_frame):
			# print('frame_id:{}, slice_id:{}, pkt_id:{}'.format(self.frame_id, self.slice_id, self.pkt_id))
			data = self.Pack_DataFrame(self.client_ip, self.client_port, self.frame_id, self.slice_id)
			# print('frame_id:{}, slice_id:{}, pkt_id:{}'.format(self.frame_id, self.slice_id, self.pkt_id))
			self.pkt_id += 1
			self.slice_id += 1
		else:
			self.slice_id = 0
			self.frame_id += 1
			data = self.Pack_DataFrame(self.client_ip, self.client_port, self.frame_id, self.slice_id)
			# print('frame_id:{}, slice_id:{}, pkt_id:{}'.format(self.frame_id, self.slice_id, self.pkt_id))
			self.pkt_id += 1
			self.slice_id += 1

		return data

	def window_is_open(self):
		return self.pkt_id - self.next_ack < self.cwnd
	
	def check_flow(self):
		if(not self.multi_flow):
			return True
		else:
			if(self.trace_point>=self.flow_start_time and (self.flow_end_time=='end' or self.trace_point<=self.flow_end_time)):
				return True
			else:
# 				print("trace_count = ", self.trace_count)
				return False
				
	def Pack_DataFrame(self, des_ip, des_port, frame_id, slice_id):
		
		## Packet header: 32 bytes
		## Payload:       1280 bytes
		packet_len = 1312
		dataFrame = b''
		dataFrame += START_CODE.encode()
		dataFrame += packet_len.to_bytes(2, byteorder="big")
		dataFrame += ipaddress.ip_address(self.source_addr[0]).packed  
		dataFrame += self.source_addr[1].to_bytes(4, byteorder="big")
		dataFrame += ipaddress.ip_address(des_ip).packed
		dataFrame += des_port.to_bytes(4, byteorder="big")
		dataFrame += self.pkt_id.to_bytes(4, byteorder="big")
		dataFrame += frame_id.to_bytes(4, byteorder="big")
		dataFrame += slice_id.to_bytes(4, byteorder="big")
		dataFrame += int((time.time()%(10**6))*1000).to_bytes(4, byteorder="big")
# 		print('sending timestamp = ', int((time.time()%(10**6))*1000))
		dataFrame += DATA_LOAD
# 		print('pack pkt_id = {}'.format(self.pkt_id))

		return dataFrame

	def Pack_H264Packet(self, des_ip, des_port, frame_id, payload):

		## Packet header: 28 bytes
		payload_len = len(payload)
		packet_len = 28 + payload_len
		
		dataFrame = b''
		dataFrame += START_CODE.encode()
		dataFrame += packet_len.to_bytes(2, byteorder="big")
		dataFrame += ipaddress.ip_address(self.source_addr[0]).packed
		dataFrame += self.source_addr[1].to_bytes(4, byteorder="big")
		dataFrame += ipaddress.ip_address(des_ip).packed        ## 4 bytes
		dataFrame += des_port.to_bytes(4, byteorder="big")      ## 4 bytes
		dataFrame += self.pkt_id.to_bytes(4, byteorder="big")   ## 4 bytes
		dataFrame += frame_id.to_bytes(4, byteorder="big")      ## 4 bytes
		dataFrame += int((time.time()%(10**6))*1000).to_bytes(4, byteorder="big")  ## 4 bytes, 会有时间回环问题
		dataFrame += payload
		return dataFrame

	def Unpack_ackFrame(self, ackData):
		ackFrame = {}
		# ackFrame['source_ip'] = str(ipaddress.ip_address(ackData[0:4]))					
		# ackFrame['source_port'] = int.from_bytes(ackData[4:8], byteorder="big")
		# ackFrame['dest_ip'] = str(ipaddress.ip_address(ackData[8:12]))
		# ackFrame['dest_port'] = int.from_bytes(ackData[12:16], byteorder="big")
		ackFrame['pkt_id'] = int.from_bytes(ackData[16:20], byteorder="big")
		ackFrame['frame_id'] = int.from_bytes(ackData[20:24], byteorder="big")
		ackFrame['slice_id'] = int.from_bytes(ackData[24:28], byteorder="big")
		ackFrame['create_time'] = int.from_bytes(ackData[28:32], byteorder="big")
		ackFrame['loss'] = int.from_bytes(ackData[32:36], byteorder="big")/100
		ackFrame['recv_throughput'] = int.from_bytes(ackData[36:40], byteorder="big")/100

		return ackFrame


	def construct_data(self, rtt, queue_delay, packet_loss, send_rate, delivery_rate, utility, cwnd, bw_limit, log_info):

		record = {}
		record['rtt'] = float(rtt)
		record['queue_delay'] = float(queue_delay)
		record['packet_loss'] = float(packet_loss)
		record['server_sending_rate'] = float(send_rate)
		record['delivery_rate'] = float(delivery_rate)
		record['utility'] = float(utility)
		record['cwnd'] = int(cwnd)
		
		if(self.primary_flow):
			record['bw_limit'] = float(bw_limit)
# 		if('intervention_prob' in log_info):
# 			record['intervention_prob'] = log_info['intervention_prob']
		return record

	def getNextBandwidth(self):
		
# 		print('********** trace = ', self.trace)
		progress = round((self.trace_point/len(self.trace))*100, 1)
		print("\r", end="")
		print("epoch progress: {}% ".format(progress), end="")
		sys.stdout.flush()
		if(self.trace_point>=len(self.trace)):
			if(self.trace_count>=(self.exp_round-1)):
				logging.debug("finish trace")
				print(f"finish trace with trace_point = {self.trace_point}")
				self.running = False
				self.trace_point = 0
				# self.exit()
# 				sys.exit(0)
			else:
				self.trace_point = 0
				self.trace_count+=1
		bw = self.trace[self.trace_point]
# 		print('get trace self.trace[{}]: {}'.format(self.trace_point, bw))
		self.trace_point+=1
		
		return bw
	
	def exit(self):
		Popen("kill -9 $(ps -aux | grep monax_server.py | awk '{print $2}')", shell=True)




if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--time_range',default='0,end')
	parser.add_argument('--time_mark', default="default")
	parser.add_argument('--epoch', type=int, default=0)
	args = parser.parse_args()

	server = Server(global_cfg = 'global.conf', multiflow_cfg = 'multi-flow.conf', 
					sender_id=args.id, time_range=args.time_range, time_mark=args.time_mark, 
					epoch=args.epoch)
	server.start()
	print("server exit!!!")

