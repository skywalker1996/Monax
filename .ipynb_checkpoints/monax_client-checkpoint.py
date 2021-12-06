import matplotlib 
matplotlib.use('Agg') 
from configs.config import Config
import json
import time
from queue import Queue  
import numpy 
import socket
from threading import Thread, Lock
from lib.monitorFlux import MonitorFlux
import ipaddress
import selectors
import logging
import sys
import argparse

from utils.helpers import (
    curr_ts_ms, apply_op, get_open_udp_port,
    READ_FLAGS, WRITE_FLAGS, ALL_FLAGS)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='./logs/client.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, filemode='w')
	
	
START_CODE = '{st}'
LEN_START_CODE = len(START_CODE)

class Client(object):
	def __init__(self, global_cfg, multiflow_cfg, receiver_id):
		

		logging.debug("init start") 
		self.config = Config(global_cfg)
		
		self.recv_buf_size = int(self.config.get("client", "recv_buf_size"))
		self.send_buf_size = int(self.config.get("client", "send_buf_size"))
		
		self.receiver_id = receiver_id
		self.multi_flow = True if int(self.config.get("TC", "Multiflow"))==1 else False
		
		self.receiver_id = receiver_id
		
		if(self.multi_flow):
			self.multiflow_config = Config(multiflow_cfg)
			self.ip = self.multiflow_config.get("client-"+str(self.receiver_id), "ip")
			self.port = int(self.multiflow_config.get("client-"+str(self.receiver_id), "port"))
			
			if(receiver_id==0):
				self.protocal = 'UDP'
				print("********** Monax primary flow")
			else:
				self.protocal = 'TCP'
				print("********** TCP flow")
		else:
			self.ip = self.config.get("client", "ip")
			self.port = int(self.config.get("client", "port"))
			self.protocal = self.config.get("TC", "protocal")

		self.recv_buffer = Queue(self.recv_buf_size)
		self.send_buffer= Queue(self.send_buf_size)
		
		self.socket_read_buffer = b''
		
		self.ack_period = int(self.config.get("client", "ack_period"))

		# every ack period' data is a batch
		self.last_batch = 0
		self.upload_record = True
		self.video_streaming = True if int(self.config.get("experiment", "VIDEO_STREAMING"))==1 else False
		
		db_para = {}
		db_para['url'] = self.config.get('database', 'url')
		db_para['token'] = self.config.get('database', 'token')
		db_para['org'] = self.config.get('database', 'org')
		db_para['bucket'] = self.config.get('database', 'bucket')

		self.Monitor = MonitorFlux(db_para=db_para, 
								   plot_number=5, 
								   monitor_targets=None)

		

		self.router = {}
		
		self.running = True
		
		self.send_ack_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.send_ack_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		
		self.send_ffplay_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.send_ffplay_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

		### recv socket
		if(self.protocal=='UDP'):
			self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			self.client_sock.bind((self.ip, self.port))
		elif(self.protocal=='TCP'):
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
			sock.bind((self.ip, self.port))
			sock.listen(5)
			self.client_sock, self.client_address = sock.accept()
			self.client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
			
		logging.debug("init successfully") 
		
		


	def start(self):

# 		self.sendAgent_thread = Thread(target=self.sendAgent, args=())
# 		self.sendAgent_thread.daemon = True

		self.recvAgent_thread = Thread(target=self.recvAgent, args=())
		self.recvAgent_thread.daemon = True

# 		self.sendAgent_thread.start()
		self.recvAgent_thread.start()

# 		self.sendAgent_thread.join()
		self.recvAgent_thread.join()


# 	def sendAgent(self):
# 		if(self.protocal=='UDP'):
# 			while(True):
# 				data = self.send_buffer.get()
# 				self.client_sock.sendto(data, self.router['back'])
# 				if DEBUG:
# 					self.Log_send("send ack back!!") 
# #                 print('send ack back!!!')
# #                 self.send_back_sock.sendto(data,(self.server_ip, self.server_port)

# 		elif(self.protocal=='TCP'):
# # 			while(self.send_buffer.qsize()==0):
# # 				continue
# # 			sock.connect((self.server_ip, self.server_port))
# 			while(True):
# 				data = self.send_buffer.get()
# 				self.router['back'].send(data)


	def recvAgent(self):

		logging.debug('start recieving data!')

		recv_size = 0
		if(self.protocal=='UDP'):
			data,address = self.client_sock.recvfrom(2000)
			if('back' not in self.router):
				self.router['back'] = address
# 				print('send ack to address: ',address)
		elif(self.protocal=='TCP'):
			data,address = self.client_sock.recvfrom(2000)
		
		self.socket_read_buffer+=data
		packet = self.packetsplit(1)[0]
# 		packet = data
		
		
# 		data = sock.recv(2000)
		if(self.video_streaming):
			dataFrame = self.Unpack_VideoPacket(packet)
		else:
			dataFrame = self.Unpack_DataFrame(packet)
# 			print(f"client recv time = {dataFrame['create_time']}")
#         self.send_ffplay_sock.sendto(dataFrame['data'], ("192.168.0.107", 8878))
		start_time = time.time()
		recv_size+=len(data)
		estimated_bw = 0.0
		self.one_way_delay = 0.0
		pkt_log = {}
		self.current_window = int(dataFrame['pkt_id']/self.ack_period)
# 		print(f'initial window = {self.current_window}')
		self.current_frame_id = dataFrame['frame_id']
# 		print(f'#143 add window: {self.current_window}')
		pkt_log[self.current_window] = 1

		while(self.running):
			try:
				if(self.protocal=='UDP'):
					data = self.client_sock.recv(5000)
	# 				print(len(data))
				elif(self.protocal=='TCP'):
					data = self.client_sock.recv(5000)
	# 				print(len(data))
			except:
				self.running = False
				break
		
			self.socket_read_buffer+=data
			packets = self.packetsplit(2)
			if(len(packets)==0):
# 				print("no packets found!")
				continue
# 			packets = [data]
			sss_time = time.time()
			for packet in packets: 
				if(self.video_streaming):
					dataFrame = self.Unpack_VideoPacket(packet)
	# 				self.send_ffplay_sock.sendto(dataFrame['data'], ("192.168.0.107", 8878))
				else:
					dataFrame = self.Unpack_DataFrame(packet)

# 				print(f"*** pkt_id = {dataFrame['pkt_id']} with timestamp = {dataFrame['create_time']}")
				
# 				print("dataFrame = ", dataFrame)

				## check loss and send ack to server
				if(dataFrame['pkt_id']%self.ack_period==0 and dataFrame['pkt_id']!=0):
					recv_count = 0
					total = 0
					# pkt_log[self.current_window]+=1
					windows = list(pkt_log.keys())
# 						print(windows)
					for window in windows[:]:
						recv_count+=pkt_log[window]
# 							print('remove window number:{} with count:{}'.format(window, pkt_log[window]))
						total+=self.ack_period
						pkt_log.pop(window)

					loss_rate = round((total-recv_count)/total,5)

# 						print('total package:{} , but receive only {}'.format(total,recv_count))
# 					print((int((time.time()%(10**6))*1000) - dataFrame["create_time"])/1.0)
					ack = self.form_ackFrame(dataFrame, loss_rate*100, estimated_bw)
					self.one_way_delay = (int((time.time()%(10**3))*1000000) - dataFrame['create_time'])/1000
					# print('loss = {}'.format(loss_rate))
	# 				self.send_buffer.put(ack)
# 					print('send ack!!!')

					if(self.protocal=='UDP'):
						self.client_sock.sendto(ack, self.router['back'])
	# 					print('send ack!!!')
					elif(self.protocal=='TCP'):
						self.client_sock.send(ack)

# 					print(f"return ack with pkt id = {dataFrame['pkt_id']} and timestamp = {time.time()%1000}")
	# 				self.send_back_sock.sendto(ack,(self.server_ip, self.server_port))

					self.current_window = int(dataFrame['pkt_id']/self.ack_period)
	# 				print(f'#190 self.current_window = {self.current_window}')
					pkt_log[self.current_window] = 1 
					if(self.upload_record):
						record = self.construct_data(estimated_bw,self.one_way_delay)
						self.Monitor.pushData(measurement = 'monax-client-'+str(self.receiver_id), datapoints = [record], tags = {'version': 0.1} )
				else:
					if(dataFrame['frame_id']>=self.current_frame_id-1):
						if(int(dataFrame['pkt_id']/self.ack_period) in pkt_log):
	# 						print(f"#198 add window: {int(dataFrame['pkt_id']/self.ack_period)}")
							pkt_log[int(dataFrame['pkt_id']/self.ack_period)]+=1
							# print('batch id = {} and count = {} and pktid={}'.format(int(dataFrame['pkt_id']/self.ack_period), pkt_log[int(dataFrame['pkt_id']/self.ack_period)],dataFrame['pkt_id']))
						else:
	# 						print(f"#201 pkt_id={dataFrame['pkt_id']}")
	# 						print(f"#202 add window: {int(dataFrame['pkt_id']/self.ack_period)}")
							pkt_log[int(dataFrame['pkt_id']/self.ack_period)]=1
					else:
						continue

				current_time = time.time()
				if(current_time-start_time>=1):
					estimated_bw = (recv_size*8)/10**6  #Mbps
					recv_size = 0
					start_time = time.time()
					logging.debug("Estimated throughput = {}".format(estimated_bw))
	# 				if DEBUG:
	# 					self.Log_send("Estimated throughput = {}".format(estimated_bw)) 
				else:
					recv_size+=len(packet)
			
# 			print("Cost time = ", time.time()-sss_time)
			


	def form_ackFrame(self, dataFrame, loss_rate, recv_throughput):

		if("slice_id" not in dataFrame):
			dataFrame["slice_id"] = 0
		ackFrame = {}
		ackFrame['create_time'] = dataFrame['create_time']
		ackFrame['loss'] = loss_rate
		ackFrame = b''
		ackFrame += ipaddress.ip_address(self.ip).packed
		ackFrame += self.port.to_bytes(4, byteorder="big")
		ackFrame += ipaddress.ip_address(dataFrame['source_ip']).packed
		ackFrame += dataFrame['source_port'].to_bytes(4, byteorder="big")
		ackFrame += dataFrame['pkt_id'].to_bytes(4, byteorder="big")
		ackFrame += dataFrame["frame_id"].to_bytes(4, byteorder="big")
		ackFrame += dataFrame["slice_id"].to_bytes(4, byteorder="big")
		ackFrame += dataFrame["create_time"].to_bytes(4, byteorder="big")
		ackFrame += int(loss_rate*100).to_bytes(4, byteorder="big")  # because to_bytes does not surpport float
		ackFrame += int(recv_throughput*100).to_bytes(4, byteorder="big")  #  to_bytes does not surpport float

		return ackFrame

	def Unpack_DataFrame(self, data):

		dataFrame = {}
# 		print(data)
		dataFrame['source_ip'] = str(ipaddress.ip_address(bytes(data[0:4])))
		dataFrame['source_port'] = int.from_bytes(bytes(data[4:8]), byteorder="big")
		dataFrame['dest_ip'] = str(ipaddress.ip_address(bytes(data[8:12])))
		dataFrame['dest_port'] = int.from_bytes(bytes(data[12:16]), byteorder="big")
		dataFrame['pkt_id'] = int.from_bytes(bytes(data[16:20]), byteorder="big")
		dataFrame['frame_id'] = int.from_bytes(bytes(data[20:24]), byteorder="big")
		dataFrame['slice_id'] = int.from_bytes(bytes(data[24:28]), byteorder="big")
		dataFrame['create_time'] = int.from_bytes(bytes(data[28:32]), byteorder="big")
		return dataFrame

	def Unpack_VideoPacket(self, data):

		dataFrame = {}
		dataFrame['source_ip'] = str(ipaddress.ip_address(bytes(data[0:4])))					
		dataFrame['source_port'] = int.from_bytes(bytes(data[4:8]), byteorder="big")
		dataFrame['dest_ip'] = str(ipaddress.ip_address(bytes(data[8:12])))
		dataFrame['dest_port'] = int.from_bytes(bytes(data[12:16]), byteorder="big")
		dataFrame['pkt_id'] = int.from_bytes(bytes(data[16:20]), byteorder="big")
		dataFrame['frame_id'] = int.from_bytes(bytes(data[20:24]), byteorder="big")
		dataFrame['create_time'] = int.from_bytes(bytes(data[24:28]), byteorder="big")
		dataFrame['data'] = bytes(data[28:])

		return dataFrame

	def construct_data(self,recv_throughput, recv_delay):

		record = {}
		record['client_recv_throughput'] = recv_throughput
		record['client_recv_delay'] = recv_delay

		return record
	
	def packetsplit(self,mode):
		index_list = []
		len_startCode = len(START_CODE)
		res = []
		flag = 0
		i = 0
		while(i<len(self.socket_read_buffer)):
			if(self.socket_read_buffer[i]=="{".encode()[0]):
				if(bytes(self.socket_read_buffer[i:i+len_startCode])==START_CODE.encode()):
					if(i+len_startCode+2<=len(self.socket_read_buffer)):
						packet_len = int.from_bytes(self.socket_read_buffer[i+len_startCode:i+len_startCode+2], byteorder="big")
						packet_start_index = i+len_startCode+2
						if(packet_start_index+packet_len<=len(self.socket_read_buffer)):
							packet = self.socket_read_buffer[packet_start_index:packet_start_index+packet_len]
							self.socket_read_buffer = self.socket_read_buffer[packet_start_index+packet_len:]
							res.append(packet)
							if(mode==1):
								break
						else:
							break
					else:
						break
				else:
					i+=1
			else:
				i+=1

		return res
	
# 	def packetsplit(self,mode):
# 		index_list = []
# 		len_startCode = len(START_CODE)
# 		count = 0
# 		for i in range(len(self.socket_read_buffer)):
# 			if(self.socket_read_buffer[i]=="{".encode()[0]):
# 				if(bytes(self.socket_read_buffer[i:i+len_startCode])==START_CODE.encode()):
# 					index_list.append(i)
# 					count+=1
# 					if(mode==1 and count==2):
# 						break
						
# 		res = []
# 		if(len(index_list)==0):
# 			return res
# 		elif(len(index_list)==1):
# 			res.append(self.socket_read_buffer[index_list[0]+len_startCode:])
# 			self.socket_read_buffer = []
# 		else:
# 			for i in range(len(index_list)-1):
# 				data = self.socket_read_buffer[index_list[i]+len_startCode:index_list[i+1]]
# 				res.append(bytes(data))
# 			self.socket_read_buffer = self.socket_read_buffer[index_list[-1]:]

# 		return res
    
	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=1)
	args = parser.parse_args()
	
	client = Client(global_cfg = 'global.conf', multiflow_cfg = 'multi-flow.conf', receiver_id=args.id)
	client.start()
	print("client exit!!!")
