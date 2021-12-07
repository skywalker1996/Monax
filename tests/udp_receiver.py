import socket
import time
import json

recv_proxy = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
recv_proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_proxy.bind(("172.26.254.98", 10000))
# recv_proxy.bind(("127.0.0.1", 10000))
# recv_proxy.bind(('100.64.0.2', 10000))
start_time = time.time()
recv_count = 0


while(True):
	data, address = recv_proxy.recvfrom(100)
	print(f"recv data len = {len(data)} from address: {address}")
	recv_proxy.sendto("monitor".encode(), address)
	
# 	recv_count+=len(data)
# 	msg = json.loads(data.decode())
# 	print("pkt id = ", msg['id'])
# 	self.recv_buffer.put(data)
# 	print(addr)
# 	print(data.decode())
# 	recv_proxy.sendto("ack packet".encode(), addr)
# 	recv_proxy.sendto("ack packet".encode(), ('100.64.0.2', 10000))

# 	current_time = time.time()
# 	###modify the bandwidth every 1 second
# 	if(current_time - start_time >= 1):
# 		recv_throughput = (recv_count*8)/(10**6) 
# 		start_time = time.time()
# 		recv_count = 0
# 		print('******recv throughput = {} Mbps'.format(recv_throughput))
# 	else:
# 		recv_count+=len(data)