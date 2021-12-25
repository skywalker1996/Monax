import socket
import time
import json

server_send_proxy=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
# server_send_proxy.connect(("127.0.0.1", 10000))
send_count = 0
start_time = time.time()
current_time = time.time()
pkt_id = 0

# data = {}
# data['id'] = pkt_id
# data['data'] = '1'*1316

packet = ""
for i in range(20):
	packet += str(i)
packet = packet.encode()

count = 10

server_send_proxy.bind(("172.26.254.98", 10001))
# server_send_proxy.bind(("47.101.194.15", 10000))

# server_send_proxy.bind(("100.64.0.4", 10000))
# server_send_proxy.bind(("100.64.0.2", 10000))
# server_send_proxy.sendto(packet, ("172.26.254.98", 10000))
# server_send_proxy.sendto(packet, ("100.64.0.1", 10000))
# server_send_proxy.sendto(packet, ("127.0.0.1", 10000))
server_send_proxy.sendto(packet, ("47.101.194.15", 10000))
data, address = server_send_proxy.recvfrom(500)
print(f"recv data len = {len(data)} from address: {address}")




# while(True):
# 	count-=1;
# # 	data['id'] = pkt_id
	
# # 	server_send_proxy.sendto(packet, ("172.26.254.98", 10000))
# 	server_send_proxy.sendto(packet, ("127.0.0.1", 10000))
# # 	server_send_proxy.send(packet)
	

# 	pkt_id += 1

# # 	print("pkt id = ", pkt_id)
# # 	time.sleep(0.0001)
# 	current_time = time.time()
# 	if(current_time - start_time >= 1):
# 		send_throughput = (send_count*8)/10**6 
# 		start_time = time.time()
# 		send_count = 0
# 		print('******server send throughput = {} Mbps'.format(send_throughput))
# 	else:
# 		send_count+=len(packet)
		
# 	msg = server_send_proxy.recv(1000)
# 	print("recv msg:"+msg)
# server_send_proxy.sendto("hello world".encode(), ("192.168.0.141",9003))
server_send_proxy.close()
