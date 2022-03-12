from subprocess import Popen
from sys import path
import time
import socket
from configs.Common import *
from traces.trace_tool import getTrace
from configs.config import Config
from utils.helpers import PullData, cal_performance
import numpy as np
import tempfile
import pandas as pd
import os


GLOBAL_CONFIG_FILE = 'global.conf'
MULTI_FLOW_CONFIG_FILE = 'multi-flow.conf'

TIME_MARK = str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))

out_temp = tempfile.SpooledTemporaryFile(1000000)
fileno = out_temp.fileno()


configs = Config(GLOBAL_CONFIG_FILE)

### initiaize
TRACE = configs.get("experiment", "TRACE")
TRACE_EPOCH = int(configs.get("experiment", "TRACE_EPOCH"))
TRACE_PORTION = float(configs.get("experiment", "TRACE_PORTION"))
VIDEO_PATH = configs.get("experiment", "VIDEO")
BASE_DELAY = configs.get("experiment", "BASE_DELAY")

ENV = configs.get("experiment", "ENV")


STREAM_TYPE = configs.get("experiment", "STREAM_TYPE")
CC = configs.get("experiment", "CC")
RECORD_TYPE = configs.get("experiment", "RECORD_TYPE")


MULTI_FLOW = True if int(configs.get("multiflow", "multiflow"))==1 else False
MULTI_FLOW_MODE = configs.get("multiflow", "multiflow_mode")
COMPETE_FLOW = configs.get("multiflow", "compete_flow")



trace = getTrace(TRACE)
if(len(trace)==0):
	bd = float(TRACE.split('/')[-1].split('mbps')[0])
	trace = [bd]*600

trace = trace[:int(TRACE_PORTION*len(trace))] 
PER_EPOCH_TIME = len(trace)


EXPERIMENT_TIME = PER_EPOCH_TIME * TRACE_EPOCH

bw_average = np.array(trace).mean()


db_para = {}
db_para['url'] = configs.get('database', 'url')
db_para['token'] = configs.get('database', 'token')
db_para['org'] = configs.get('database', 'org')
db_para['bucket'] = configs.get('database', 'bucket')



def generate_multiFlow_configs(cc, count, ip):
	configs = {}
	flow_count = 0
	for i in range(len(cc)):
		for j in range(count[i]):
			server_section = {}
			client_section = {}

			server_section["ip"] = ip
			server_section["CC"] = cc[i]

			client_section["ip"] = ip
			client_section["CC"] = cc[i]
			client_section["ip"] = ip
			client_section["port"] = 9000 + flow_count
			client_section["CC"] = cc[i]

			configs[f"server-{flow_count}"] = server_section
			configs[f"client-{flow_count}"] = client_section

			flow_count += 1

	return configs

def generate_multiflow_configs(compete_flow_count):

	cc = [CC, COMPETE_FLOW]
	count = [1, compete_flow_count]
	ip = "172.26.254.98"
	multi_flow_config = Config(MULTI_FLOW_CONFIG_FILE)
	configs = generate_multiFlow_configs(cc, count, ip)
	multi_flow_config.write_sections(configs)
	return


def generate_comds(multi_flow_config, epoch, mark):
	
	sections = multi_flow_config.sections()
	
	interval = int(configs.get('multiflow', 'interval'))		
	flow_num = int(len(sections)/2)
	
# 	0,end                    id*interval, flow_num+(flow_num-id-1) 
# 	1*interval,
# 	2*interval,
# 	.
# 	.
# 	.
# 	(flow_num-2)*interval, (flow_num+1)*interval
# 	(flow_num-1)*interval, flow_num * interval
	   
	time_range = [(0,'end')]
	
	if(MULTI_FLOW_MODE == MULTI_FLOW_MODE_SAME):
		for i in range(1,flow_num):
			time_range.append((0,'end'))
	elif(MULTI_FLOW_MODE == MULTI_FLOW_MODE_SEPARATE):
		for i in range(1,flow_num):
			time_range.append((i*interval, (2*flow_num-i-1)*interval))
	else:
		raise Exception("multi flow mode error!")
		
	server_comds = []
	client_comds = []
	
	for i in range(flow_num):
		server_comds.append(f'python monax_server.py --id {i} --time_range {time_range[i][0]},{time_range[i][1]} --mark {mark} --time_mark {TIME_MARK} --epoch {epoch}')
		client_comds.append(f'python monax_client.py --id {i} --mark {mark} --time_mark {TIME_MARK} --epoch {epoch}')
	
	return server_comds, client_comds
	
	

def experiment(epoch, mark):	

	### kill existing monax programs
	print("starting ...")
	Popen("kill -9 $(ps -aux | grep monax | awk '{print $2}')", shell=True)
	time.sleep(2)

	if(MULTI_FLOW):
		multi_flow_config = Config(MULTI_FLOW_CONFIG_FILE)
		server_comds, client_comds = generate_comds(multi_flow_config, epoch, mark)

		server_comds = ' & '.join(server_comds)
		client_comds = ' & '.join(client_comds)
	else:
		server_comds = f'python monax_server.py --id 0 --mark {mark} --time_mark {TIME_MARK} --epoch {epoch}'
		client_comds = f'python monax_client.py --id 0 --mark {mark} --time_mark {TIME_MARK} --epoch {epoch}'
	
	### start monax client and server
	start_server = ['mm-delay', str(BASE_DELAY), 
					'mm-link', TRACE, TRACE,
					'--uplink-queue=droptail --uplink-queue-args=packets=2048',
					'-- sh -c', f"'{server_comds}'"]
	
	start_client = client_comds
	start_server = ' '.join(start_server)

# 	client = Popen(start_client, stdout=fileno, stderr=fileno, shell=True)
# 	server = Popen(start_server, stdout=fileno, stderr=fileno, shell=True)
	client = Popen(start_client, shell=True)
	time.sleep(5)

	server = Popen(start_server, shell=True)

	print("experiment start success")

	if(MULTI_FLOW):
		time.sleep(10)
		# print(os.popen("ps -aux | grep monax_server.py").read())
		### wait finish
		while(True):
			if('python' not in os.popen("ps -aux | grep monax_server.py").read()):
				print(os.popen("ps -aux | grep monax_server.py").read())
				break
			else:
				continue

	server.wait()
	client.kill()

	print('experiment finish!')
	Popen("kill -9 $(ps -aux | grep monax | awk '{print $2}')", shell=True)

	record_file = str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))+'.csv'

	if(RECORD_TYPE == RECORD_TYPE_DATABASE):
		file_path = PullData(db_para, int(PER_EPOCH_TIME), record_file)
		

	elif(RECORD_TYPE == RECORD_TYPE_CSV):

		if(MULTI_FLOW):
			save_dir = f"./record/{ENV}/{TIME_MARK}/{mark}"
		else:
			save_dir = f"./record/{ENV}/{TIME_MARK}"
		
	
		record_file = f"server_0_{CC}_epoch_{epoch}.csv"
		file_path = os.path.join(save_dir, record_file)
		
	res = cal_performance(file_path)

	return res

if __name__ == '__main__':

	results = []

	
	if(MULTI_FLOW):
		for i in range(16):
			compete_flow_num = i+1
			generate_multiflow_configs(compete_flow_num)
			mark = f"{CC}-{COMPETE_FLOW}-{compete_flow_num}-{MULTI_FLOW_MODE}"
			print(f"experiment #{i} --- compete flow number = {compete_flow_num}")
			res = experiment(0, mark)
			print(res)
			results.append(res)
		

	else:
		mark = str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))
		EPOCH = 1
		for i in range(EPOCH):
			print(f"epoch: {i}")
			res = experiment(i, mark)
			print(res)
			results.append(res)

		#### compute avg results

	dataframe = pd.DataFrame({'RTT_average':[r['RTT_average'] for r in results], 
							'queue_delay_average':[r['queue_delay_average'] for r in results], 
							'end2end_average':[r['end2end_average'] for r in results], 
							'Loss_average':[r['Loss_average'] for r in results],
							'send_rate_average':[r['send_rate_average'] for r in results], 
							'delivery_rate_average':[r['delivery_rate_average'] for r in results]})


	file_dir = f"./results/{ENV}/{STREAM_TYPE}"

	if(not os.path.exists(file_dir)):
		os.makedirs(file_dir)
	file_name = f"{CC}_{TIME_MARK}.csv"
	dataframe.to_csv(os.path.join(file_dir, file_name), sep=',', mode='w+')
	print("========= Experiment finish ==========")

# 	print('============ Results ============')
# 	print('***** 平均RTT = {} ms'.format(round(RTT_average,4)))
# 	print('***** 平均排队时延 = {} ms'.format(round(queue_delay_average,4)))
# 	print('***** 平均端到端时延 = {} ms'.format(round(end2end_average,4)))
# 	print('***** 平均 send rate = {} Mbps'.format(round(send_rate_average,4)))
# 	print('***** 平均 delivery rate = {} Mbps'.format(round(delivery_rate_average,4)))

# 	print('***** 平均带宽 = {} Mbps'.format(round(bw_average,4)))




