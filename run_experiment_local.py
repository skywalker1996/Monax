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

out_temp = tempfile.SpooledTemporaryFile(1000000)
fileno = out_temp.fileno()


configs = Config('global.conf')
multi_flow_config = Config('multi-flow.conf')

### initiaize
TRACE = configs.get("experiment", "TRACE")
TRACE_EPOCH = int(configs.get("experiment", "TRACE_EPOCH"))
TRACE_PORTION = float(configs.get("experiment", "TRACE_PORTION"))
VIDEO_PATH = configs.get("experiment", "VIDEO")
BASE_DELAY = configs.get("experiment", "BASE_DELAY")


STREAM_TYPE = configs.get("experiment", "STREAM_TYPE")
CC = configs.get("experiment", "CC")
RECORD_TYPE = configs.get("experiment", "RECORD_TYPE")

TIME_MARK = str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))


MULTI_FLOW = True if int(configs.get("TC", "Multiflow"))==1 else False


trace = getTrace(TRACE)
trace = trace[:int(TRACE_PORTION*len(trace))] 
PER_EPOCH_TIME = len(trace)
EXPERIMENT_TIME = PER_EPOCH_TIME * TRACE_EPOCH

bw_average = np.array(trace).mean()


db_para = {}
db_para['url'] = configs.get('database', 'url')
db_para['token'] = configs.get('database', 'token')
db_para['org'] = configs.get('database', 'org')
db_para['bucket'] = configs.get('database', 'bucket')



def generate_comds(multi_flow_config, epoch):
	
	sections = multi_flow_config.sections()
	
	interval = int(multi_flow_config.get('options', 'interval'))		
	flow_num = int((len(sections)-1)/2)
	
# 	0,end                    id*interval, flow_num+(flow_num-id-1) 
# 	1*interval,
# 	2*interval,
# 	.
# 	.
# 	.
# 	(flow_num-2)*interval, (flow_num+1)*interval
# 	(flow_num-1)*interval, flow_num * interval
	   
	time_range = [(0,'end')]
	
	for i in range(1,flow_num):
		time_range.append((i*interval, (2*flow_num-i-1)*interval))
		
	server_comds = []
	client_comds = []
	
	for i in range(flow_num):
		server_comds.append(f'python monax_server.py --id {i} --time_range {time_range[i][0]},{time_range[i][1]} --time_mark {TIME_MARK} --epoch {epoch}')
		client_comds.append(f'python monax_client.py --id {i} --time_mark {TIME_MARK} --epoch {epoch}')
	
	return server_comds, client_comds
	
	

def experiment(epoch):	

	### kill existing monax programs
	print("starting ...")
	Popen("kill -9 $(ps -aux | grep monax | awk '{print $2}')", shell=True)
	time.sleep(2)

	if(MULTI_FLOW):
		server_comds, client_comds = generate_comds(multi_flow_config, epoch)

		server_comds = ' & '.join(server_comds)
		client_comds = ' & '.join(client_comds)
	else:
		server_comds = f'python monax_server.py --id 0 --time_mark {TIME_MARK} --epoch {epoch}'
		client_comds = f'python monax_client.py --id 0 --time_mark {TIME_MARK} --epoch {epoch}'
	
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
		res = cal_performance(file_path)

	elif(RECORD_TYPE == RECORD_TYPE_CSV):

		save_dir = f"./record/{TIME_MARK}"
	
		record_file = f"server_{CC}_epoch_{epoch}.csv"
		file_path = os.path.join(save_dir, record_file)
		res = cal_performance(file_path)

	return res

if __name__ == '__main__':

	results = []

	EPOCH = 1
	for i in range(EPOCH):
		print(f"epoch: {i}")
		res = experiment(i)
		print(res)
		results.append(res)

	#### compute avg results

	dataframe = pd.DataFrame({'RTT_average':[r['RTT_average'] for r in results], 
							  'queue_delay_average':[r['queue_delay_average'] for r in results], 
							  'end2end_average':[r['end2end_average'] for r in results], 
							  'Loss_average':[r['Loss_average'] for r in results],
							  'send_rate_average':[r['send_rate_average'] for r in results], 
							  'delivery_rate_average':[r['delivery_rate_average'] for r in results]})

	file_path = f"./results/{STREAM_TYPE}/{CC}_{TIME_MARK}.csv"
	dataframe.to_csv(file_path, sep=',', mode='w+')
	print("========= Experiment finish ==========")

# 	print('============ Results ============')
# 	print('***** 平均RTT = {} ms'.format(round(RTT_average,4)))
# 	print('***** 平均排队时延 = {} ms'.format(round(queue_delay_average,4)))
# 	print('***** 平均端到端时延 = {} ms'.format(round(end2end_average,4)))
# 	print('***** 平均 send rate = {} Mbps'.format(round(send_rate_average,4)))
# 	print('***** 平均 delivery rate = {} Mbps'.format(round(delivery_rate_average,4)))

# 	print('***** 平均带宽 = {} Mbps'.format(round(bw_average,4)))




