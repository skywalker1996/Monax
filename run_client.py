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

def experiment(epoch):	

	### kill existing monax programs
	print("starting ...")
	Popen("kill -9 $(ps -aux | grep monax_client | awk '{print $2}')", shell=True)
	time.sleep(2)

	
	start_client = f'python monax_client.py --id 0 --time_mark {TIME_MARK} --epoch {epoch}'
	
	client = Popen(start_client, shell=True)
	time.sleep(5)


	print("experiment start success")

	time.sleep(50000)
	client.wait()

	print('experiment finish!')
	Popen("kill -9 $(ps -aux | grep monax_client | awk '{print $2}')", shell=True)

	return

if __name__ == '__main__':

	experiment(0)

	print("========= Experiment finish ==========")

# 	print('============ Results ============')
# 	print('***** 平均RTT = {} ms'.format(round(RTT_average,4)))
# 	print('***** 平均排队时延 = {} ms'.format(round(queue_delay_average,4)))
# 	print('***** 平均端到端时延 = {} ms'.format(round(end2end_average,4)))
# 	print('***** 平均 send rate = {} Mbps'.format(round(send_rate_average,4)))
# 	print('***** 平均 delivery rate = {} Mbps'.format(round(delivery_rate_average,4)))

# 	print('***** 平均带宽 = {} Mbps'.format(round(bw_average,4)))




