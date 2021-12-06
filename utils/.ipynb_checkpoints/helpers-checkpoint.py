# Copyright 2018 Francis Y. Yan, Jestin Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import os
import time
import errno
import selectors
import socket
import numpy as np
import operator
import numpy as np

import sys 
import os.path as osp
print(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from lib.influx_operator import InfluxOP
import pandas as pd



READ_FLAGS = selectors.EVENT_READ
WRITE_FLAGS = selectors.EVENT_WRITE
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS

math_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}


def apply_op(op, op1, op2):
    return math_ops[op](op1, op2)


def curr_ts_ms(): 
    if not hasattr(curr_ts_ms, 'epoch'):
        curr_ts_ms.epoch = time.time()

    return int((time.time() - curr_ts_ms.epoch) * 1000)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_open_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def normalize(state):
    return [state[0] / 200.0, state[1] / 200.0,
            state[2] / 200.0, state[3] / 5000.0]


def one_hot(action, action_cnt):
    ret = [0.0] * action_cnt
    ret[action] = 1.0
    return ret


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class RingBuffer(object):
    def __init__(self, length):
        self.full_len = length
        self.real_len = 0
        self.index = 0
        self.data = np.zeros(length)

    def append(self, x):
        self.data[self.index] = x
        self.index = (self.index + 1) % self.full_len
        if self.real_len < self.full_len:
            self.real_len += 1

    def get(self):
        idx = (self.index - self.real_len + np.arange(self.real_len)) % self.full_len
        return self.data[idx]

    def reset(self):
        self.real_len = 0
        self.index = 0
        self.data.fill(0)


class MeanVarHistory(object):
    def __init__(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0

    def append(self, x):
        """Append x to history.

        Args:
            x: a list or numpy array.
        """
        # x: a list or numpy array
        length_new = self.length + len(x)
        ratio_old = float(self.length) / length_new
        ratio_new = float(len(x)) / length_new

        self.length = length_new
        self.mean = self.mean * ratio_old + np.mean(x) * ratio_new
        self.square_mean = (self.square_mean * ratio_old +
                            np.mean(np.square(x)) * ratio_new)
        self.var = self.square_mean - np.square(self.mean)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var if self.var > 0 else 1e-10

    def get_std(self):
        return np.sqrt(self.get_var())

    def normalize_copy(self, x):
        """Normalize x and returns a copy.

        Args:
            x: a list or numpy array.
        """
        return [(v - self.mean) / self.get_std() for v in x]

    def normalize_inplace(self, x):
        """Normalize x in place.

        Args:
            x: a numpy array with float dtype.
        """
        x -= self.mean
        x /= self.get_std()

    def reset(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0

		
def PullData(db_para, time, record_file):
	
	client = InfluxOP(url=db_para["url"], token=db_para["token"], org=db_para["org"], bucket=db_para["bucket"])
# 	field = 'network_delay'
# 	tags = {'version':0.1}
	print('start pull')
	results = client.pullData(measurement="monax-server-0", field=None, tags=None, time_range=str(time)+'s')

	network_delay = np.array([res['network_delay'] for res in results])
	mask = network_delay<5000
	network_delay = network_delay[mask]
	print(len(network_delay))


	packet_loss = np.array([res['packet_loss'] for res in results])[mask]
	
	print(len(packet_loss))

	server_sending_rate = np.array([res['server_sending_rate'] for res in results])[mask]
	delivery_rate = np.array([res['delivery_rate'] for res in results])[mask]
# 	print(server_sending_rate)
	
	queue_delay = np.array([res['queue_delay'] for res in results])[mask]
	
	print('pull successfully!')

	dataframe = pd.DataFrame({'network_delay':network_delay, 'packet_loss':packet_loss, 'server_sending_rate':server_sending_rate, 'delivery_rate':delivery_rate, 'queue_delay':queue_delay})
# 	if(os.)
	file_path = "./record/"+record_file
	dataframe.to_csv(file_path, sep=',', mode='w+')

	return file_path

def cal_performance(record_file):
	
	df = pd.read_csv(record_file)
	network_delay = df['network_delay'].values
	packet_loss = df['packet_loss'].values
	server_sending_rate = df['server_sending_rate'].values
	delivery_rate = df['delivery_rate'].values
	queue_delay = df['queue_delay'].values
	
	#1. 平均RTT
	RTT_average = network_delay.mean()

	#2.时延超标率，RTT>70ms
	total_count = network_delay.shape[0]
	exceed_count = (network_delay>70).sum()
	RTT_exceed_ratio = exceed_count/total_count

	#3.平均丢包率
	Loss_average = packet_loss.mean()

	#4.average sending rate
	
# 	for i in range(len(server_sending_rate)):
# 		if(server_sending_rate[i]==0):
# 			continue
# 		else:
# 			non_zero_index_send = i
# 			break
			
# 	#5. average receiving rate
# 	for i in range(len(delivery_rate)):
# 		if(delivery_rate[i]==0):
# 			continue
# 		else:
# 			non_zero_index_recv = i
# 			break
	non_zero_index_recv = 0
	non_zero_index_send = 0
			
# 	print(server_sending_rate)
			
	#5. 平均排队时延
	queue_delay_average = queue_delay.mean()
	
	#6. 平均端到端时延
	end2end_average = (network_delay+queue_delay).mean()
			
	send_rate_average = server_sending_rate[non_zero_index_send:].mean()
	
	delivery_rate_average = delivery_rate[non_zero_index_recv:].mean()
	
	
	
	res = {}
	res['RTT_average'] = RTT_average
	res['RTT_exceed_ratio'] = RTT_exceed_ratio
	res['Loss_average'] = Loss_average
	res['send_rate_average'] = send_rate_average
	res['delivery_rate_average'] = delivery_rate_average
	res['queue_delay_average'] = queue_delay_average
	res['end2end_average'] = end2end_average
	
	return res


def getCurrentCC():
    
    result = os.popen('sysctl net.ipv4.tcp_congestion_control').read()
    CC = result.split('=')[1].split()[0]
    return CC
    

	