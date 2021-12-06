import numpy as np
import matplotlib.pyplot as plt
import os
import sys

PACKET_SIZE = 1500.0  # bytes
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1024**2
MILLISECONDS_IN_SECONDS = 1000.0 
N = 100                 
# TRACE_PATH = r'data/throughput_traces/experiment_traces/aerial1'
# TRACE_PATH = r'FCC_broadband_badnwidth_mahi'
TRACE_PATH = r'./Belgium_4GLTE_bandwidth_mahi'
# TRACE_PATH = './FCC_broadband_badnwidth_mahi'
# TRACE_PATH = './Belgium_4GLTE_bandwidth_mahi'
# TRACE_PATH = r'data/throughput_traces/Norway_HSDPA_bandwidth_mahi'

if(len(sys.argv)>1):
	TRACE_INDEX =sys.argv[1]
else:
	TRACE_INDEX = 'report_foot_0001.log'
# TRACE_INDEX = 'report_train_0001.log'
# TRACE_INDEX = 'test_fcc_trace_1171_http---www.yahoo.com_0'
# TRACE_INDEX = 'test_fcc_trace_10652_http---www.amazon.com_0'
# TRACE_INDEX = 'test_fcc_trace_14731_http---www.facebook.com_0'


def plot(path):
	time_all = []
	packet_sent_all = []
	last_time_stamp = 0
	packet_sent = 0
	with open(path, 'rb') as f:
		for line in f:
			time_stamp = int(line.split()[0])
			if time_stamp == last_time_stamp: 
				packet_sent += 1
				continue
			else:
				time_all.append(last_time_stamp)
				packet_sent_all.append(packet_sent)
				packet_sent = 1
				last_time_stamp = time_stamp
	# print(len(time_all))
	# print(packet_sent_all)
	time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
	throuput_all = (((PACKET_SIZE * BITS_IN_BYTE * np.array(packet_sent_all[1:])) / time_window) * MILLISECONDS_IN_SECONDS) / MBITS_IN_BITS
	print(len(throuput_all))
	print(throuput_all)
	plt.plot(np.array(time_all[1:]) / MILLISECONDS_IN_SECONDS, 
			np.convolve(throuput_all, np.ones(N,)/N, mode='same'))
	plt.xlabel('Time (second)')
	plt.ylabel('Throughput (Mbit/sec)')
# 	plt.xlim(0, 100)
	plt.show()

if __name__ == "__main__":
	for trace in os.listdir(TRACE_PATH):
		if TRACE_INDEX in trace:
			print(trace)
			plot(TRACE_PATH + '/' + trace)	
            