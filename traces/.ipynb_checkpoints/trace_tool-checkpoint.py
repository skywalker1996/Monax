import numpy as np
import matplotlib.pyplot as plt
import os

def getTrace(path):
	PACKET_SIZE = 1500.0  # bytes
	BITS_IN_BYTE = 8.0
	MBITS_IN_BITS = 1024**2
	MILLISECONDS_IN_SECONDS = 1000.0 
	N = 100
	time_all = []
	packet_sent_all = []
	last_time_stamp = 0
	packet_sent = 0
	with open(path, 'rb') as f:
		for line in f:
			if(len(line.split())==0):
				continue
			time_stamp = int(line.split()[0])
			if time_stamp == last_time_stamp:
				packet_sent += 1
				continue
			else:
				time_all.append(last_time_stamp)
				packet_sent_all.append(packet_sent)
				packet_sent = 1
				last_time_stamp = time_stamp

		traces = []
		start_time = 0   # seconds
		count = 0
		pkt_sum = 0
		epoch_start_ts = time_all[0]
		for i in range(len(time_all)):
			if(time_all[i]>=start_time*MILLISECONDS_IN_SECONDS and time_all[i]<=(start_time+1)*MILLISECONDS_IN_SECONDS):
				pkt_sum += packet_sent_all[i]
			else:
				start_time+=1

				throughputInSec = ((PACKET_SIZE * BITS_IN_BYTE * pkt_sum) / (time_all[i]-epoch_start_ts))*MILLISECONDS_IN_SECONDS / MBITS_IN_BITS
				traces.append(throughputInSec)
				epoch_start_ts = time_all[i]
				pkt_sum = 0

	return traces


def gen_static_trace(bandwidth):
	packets_per_second = bandwidth*(10**6)/8/1500
