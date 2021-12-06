import numpy as np
# import matplotlib.pyplot as plt
import os

PACKET_SIZE = 1500.0  # bytes
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1024 ** 2
MILLISECONDS_IN_SECONDS = 1000.0
N = 100

TRACE_PATH=r'/home/zhijian/workspace_fast/Monax/traces/mahimahi/trace/subset'
# TRACE_PATH=r'data/throughput_traces/experiment_traces/040'
TRACE_INDEX='test_fcc_trace_1171_http---www.yahoo.com_0'

def new_trace(path, wpath, new_avg_throughput):
	time_all = []
	packet_sent_all = []
	last_time_stamp = 0
	packet_sent = 0
	stamp_sum_persec = []  # 每秒内发送pkt的时间戳
	stamp_persec = 0
	second = 0
	pkt_sum_persec = []  # 每秒发送包总数量
	pkt_persec = 0
	throughput_all_sec = []  # 按秒计算的throughput
	with open(path, 'rb') as f:
		for line in f:
			line1 = bytes.decode(line)
			time_stamp = int(line.split()[0])
			if time_stamp == last_time_stamp:
				packet_sent += 1
				continue
			else:
				# 进入下一个时间戳，时间戳加入time_all
				time_all.append(last_time_stamp)
				stamp_persec = stamp_persec + 1  # 当前second 时间戳数量加一
				packet_sent_all.append(packet_sent)
				pkt_persec = pkt_persec + packet_sent
				packet_sent = 1
				last_time_stamp = time_stamp
				if int(time_stamp / 1000) != second:  # 判断是否每秒最后一个时间戳，完成记录并更新
					stamp_sum_persec.append(stamp_persec)
					pkt_sum_persec.append(pkt_persec)
					pkt_persec = 0
					stamp_persec = 0
					gap = int(time_stamp / 1000) - second
					if gap > 1:  # 如果某一秒没有包，加0保持list大小与实际seconds一致
						for i in range(1, gap):
							stamp_sum_persec.append(0)
							pkt_sum_persec.append(0)
					second = int(time_stamp / 1000)
	# 最后一秒的数据
	time_all.append(last_time_stamp)
	stamp_persec = stamp_persec + 1
	packet_sent_all.append(packet_sent)
	pkt_persec = pkt_persec + packet_sent
	stamp_sum_persec.append(stamp_persec)
	pkt_sum_persec.append(pkt_persec)
	# print(len(stamp_sum_persec))
	# print(stamp_sum_persec)
	# print(pkt_sum_persec)
	my_sum_thr = 0
	for i in range(len(stamp_sum_persec)):
		throuput_sec = pkt_sum_persec[i] * PACKET_SIZE * BITS_IN_BYTE / MBITS_IN_BITS
		throughput_all_sec.append(throuput_sec)
		my_sum_thr = my_sum_thr + throuput_sec

	my_avg_thr = my_sum_thr / len(throughput_all_sec)
	time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
	throuput_all = PACKET_SIZE * \
				   BITS_IN_BYTE * \
				   np.array(packet_sent_all[1:]) / \
				   time_window * \
				   MILLISECONDS_IN_SECONDS / \
				   MBITS_IN_BITS
	sum_throughput = np.sum(throuput_all, axis=0)
	avg_throughput = sum_throughput / len(throuput_all)
	amount = avg_throughput / new_avg_throughput  # 减小比例
	amount2 = new_avg_throughput / avg_throughput  # 增大比例

	# print(avg_throughput)
	amount = int(amount + 0.5)
	amount2 = int(amount2+0.5)

	if amount >= 2:
		last_second = 0
		f1 = open(wpath, 'w')
		mylen = len(time_all)
		count = 0
		for i in range(mylen):
			mystamp = time_all[i]
			num_pkt = packet_sent_all[i]
			now_sec = int(mystamp / MILLISECONDS_IN_SECONDS)
			if now_sec != last_second:
				count = 0
				last_second = now_sec
			for j in range(num_pkt):
				count = count + 1
				if count % amount == 0:
					mystamp = str(mystamp)+'\n'
					f1.write(mystamp)
	elif amount2 >= 2:
		f1 = open(wpath, 'w')
		mylen = len(time_all)
		for i in range(mylen):
			mmystamp = time_all[i]
			mmystamp = str(mmystamp)+'\n'
			nnum_pkt = packet_sent_all[i]
			new_num_pkt = nnum_pkt * amount2
			for j in range(new_num_pkt):
				f1.write(mmystamp)
	else:
		change_throughput = abs(new_avg_throughput - avg_throughput)  # throughput改变量
		change_in_pkts = change_throughput * MBITS_IN_BITS / BITS_IN_BYTE / PACKET_SIZE  # change in pkts
		f1 = open(wpath, 'w')
		lentime = len(time_all)
		last_sec = 0
		stamp_count = 0
		for i in range(lentime):
			stamp = time_all[i]  # 时间戳
			sec = int(stamp / MILLISECONDS_IN_SECONDS)  # 确定当前时间戳对应秒数
			all_stamp = stamp_sum_persec[sec]  # 找到当前sec有几个时间戳发了包
			if all_stamp > 0:
				tochange = change_in_pkts / all_stamp  # 每个时间戳要调整包数量
			else:
				tochange = 0
			change_num = int(tochange)
			left = tochange - change_num  # 为了严格保证每秒发包数量改变相同数量
			if left > 0:
				change_num2 = int(1 / left)
			else:
				change_num2 = 10000
			if sec == last_sec:
				stamp_count = stamp_count + 1
				if stamp_count % change_num2 == 0:
					if new_avg_throughput > avg_throughput:

						pktnum = packet_sent_all[i] + change_num + 1  # 对应时间戳发送pkt
					else:
						pktnum = packet_sent_all[i] - change_num - 1
						if pktnum < 0:
							pktnum = 0
				else:
					if new_avg_throughput > avg_throughput:
						pktnum = packet_sent_all[i] + change_num
					else:
						pktnum = packet_sent_all[i] - change_num
			else:
				stamp_count = 0
				last_sec = sec
			stamp = str(stamp) + '\n'
			if(len(stamp)==1):
				continue
			if pktnum > 0:
				for j in range(pktnum):
					f1.write(stamp)


# if __name__ == "__main__":
#	for trace in os.listdir(TRACE_PATH):
#		if TRACE_INDEX in trace:
#			print(trace)
#			plot(TRACE_PATH + '\\' + trace)

if __name__ == "__main__":
	for trace in os.listdir(TRACE_PATH):
# 		if TRACE_INDEX == trace:
		if(trace[0:6]!='report'):
			continue
		print(trace)
		new_avg_throughput=15  # Mbps
		TRACE_FILE_PATH=TRACE_PATH + '/' + trace
		NEWTRACE_PATH= TRACE_PATH + '/results/'+ str(new_avg_throughput)+'Mb_'+ trace
		new_trace(TRACE_FILE_PATH, NEWTRACE_PATH, new_avg_throughput) 

		record = []
		with open(NEWTRACE_PATH) as f:
			for line in f:
				if(len(line.split())>0):
					record.append(line.split()[0])

		with open(NEWTRACE_PATH, 'w') as f:
			for i in record:
				f.write(i+'\n')
						
			
