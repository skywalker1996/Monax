import numpy as np
from matplotlib import pyplot as plt

# class EMA
server_sending_rate = 10



def rising_sequence(start, end):
	return list(np.arange(start,end,2))

def falling_sequence(start, end):
	return list(np.arange(start,end,-2))


def gen_hml():
	seq_len = {}
	seq_len['high'] = 20
	seq_len['middle'] = 20
	seq_len['low'] = 20

	levels = {}
	levels['high'] = 2
	levels['middle'] = 1.1
	levels['low'] = 0.5
	Bandwidth_seq = []

	## 5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['high']+2*np.random.randn() for i in range(seq_len['high'])])

	## falling
	Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['high'],server_sending_rate*levels['middle']))

	## 1.1 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['middle']+2*np.random.randn() for i in range(seq_len['middle'])])

	## falling
	Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['low']))

	## 0.5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['low']+2*np.random.randn() for i in range(seq_len['low'])])

	## rising
	Bandwidth_seq+=(rising_sequence(server_sending_rate*levels['low'],server_sending_rate*levels['middle']))

	## 1.1 * server_sending_rate
	Bandwidth_seq+=[server_sending_rate*levels['middle']+2*np.random.randn() for i in range(seq_len['middle'])]

	## rising
	Bandwidth_seq+=rising_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['high'])

	return Bandwidth_seq



def gen_hl():
	seq_len = {}
	seq_len['high'] = 20
	seq_len['low'] = 20

	levels = {}
	levels['high'] = 2
	levels['low'] = 1
	Bandwidth_seq = []

	## 5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['high']+np.random.randn() for i in range(seq_len['high'])])

	## falling
	Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['high'],server_sending_rate*levels['low']))

	# ## 1.1 * server_sending_rate
	# Bandwidth_seq+=([server_sending_rate*levels['middle']+0.5*np.random.randn() for i in range(seq_len['middle'])])

	# ## falling
	# Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['low']))

	## 0.5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['low']+np.random.randn() for i in range(seq_len['low'])])

	## rising
	Bandwidth_seq+=(rising_sequence(server_sending_rate*levels['low'],server_sending_rate*levels['high']))

	# ## 1.1 * server_sending_rate
	# Bandwidth_seq+=[server_sending_rate*levels['middle']+0.5*np.random.randn() for i in range(seq_len['middle'])]

	# ## rising
	# Bandwidth_seq+=rising_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['high'])

	return Bandwidth_seq



def gen_hl_smooth():
	seq_len = {}
	seq_len['high'] = 20
	seq_len['low'] = 20

	levels = {}
	levels['high'] = 2
	levels['low'] = 0.5
	Bandwidth_seq = []

	## 5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['high'] for i in range(seq_len['high'])])

	## falling
	Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['high'],server_sending_rate*levels['low']))

	# ## 1.1 * server_sending_rate
	# Bandwidth_seq+=([server_sending_rate*levels['middle']+0.5*np.random.randn() for i in range(seq_len['middle'])])

	# ## falling
	# Bandwidth_seq+=(falling_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['low']))

	## 0.5 * server_sending_rate
	Bandwidth_seq+=([server_sending_rate*levels['low'] for i in range(seq_len['low'])])

	## rising
	Bandwidth_seq+=(rising_sequence(server_sending_rate*levels['low'],server_sending_rate*levels['high']))

	# ## 1.1 * server_sending_rate
	# Bandwidth_seq+=[server_sending_rate*levels['middle']+0.5*np.random.randn() for i in range(seq_len['middle'])]

	# ## rising
	# Bandwidth_seq+=rising_sequence(server_sending_rate*levels['middle'],server_sending_rate*levels['high'])

	return Bandwidth_seq

def gen_static():
	seq_len = 20
	Bandwidth_seq = []
	
	## 5 * server_sending_rate
	Bandwidth_seq+=([20 for i in range(seq_len)])
	
	

	return Bandwidth_seq

Bandwidth_seq = gen_static()
f = open('../traces/trace_static.txt', 'w+')
for bd in Bandwidth_seq:
	f.write(str(round(bd,2))+'\n')

f.close()

plt.figure(figsize=(15,10))
plt.title("demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(list(range(len(Bandwidth_seq))),Bandwidth_seq) 
plt.show()
