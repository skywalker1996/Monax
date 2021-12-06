import sys 
import os.path as osp
print(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from lib.influx_operator import InfluxOP
import pandas as pd


# url = "http://39.98.74.212:9999"
# token = "tjqxBljXpOtdT5LlPuf8_Os3WwkNandhdv7C4SyIdvVRSUUiarYgG7SufMciV4qxe6lmtVlNdoUUGqyrCDoWIg=="
# org="xjtu"
# bucket="GTS"
# measurement = "monax-server_tc"

# client = InfluxOP(url=url, token=token, org=org, bucket=bucket)

#tags = None

# client.pushData(measurement=measurement, datapoints=demo, tags=tags)
# time_range='60m'

def PullData(db_para, time):
	
	client = InfluxOP(url=db_para["url"], token=db_para["token"], org=db_para["org"], bucket=db_para["bucket"])
	field = 'network_delay'
	tags = {'version':0.1}
	print('start pull')
	results = client.pullData(measurement=db_para["measurement"], field=None, tags=None, time_range=time)
	print('pull successfully!')

	network_delay = [res['network_delay'] for res in results]
	print(len(network_delay))


	packet_loss = [res['packet_loss'] for res in results]
	print(len(packet_loss))

	server_sending_rate = [res['server_sending_rate'] for res in results]
	

	dataframe = pd.DataFrame({'network_delay':network_delay, 'packet_loss':packet_loss, 'server_sending_rate':server_sending_rate})
# 	if(os.)
	dataframe.to_csv("record/0822_1720.csv", sep=',', mode='w+')

	return 

# data = pd.read_csv("../data/server.csv")
# print(data.values[0:10])
# PullData()