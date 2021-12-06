import time
from lib.monitorFlux import MonitorFlux
import cv2
from configs.config import Config

config = Config('global.conf')

db_para = {}
db_para['url'] = config.get('database', 'url')
db_para['token'] = config.get('database', 'token')
db_para['org'] = config.get('database', 'org')
db_para['bucket'] = config.get('database', 'bucket')


tags = {'version':0.1}

targets = []
targets.append({'measurement': 'server', 'field_name': 'network_delay', 'tags': tags, 'time_range': '10s'})
targets.append({'measurement': 'server', 'field_name': 'packet_loss', 'tags': tags, 'time_range': '10s'})
targets.append({'measurement': 'server', 'field_name': 'server_send_throughput', 'tags': tags, 'time_range': '10s'})
targets.append({'measurement': 'middleware', 'field_name': 'queue_size', 'tags': tags, 'time_range': '10s'})
targets.append({'measurement': 'middleware', 'field_name': 'midware_recv_throughput', 'tags': tags, 'time_range': '10s'})

# targets.append({'measurement': 'state_monitor', 'field_name': 'utility', 'tags': None, 'time_range': '30s', 'reset':False })
# targets.append({'measurement': 'state_monitor', 'field_name': 'recv_buffer_size', 'tags': None, 'time_range': '30s', 'reset':False })
# targets.append({'measurement': 'state_monitor', 'field_name': 'server_sending_delay', 'tags': None, 'time_range': '30s', 'reset':False })


Monitor = MonitorFlux(db_para=db_para, plot_number=5, fig_size=(15,15),monitor_targets=targets)

# results = Monitor.pullData(measurement='state_monitor',tags=None, time_range='5d')
# print(results)

running = True
while running:
	if cv2.waitKey(1) & 0xFF == ord('q'):
		running = False
		continue
	Monitor.updateMonitor()
	time.sleep(0.2)

