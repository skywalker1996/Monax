from rate_control_FTRL import *

RC_Module = rate_control_module_FTRL(5, 0.2, 1, 0.2, 0.2,100)






state = {'network_delay': [10,20,30,40,50,60,70,80,90,100],
		  'packet_loss': [0.1,0.2,0.3,0.4,0.5,0.1,0.2,0.3,0.4,0.5],
		  'sending_rate': 15.5, 
		  'throughput_error': 2.3, 
		  'queue_length': 150,
		  'cwnd': 100}

cwnd, utility, log_info= RC_Module.action(state)

print(cwnd, utility, log_info)