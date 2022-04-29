

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#pylint: disable=wrong-import-position
from configs.Common import *
from rate_control_Monax_new import rate_control_module_Monax


if __name__ == "__main__":
    RC_Module = rate_control_module_Monax(5, 30)
    rtt_record = [i for i in range(1,20)]
    packet_loss_record = [i/10 for i in range(1,20)]
    sending_rate = 10
    throughput_error = 0.5
    cwnd = 30

    state = { STATE_RTT: rtt_record[-10:],
            STATE_LOSS: packet_loss_record[-10:],
            STATE_SENDING_RATE: sending_rate, 
            STATE_THROUGHPUT_ERROR: throughput_error, 
            STATE_QUEUE_LENGTH: 0,
            STATE_CWND: cwnd}

    cwnd, utility, log_info=RC_Module.action(state)
    print(cwnd)