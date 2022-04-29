import numpy as np

from configs.Common import *

def utility_function(vector,RTT_threshold,RTT_prediction):
    sr_weight = 0.8
    sr_weight_min = 0.7
    sr_weight_max = 0.9
    sr_weight_stepsize = 0.01

    if(vector[4]>1):
        if(sr_weight>sr_weight_min-sr_weight_stepsize):
            sr_weight -= sr_weight_stepsize
    if(vector[2]-RTT_threshold<-10):
        if(sr_weight<sr_weight_max+sr_weight_stepsize):
            sr_weight += sr_weight_stepsize
#         print(sr_weight)


    weights=[sr_weight,-0.9,-0.01,-0.4,0.001,0.001]
    # sample=[sending_rate,network_delay_gradient,monitor_para['network_delay'][-1],current_loss]
    
    sending_rate_part = vector[0] * weights[0]
    RTT_g_part = weights[1] * vector[1]*vector[0]
    RTT_exceed_part = weights[2]* np.maximum(vector[2]-RTT_threshold, 0)*vector[0]
    loss_part = weights[3] * vector[3]*vector[0]
    RTT_max_part=weights[2] * vector[2]*vector[0]
    if (RTT_prediction==True):
        RTT_prediction_part=weights[4]*vector[4]*vector[0]+weights[5]*vector[5]*vector[0]
    
    if((vector[2]-RTT_threshold) > 0):        
        utility =RTT_g_part+ loss_part
#             print("slow down mode")
        if (RTT_prediction==True):
            utility +=RTT_prediction_part
    else:
        # print("speed up mode")
        utility = sending_rate_part+loss_part
        if (RTT_prediction==True):
            utility +=RTT_prediction_part
    return utility


# def utility_function(state,e2e_delay_threshold):

#     sending_rate = state[STATE_SENDING_RATE]
#     rtt = state[STATE_RTT]
#     loss = state[STATE_LOSS]
#     throughput_err = state[STATE_THROUGHPUT_ERROR]
#     ack_t = state[STATE_ACK_TIMESTAMP]
#     queue_delay = state[STATE_QUEUE_DELAY]
#     e2e_delay = state[STATE_E2E_DELAY]

#     pre_rtt_avg = sum(state[STATE_RTT][-4:-1])/3
#     # label = self.history_decision[-1] if monitor_para[STATE_RTT][-1]<pre_rtt_avg else 1- self.history_decision[-1]
#     label = 1 if state[STATE_RTT][-1]<pre_rtt_avg else 0


#     rtt_g_list=[(rtt[-3]-rtt[-4])/(rtt[-4]),
#                 (rtt[-2]-rtt[-3])/(rtt[-3]), 
#                 (rtt[-1]-rtt[-2])/(rtt[-2])]

#     sr_g_list = []
    

#     weights = {
#         STATE_SENDING_RATE: 0.1,
#         STATE_RTT:-1, 
#         STATE_LOSS:-0.1,
#         STATE_RTT_GRADIENT:-0.5,
#         STATE_THROUGHPUT_ERROR:-0.1
#     }
    
#     # sending_rate_part = weights[STATE_SENDING_RATE] * sending_rate[-1]
#     # rtt_gradient_part = weights[STATE_RTT_GRADIENT] * rtt_g_list[-1]
#     # loss_part = weights[STATE_LOSS] * loss[-1]
#     # rtt_part = weights[STATE_RTT] * rtt[-1]


#     rtt_gradient_part = sum(rtt_g_list)/3
 
#     if((e2e_delay[-1]-e2e_delay_threshold) > 0):        
#         utility = -1 * (sum(rtt_g_list)/3)
# #             print("slow down mode")
#     else:
#         # print("speed up mode")
#         utility = sending_rate[-1]
#     return utility
