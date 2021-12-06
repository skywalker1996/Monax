import numpy as np

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