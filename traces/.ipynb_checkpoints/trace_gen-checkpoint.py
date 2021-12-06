import random

### 输入：input_file:输入带宽文件路径
###       output_file:输出mahimahi trace路径
###      interval: 带宽文件的采样间隔，单位为 秒

def trace_gen(input_file, output_file, interval):
    PACKET_SIZE = 1500.0  # bytes
    BITS_IN_BYTE = 8.0
    MBITS_IN_BITS = 1024**2
    MILLISECONDS_IN_SECONDS = 1000.0

    band_list = []
    with open(input_file,'r') as f:
        band = f.readline()
        while(band):
            band_list.append(float(band.replace('\n', '').strip()))
            band = f.readline()
    mahimahi_trace = []
    
    ## 每个带宽值持续的period(ms)
    period_len = int(MILLISECONDS_IN_SECONDS * interval)
    period_start = 0
    for band in band_list:
        bitsPerPeriod = int(band*interval*(10**6))
        bytesPerPeriod = int(bitsPerPeriod/8)
        packetsPerPeriod = int(round(bytesPerPeriod/PACKET_SIZE))
        mahimahi_trace+=(sorted([random.randint(period_start,period_start+period_len-1) for _ in range(packetsPerPeriod)]))
        period_start+=period_len
        
    
    with open(output_file, "w+") as f:
        for d in mahimahi_trace:
            f.write(str(d)+'\n')
            
    print('生成成功！')
            

trace_gen('trace1.txt', 'mahimahi.up', 0.034)