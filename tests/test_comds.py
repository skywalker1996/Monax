from subprocess import Popen

start_server = "mm-delay 10 mm-link /home/zhijian/workspace_fast/Monax/traces/mahimahi/trace/subset/report_bus_0001.log /home/zhijian/workspace_fast/Monax/traces/mahimahi/trace/subset/report_bus_0001.log --uplink-queue=droptail --uplink-queue-args=packets=2048  'ls -al'"

server = Popen(start_server, shell=True)
