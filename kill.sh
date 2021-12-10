kill -9 $(ps -aux | grep monax | awk '{print $2}')
