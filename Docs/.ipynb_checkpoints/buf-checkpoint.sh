
# for name in "$@"
# do
# 	echo $name
# 	if [ $name = "mn" ]; then
# 		sudo mn -c
# 	else
# 		sudo kill 9 $(ps -aux | grep server.py | awk '{print $2}')
# 		sudo kill 9 $(ps -aux | grep client.py | awk '{print $2}')
# 		sudo kill 9 $(ps -aux | grep visualization.py | awk '{print $2}')
# 		sudo mn -c
# 	fi

# done