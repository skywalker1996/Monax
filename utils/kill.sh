if [ $1 = "db" ]; then
	sudo kill $(ps -aux | grep influx | awk '{print $2}')
else
	sudo kill $(ps -aux | grep monax_server.py | awk '{print $2}')
	sudo kill $(ps -aux | grep monax_client.py | awk '{print $2}')
    sudo kill $(ps -aux | grep monax_middleware.py | awk '{print $2}')
	sudo kill $(ps -aux | grep monax_visualization.py | awk '{print $2}')

sudo mn -c
fi