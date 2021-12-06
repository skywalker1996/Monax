import sys 
import os.path as osp
sys.path.append(osp.abspath(osp.dirname(osp.dirname(__file__))))
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import datetime
import time

class InfluxOP(object):


	def __init__(self, url, token, org, bucket):

		"""
		input:
			host: ip address of the influxdb service
			port: port of the influxdb service, 8086 in common
			username & password: influxdb authentication params
		"""
		self.client = InfluxDBClient(url=url, token=token, org=org)
		self.write_api = self.client.write_api(write_options=ASYNCHRONOUS)
		self.query_api = self.client.query_api()
		self.bucket = bucket
		self.org = org


	def pushData(self, measurement, datapoints, tags=None):
		"""
		Push data piece to the database

		input: database: destination database  
			   measurement: destination sheet
			   datapoints: datapoints' list to push
			   (each data point is a json-style dictionary, all the points should be packed into a list)
			   [
			   		{"field_name_1": field_value_1, "field_name_2": field_value_2}
			   ]
			   tags: tags for the datapoints, default is None
			   {
               		"user": "Carol",
                	"brushId": "6c89f539-71c6-490d-a28d-6c5d84c0ee2f"
         	   }

         	   ***Notice: these datapoints will be assigned with the same time(which is current time).
        output: True if success, else False
		"""
		try:
			formed_datapoints = self.formDataPoints(measurement, datapoints, tags)
			# print(formed_datapoints)
			self.write_api.write(self.bucket, self.org, formed_datapoints)
			return True
		except Exception as e:
			print('======> Write Error: ',e)
			return False



	def pullData(self, measurement, field, tags=None, time_range='30s'):
		"""
		pull data from the database

		input: database: database name
			   measurement: sheet name
			   tags: tag filter, using python dictionary, example: {'version':'0.3'}
			   time_range: only pull date between now()-time_range and now(), default is '30m'
			   				`ns` nanoseconds  
							`us or µs` microseconds  
							`ms`   milliseconds  
							`s`   seconds     
							`m`   minutes   
							`h` hours   
							`d`  days   
							`w`  weeks  
		ouput: a list of results, every element is a dictionary for one data point									
		"""
		try:
			query_sent = 'from(bucket:"%s") |> range(start: -%s)'%(self.bucket,time_range)
			if(field==None):
				query_sent+='|> filter(fn: (r) => r._measurement == "%s")'%(measurement)
			else:
				query_sent+='|> filter(fn: (r) => r._field == "%s")'%(field)
				query_sent+='|> filter(fn: (r) => r._measurement == "%s")'%(measurement)
											
			if(tags != None):
				tag_filter = ''
				for tag in tags.keys():
					tag_filter+='|> filter(fn: (r) => r.%s == "%s")'%(tag, tags[tag])

				query_sent += tag_filter
				tables = self.query_api.query(query_sent)

			else:
				tables = self.query_api.query(query_sent)

			results = []

			for i in range(len(tables[0].records)):
				result = {}
				result['time'] = str(tables[0].records[i].values['_time'])
				for table in tables:
					result[table.records[i].values['_field']] = table.records[i].values['_value']
				results.append(result)
			return results
				
		except Exception as e:
			print('======> Database Error: ',e)
			return []


	def formDataPoints(self, measurement, datapoints, tags):

		formed_datapoints = []
		for datapoint in datapoints:
			dataPack = {}
			if('time' in datapoint):
				dataPack['time'] = datapoint['time']
				datapoint.pop('time')
			else:			
				dataPack['time'] = datetime.datetime.utcnow().isoformat()

			dataPack['measurement'] = measurement
			dataPack['fields'] = datapoint
			if(tags is not None):
				dataPack['tags'] = tags

			formed_datapoints.append(dataPack)
		return formed_datapoints




if __name__ == '__main__':


	url = "http://192.168.2.171:9999"
	token = "t-_unv70lBaFM07aMINElDG6i8rY_wCibTa-754_EfIXyhQSbVogaomASNoZey3pqoDLZMXRARdVycVNaaa-1Q=="
	org="xjtu"
	bucket="GTS"
	measurement = "monax-server_tc"

	client = InfluxOP(url=url, token=token, org=org, bucket=bucket)
	
	demo = [{'network_delay': 44, 'packet_loss': 0.02, 'sending_queue_delay': 340, 'time':datetime.datetime.utcnow().isoformat()}]

	tags = {'version':'0.1'}

	#tags = None

	# client.pushData(measurement=measurement, datapoints=demo, tags=tags)
	# time_range='60m'
	field = 'network_delay'
	tags = {'version':0.1}
	print('start pull')
	results = client.pullData(measurement=measurement, field=None, tags=None, time_range='30m')
	print(results)

