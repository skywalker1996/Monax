import sys 
import os.path as osp
sys.path.append(osp.abspath(osp.dirname(osp.dirname(__file__))))
from lib.influx_operator import InfluxOP
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import time 
from threading import Thread, Lock
import matplotlib.font_manager as font_manager
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import dateutil.parser
from datetime import datetime, timezone

def UTCStr_to_LocalDate(UTC_str):
	"""
		switch UTC time to local timezone
		input: iso utc time string, ("2020-02-26T08:23:57.862296")
		output: iso time string (local timezone)
	"""
	utc_date = dateutil.parser.parse(UTC_str)
	local_date = utc_date.replace(tzinfo=timezone.utc).astimezone(tz=None)
	local_str = local_date.strftime("%Y-%m-%d %H:%M:%S.%f")
	return dateutil.parser.parse(local_str)

class MonitorFlux(object):

	def __init__(self, db_para, plot_number = None, fig_size = None, monitor_targets=None):

		self.DBoperator = InfluxOP(db_para["url"], db_para["token"], db_para["org"], db_para["bucket"])

		self.monitor_targets = monitor_targets

		if(monitor_targets):
			self.fig_count = 0
			self.monitor_obj = []
			self.ax_list = {}
			self.line_list = {}
			self.subplot_width = (int(plot_number**(1/2))+1, int(plot_number**(1/2))+1)
			self.figsize = fig_size
			self.monitor = Thread(target=self.initMonitor, args=()) 
			self.monitor.daemon = True
			self.monitor.start()
			self.plot_ready = False

	def updateMonitor(self):
		# print('update plot!!')
		if(not self.plot_ready):
			return
		for obj in self.monitor_obj:
			results = self.DBoperator.pullData(measurement=obj['measurement'], field=obj['field'],tags=obj['tags'], time_range=obj["time_range"])
			if(len(results)==0):
				print("what the fuck!!")
				continue
			x = [UTCStr_to_LocalDate(data['time']) for data in results]
			y = [data[obj['field']] for data in results]
			self.plot_update_module(obj['field'], x, y)
			plt.xticks(rotation=70)

		plt.draw()


	def initMonitor(self):

		fig = plt.figure(figsize=self.figsize)
		for target in self.monitor_targets:
			self.createFigure(target["measurement"], target["field_name"], tags=target['tags'], time_range=target['time_range'])
			plt.xticks(rotation=70)

		self.plot_ready = True
		
		plt.show()


	def createFigure(self, measurement, field_name, tags=None, time_range='1m'):
		self.fig_count += 1
		self.monitor_obj.append({'measurement': measurement, 'field': field_name, 'tags': tags, 'time_range': time_range})

		fig_loca = str(self.subplot_width[0])*2 + str(self.fig_count)
		print('create fig loca: ', int(fig_loca))

		results = self.DBoperator.pullData(measurement=measurement, field=field_name, tags=tags, time_range=time_range)
		if(len(results)>0):
			x = [UTCStr_to_LocalDate(data['time']) for data in results]
			y = [data[field_name] for data in results]
		else:
			x = [datetime.now()]
			y = [0]


		self.plot_initialize_module(loca=int(fig_loca), name=field_name, x=x, y=y, label=field_name)
		

	#initialize the figure
	def plot_initialize_module(self, loca, name, x, y, label):
		self.ax_list[name] = plt.subplot(loca)
		if(len(y)>0):
			self.ax_list[name].set_ylim([0, int(max(y)*2)])
			self.ax_list[name].set_xlim([x[0], x[-1]])
		self.ax_list[name].set_autoscale_on(False)
		self.ax_list[name].grid(True)
		self.ax_list[name].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))  
		self.line_list[name], = self.ax_list[name].plot(x, y, label=label, color='cornflowerblue')
		self.ax_list[name].legend(loc='upper center', ncol=4, prop=font_manager.FontProperties(size=10))

	def plot_update_module(self, name, x, y):

		self.line_list[name].set_ydata(y)
		self.line_list[name].set_xdata(x)
		self.ax_list[name].set_xlim([x[0], x[-1]])
		self.ax_list[name].set_ylim(0, max(y)*2)


	def pushData(self, measurement, datapoints, tags=None):

		return self.DBoperator.pushData(measurement, datapoints, tags)

	def pullData(self, measurement, field, tags=None, time_range='30m'):

		return self.DBoperator.pullData(measurement, field, tags, time_range)


