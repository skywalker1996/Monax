import configparser
import os

class Config(object):
	def __init__(self, cfg_file):

		self.cfg_file = cfg_file
		self.config = configparser.ConfigParser()
		file_root = os.path.dirname(__file__)
		self.config_path = os.path.join(file_root, self.cfg_file)
		self.config.read(self.config_path)

	def get(self, section, key):
		return self.config.get(section, key)

	def sections(self):
		return self.config.sections()
	
	def write_sections(self, configs):

		new_config = configparser.ConfigParser()
		for section in configs:
			new_config[section] = configs[section]

		with open(self.config_path, 'w') as f:
			new_config.write(f)













	
		




	
	