import numpy as np
import pickle
import time

class H264_Stream:

	## i frame: 2
	## b frame: 1
	## p frame: 0
	def __init__(self, video_path):
		self.frame_loss_rate=0.05
		self.packet_loss_rate=0.05
		self.data_split=[]
		self.current_slice_type_convention=2
		self.loss_flag=0

		self.frame_id = 0
		self.SliceTypeMap = {'P':0, 'B':1, 'I':2}

		with open (video_path,'rb') as f:
			self.video_buffer = pickle.load(f)
		self.send_index = 0
		self.start_time = time.time()

		self.last_priority = 0


	def getNextPacket(self):

		if(self.send_index==len(self.video_buffer)):
			self.send_index = 0
			self.start_time = time.time()
		data = self.video_buffer[self.send_index][1]
		timestamp = self.video_buffer[self.send_index][0]
		if(time.time()-self.start_time<timestamp):
			return None
		self.send_index+=1

		return data, self.frame_id, self.cal_level(data)

#drop the whole slice
	def cal_level(self,data):
		res_NALU = self.analysing_NALU_header(data)
		if(res_NALU and res_NALU['NAL_unit_type']<=5): 
			res_slice = self.analysing_slice_header(res_NALU['NALU_data'])
			self.current_slice_type_convention=res_slice["slice_type_convention"]
			### new frame
			if(res_slice['first_mb_in_slice'] == 0):
				self.frame_id+=1
				if(res_slice["slice_type_convention"]==self.SliceTypeMap['I']):
					self.last_priority = 3
					return 3
				elif(res_slice["slice_type_convention"]==self.SliceTypeMap['P']):
					self.last_priority = 2
					return 2
				elif(res_slice["slice_type_convention"]==self.SliceTypeMap['B']):
					self.last_priority = 1
					return 1

		return self.last_priority


#analyze the NALU header to get the information of type of frame
	def analysing_NALU_header(self,data):
		res = {}
		for index in range(len(data)-2):
			if(data[index]==0x00 and data[index+1]==0x00 and data[index+2]==0x01):
				if(data[index-1]==0x00):
					res["slice_id"]=0 #this slice is the first slice of the frame
				data_body = data[index+3:]
				binary_data=bin(data_body[0])[2:].zfill(8) #str
				res = {}
				res["forbidden_bit"]=int(binary_data[0],2) #0 if right else 1
				res["NRI"]=int(binary_data[1:3],2) #IDR frame:3, I frame and P frame:2, B frame:0
				res["NAL_unit_type"]=int(binary_data[3:],2) #5 if IDR frame, 1 if not
				res["NALU_data"]=data_body[1:]
				if (res["NAL_unit_type"]==5):
					res["IDR_frame"]=1
				elif (res["NAL_unit_type"]==1):
					res["IDR_frame"]=0
				else: 
					#if NAL_unit_type=='7' is SPS, self.NAL_unit_type=='8' is PPS
					#if not, enter into the RTSP part.
					return res
				return res
		return None

#from binary encoder to columb encoder and vice verse which are used in slice and macroblock levels
	def Exp_Columb_Decoder(self,data):
		binary_data=bin(data)[2:].zfill(8)
		count=0
		this_data_left=0
		res = []
		while (this_data_left+count<len(binary_data)):
			if (binary_data[this_data_left+count]=='0'):
				count+=1
			else:
				element=binary_data[this_data_left+count+1:min(len(binary_data),this_data_left+count*2+1)]
				element_in_ten=int(element,2)+2**count-1 if element!='' else 2**count-1
				res.append(element_in_ten) #this_data_length=2*count+1
				this_data_left+=count*2+1
				count=0
		return res

#analyze the slice header to get the information of type of slice
	def analysing_slice_header(self,data):
		res = {}
		result = self.Exp_Columb_Decoder(data[0])
		res["first_mb_in_slice"] = result[0]
		res["slice_type"] = result[1]

		 # classify the I B and P slice
		if (res["slice_type"]==5 or res["slice_type"]==0):
			res["slice_type_convention"]=0  ## P slice
		elif (res["slice_type"]==6 or res["slice_type"]==1):
			res["slice_type_convention"]=1  ## B slice
		elif (res["slice_type"]==7 or res["slice_type"]==2):
			res["slice_type_convention"]=2  ## I slice
		#self.pic_paramer_set_id=self.data_split[2].... according to the information of log2_max_frame_num_minus4 in SPS
		#  http://guoh.org/lifelog/2013/10/h-264-bit-stream-sps-pps-idr-nalu/

		return res


#analyze the slice body to get the information of type of macroblock
	'''def slice_body(self):
		self.Exp_Columb_Encoder(self.netdata[0])'''
