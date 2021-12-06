class PiecePack(object):
	def __init__(self, frame_id, piece_id, ctime, piece):
		self.frame_id = frame_id
		self.piece_id = piece_id
		self.ctime = ctime
		self.piece = piece

	def __lt__(self, other):
		return self.ctime > other.ctime


class PacketLossTracker(object):
	def __init__(self, piece_num, Loss_log_period):
		self.frame_log = []
		self.packet_loss = 0
		self.piece_num = piece_num  #piece_num是一帧图像切分成片的数量
		self.packet_total = 0
		self.Loss_log_period = Loss_log_period
		# self.packet_total = self.piece_num * self.Loss_log_period


	def update_log(self, count):

		self.packet_loss += (self.piece_num-count)
		self.packet_total += self.piece_num
		return 


	def clear(self):
		self.packet_loss = 0
		self.packet_total = 0


	def get_current_loss(self):
		# print('packet_total = {} while packet_loss = {}'.format(self.packet_total, self.packet_loss))
		loss_rate = (self.packet_loss / self.packet_total) if self.packet_total!=0 else 0.0
		if(type(loss_rate)==int):
			loss_rate = float(loss_rate)
		return round(loss_rate,5)
