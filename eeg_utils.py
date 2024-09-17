import os
import mne
import pickle

class EEGUtils:
	"""
	这个类包含获取EEG目录的实用工具方法
	"""
	
	@staticmethod
	def get_dir(base_dir):
		"""
		获取EEG文件的所有路径
		:param base_dir: EEG基础目录  '../Per_Process/Per_Process_1/sub'
		:return: 返回所有EEG文件路径的字典
		"""
		dir_eeg = {}
		for sub_number in os.listdir(base_dir):
			sub_dir = os.path.join(base_dir, sub_number)
			for session in os.listdir(sub_dir):
				eeg_dir = os.path.join(sub_dir, session, 'eeg')
				for eeg_task_name in [f for f in os.listdir(eeg_dir) if os.path.isfile(os.path.join(eeg_dir, f))]:
					eeg_task_dir_name_without_suffix = eeg_task_name.split(".")[0]
					dir_eeg[f"{sub_number}_{session}_{eeg_task_dir_name_without_suffix}"] = os.path.normpath(
						os.path.join(eeg_dir, eeg_task_name))
		return dir_eeg


class EEGProcessor:
	"""
	这个类负责加载和滤波EEG数据
	"""
	
	def __init__(self, file_path):
		"""
		初始化类，file_path是具体的文件路径
		"""
		self.file_path = file_path
		self.raw_data = None
		self.filtered_data = {}
		self.load_data()
	
	def load_data(self):
		"""
		读取EEG数据。如果读取raw_fif失败，则尝试读取epochs。
		"""
		try:
			self.raw_data = mne.io.read_raw_fif(self.file_path, preload=True)
		except:
			try:
				self.raw_data = mne.read_epochs(self.file_path, preload=True)
			except:
				raise RuntimeError(f"读取文件失败，既无法读取 raw fif 数据，也无法读取 epochs 数据：{self.file_path}")
	
	def filter_data(self, freq_bands):
		"""
		对EEG数据进行滤波，并按传入的频带分开保存。
		:param freq_bands: 频带字典，包含多个频段的上下限
		"""
		if self.raw_data is None:
			print("未加载数据，无法滤波。")
			return
		
		# 对每个频带进行滤波，并保存到filtered_data字典中
		for band, (l_freq, h_freq) in freq_bands.items():
			self.filtered_data[band] = self.raw_data.copy().filter(l_freq=l_freq, h_freq=h_freq)
			print(f"已成功滤波 {band} 频带，范围为: {l_freq} Hz - {h_freq} Hz")
	
	def get_filtered_data(self, band_name):
		"""
		获取指定频带的滤波后的数据。
		:param band_name: 频带名称
		:return: 返回滤波后的EEG数据
		"""
		if band_name not in self.filtered_data:
			print(f"{band_name} 频带尚未滤波，请先执行 filter_data 方法。")
			return None
		return self.filtered_data[band_name]


class DEDataLoader:
	'''
	这个类可以加载所有pkl数据并且放在一个变量里
	
	base_dir = '../Result_DE/sub'
	data_loader = DEDataLoader(base_dir)  # 实例化并加载所有 .pkl 文件
	all_data = data_loader.de_data
	
	'''
	def __init__(self, base_dir):
		"""
		初始化类，遍历目录并加载所有 .pkl 文件的数据
		:param base_dir: 包含 .pkl 文件的目录路径
		"""
		self.base_dir = base_dir
		self.de_data = {}  # 用于存储所有加载的数据
		
		self.load_all_de_files()
	
	def load_all_de_files(self):
		"""
		加载目录中的所有 .pkl 文件数据，并以文件名作为属性存储
		"""
		dir_de = EEGUtils.get_dir(self.base_dir)  # 获取 .pkl 文件路径
		
		for file_name, file_path in dir_de.items():
			# 读取 .pkl 文件
			with open(file_path, 'rb') as f:
				data = pickle.load(f)
				# 使用文件名作为属性保存数据
				self.de_data[file_name] = data
	
	def get_data(self, file_name):
		"""
		根据文件名获取加载的数据
		:param file_name: .pkl 文件名
		:return: 对应文件的数据
		"""
		return self.de_data.get(file_name, None)  # 返回特定文件名的数据
