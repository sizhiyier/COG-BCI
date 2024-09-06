import csv

import mne
import numpy as np
from mne.preprocessing import find_bad_channels_lof, annotate_muscle_zscore, ICA, create_ecg_epochs
import os
from mne_icalabel import label_components


def find_folders_containing_string(directory, search_strings):
	"""
	遍历指定目录及其子目录，筛选出名称中包含特定字符串的所有子目录。

	:param directory: 要搜索的目录路径（字符串）
	:param search_strings: 要搜索的字符串列表（子目录名称中必须包含其中任意一个字符串）
	:return: 包含符合条件的子目录名称的列表
	"""
	matching_folders = []
	search_strings = [s.upper() for s in search_strings]  # 将所有搜索字符串转换为大写
	a = os.walk(directory)
	
	# 遍历目录树
	for root, dirs, files in os.walk(directory):
		# 检查当前目录的名字是否包含目标字符串
		for filename in files:
			if filename.endswith('set'):
				# 检查文件名是否包含任意一个搜索字符串
				if any(s in filename.upper() for s in search_strings):
					relative_path = os.path.join(root, filename)
					matching_folders.append(relative_path)
	
	return matching_folders
	



def insert_before_nth_from_end(original_string, insert_string, n_from_end):
	# 计算插入位置
	position = len(original_string) - n_from_end
	
	# 确保插入位置在字符串的有效范围内
	if position < 0:
		position = 0
	elif position > len(original_string):
		position = len(original_string)
	
	# 使用切片插入字符串
	new_string = original_string[:position] + insert_string + original_string[position:]
	return new_string

class Per_Process:
	def __init__(self):
		self.record_dict = {}
		self.if_baseline = True
	
	def update_record(self, key, value):
		"""
		更新记录字典。如果key存在，则将value追加到该key对应的值列表中；
		如果key不存在，则创建一个新的key，并将value作为列表中的第一个元素。

		Parameters:
		key : str
			要更新或创建的键
		value : any
			要追加或设置的值
		"""
		if key in self.record_dict:
			# 如果key存在，追加value到对应的值列表中
			if isinstance(self.record_dict[key], list):
				self.record_dict[key].append(value)
			else:
				# 如果存在的值不是列表，转换为列表并追加
				self.record_dict[key] = [self.record_dict[key], value]
		else:
			# 如果key不存在，创建一个新的键，并将value作为列表中的第一个元素
			self.record_dict[key] = [value]
		
	def save_to_csv(self):
		file_path = 'record.csv'  # 文件名固定为 record.csv
		with open(file_path, 'w', newline='') as file:
			writer = csv.writer(file)
			for key, values in self.record_dict.items():
				writer.writerow([key] + values)
	
	def get_dir(self):
		dir = {}
		sub_dirs = os.listdir('../sub')
		for sub_number in sub_dirs:
			sub_dir = '../sub/' + sub_number
			sessions = os.listdir(sub_dir)
			for session in sessions:
				sub_session_egg_dir = sub_dir + '/' + session + '/eeg'
				sub_session_egg_task_dir_lists = find_folders_containing_string(sub_session_egg_dir, ['matb','back'])
				for sub_session_egg_task_dir in sub_session_egg_task_dir_lists:
					dir[sub_number + '_' + session + '_' + sub_session_egg_task_dir.split("\\")[1][:-4]] = sub_session_egg_task_dir
		return dir
	
	def get_data(self):
		return self.get_dir()
	
	def drop_channel(self, raw):
		if 'Cz' in raw.info['ch_names']:
			raw.drop_channels('Cz')
		return raw
	
	def filter_data(self, raw):
		raw.filter(l_freq=1, h_freq=100, picks='eeg')
		raw.notch_filter(freqs=[48, 52], picks='eeg')
		
		raw.filter(l_freq=0.04, h_freq=40, picks='ecg')
		return raw
	
	def drop_bad_channels(self, raw):
		# 选取坏导
		bad_channels = find_bad_channels_lof(raw, picks='eeg')
		raw.info['bads'].extend(bad_channels)
		raw.interpolate_bads()
		
		self.update_record('bad_channels', len(bad_channels))
		return raw
	
	def drop_bad_period(self, raw):
		
		raw_copy = raw.copy()
		# 标记肌电伪迹
		annot_muscle, scores_muscle = annotate_muscle_zscore(
			raw,
			ch_type='eeg',
			filter_freq=(1, 100)
		)
		raw.set_annotations(annot_muscle)

		# 获取注释和时间
		annot = raw.annotations
		bad_intervals = [(annot.onset[i], annot.onset[i] + annot.duration[i])
		                 for i in range(len(annot)) if annot.description[i] == 'BAD_muscle']
		
		# 创建掩码以排除坏段
		mask = np.ones(len(raw.times), dtype=bool)
		for start, end in bad_intervals:
			mask[(raw.times >= start) & (raw.times < end)] = False
		
		# 过滤数据
		data_clean = raw.get_data()[:, mask]
		
		# 重新创建 Raw 对象
		info = raw.info
		raw_clean = mne.io.RawArray(data_clean, info)
		
		# 提取注释信息
		
		bad_durations = np.sum(annot.duration)
		


		self.update_record('total_bad_duration', str(bad_durations))

		return raw_clean
	def reference_data(self, raw):
		return raw.set_eeg_reference(ref_channels='average', ch_type='eeg')
	
	def ica(self,raw):
		# method='picard',

		ica = ICA(method='infomax', fit_params=dict(extended=True),max_iter="auto", random_state=97)
		ica.fit(raw)
		# scores = slope_score * focus_score * smoothness_score
		muscle_idx, muscle_idx_scores = ica.find_bads_muscle(raw, threshold=0.9)

		ecg_idx, ecg_scores = ica.find_bads_ecg(raw,ch_name='ECG1')
		ic_labels = label_components(raw, ica, method="iclabel")
		eog_idx = [
			idx for idx, (prob, lbl) in enumerate(zip(ic_labels['y_pred_proba'], ic_labels['labels']))
			if lbl in ["eye blink"] and prob > 0.9
		]
		print('*'*50)
		# print('muscle_idx: ',len(muscle_idx))
		print('ecg_idx: ',len(ecg_idx))
		print('eog_idx: ',len(eog_idx))
		combined_set =  set(ecg_idx) | set(eog_idx)
		print('total:', len(combined_set))
		print('*'*50)
		ica.exclude = list(combined_set)
		ica.apply(raw)
		
		self.update_record('muscle_idx: ', len(muscle_idx))
		self.update_record('ecg_idx: ', len(ecg_idx))
		self.update_record('eog_idx: ', len(eog_idx))
		self.update_record('total_idx: ', len(combined_set))
		
		return raw
	
	def extract_data(self, raw, file_name):
		if file_name.startswith('zero'):
			event_ids = ["6021", "6022"]
		if file_name.startswith('one'):
			event_ids = ["6121", "6122"]
		if file_name.startswith('two'):
			event_ids = ["6221", "6222", '6223']
		
		# 从注释中提取事件
		events, event_id_map = mne.events_from_annotations(raw)
		include_id = [event_id_map.get(id) for id in event_ids]
		events_taril = mne.pick_events(events, include=include_id)
		if self.if_baseline:
			epochs = mne.Epochs(raw, events_taril, tmin=-0.2, tmax=2.3,baseline=None, preload=True)
		else:
			epochs = mne.Epochs(raw, events_taril, tmin=0, tmax=2.5, baseline=None, preload=True)
		
		return epochs
	
	def run(self):

		data_dir = self.get_data()
		for key, value in data_dir.items():

			file_dir = value
			
			if file_dir.split('\\')[-1].startswith("MATB"):
				self.task = 'matb'
			else:
				self.task = 'back'
			
			raw = mne.io.read_raw_eeglab(file_dir, preload=True)

			raw.set_channel_types({'ECG1': 'ecg'})
			
			dir = os.path.normpath(value).replace('set', 'fif').split(os.sep)
			path = os.path.join('./Per_Process_3/', *dir[2:-2])
			if not os.path.exists(path):
				os.makedirs(path)
			new_file_name = os.path.join(path, dir[-1])
			
			path_matlab = os.path.join(path, 'matlab_data')
			if not os.path.exists(path_matlab):
				os.makedirs(path_matlab)
			new_file_name_matlab = os.path.join(path_matlab, dir[-1])
			

			# 去除Cz
			raw = self.drop_channel(raw)
			# 滤波1-80，50Hz陷波
			raw = self.filter_data(raw)
			# 重参考
			raw = self.reference_data(raw)
			# 去除坏导
			raw = self.drop_bad_channels(raw)
			# 去除坏段
			if self.task == 'matb':
				raw = self.drop_bad_period(raw)
			
			if self.task == 'back':
				raw = self.extract_data(raw, dir[-1])


			# 降采样
			# raw.resample(sfreq=self.sfreq)

			# ica去眼电与心电
			raw = self.ica(raw)

			raw_ecg = raw.copy().pick(picks = 'ecg')
			raw_eeg = raw.copy().pick_types(eeg=True)
			
			raw_eeg.save(insert_before_nth_from_end(new_file_name, '_eeg', 4), overwrite=True)
			raw_ecg.save(insert_before_nth_from_end(new_file_name, '_ecg', 4), overwrite=True)
			

			raw_eeg.export(insert_before_nth_from_end(new_file_name_matlab, '_eeg', 4).replace('fif', 'set'), overwrite=True)
			raw_ecg.export(insert_before_nth_from_end(new_file_name_matlab, '_ecg', 4).replace('fif', 'set'), overwrite=True)
			
		
process = Per_Process()
process.run()
process.save_to_csv()