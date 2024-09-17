import csv
import warnings
import pandas as pd
import mne
import numpy as np
from mne.preprocessing import find_bad_channels_lof, annotate_muscle_zscore, ICA, create_ecg_epochs
import os
from mne_icalabel import label_components


def find_folders_containing_string(directory, search_strings):
	"""查找包含指定字符串的子目录。"""
	if isinstance(search_strings, str):
		search_strings = [search_strings]
	search_strings = [s.upper() for s in search_strings]
	matching_folders = [
		os.path.join(root, filename)
		for root, dirs, files in os.walk(directory)
		for filename in files
		if filename.endswith('set') and any(s in filename.upper() for s in search_strings)
	]
	return matching_folders


class Per_Process:
	def __init__(self):
		self.record_dict = {}
		self.if_baseline = True
		self.base_dir = '..\\sub'
		self.task_keywords = ('matb')
		# self.task_keywords = ('back')
		self.channels_drop = ['Cz']
		self.save_folder_name = 'Per_Process_4'
		self.eeg_freqs = (1, 100)
		self.notch_freqs = (48, 52)
		self.ecg_freq = (0.04, 40)
		self.set_threshold()
		self.if_repair_artifact()
		self.process = ''
		self.first_call = True
	
	def update_record(self, key, value):
		"""更新记录字典。"""
		self.record_dict.setdefault(key, []).append(value)
	
	def save_to_csv(self, path):
		"""将记录字典保存为同一个 Excel 文件的不同 sheet，并在无法转换值时发出警告。"""
		# 计算字典值的均值，尝试将字符串转换为数值
		averages = {}
		for key, values in self.record_dict.items():
			numeric_values = []
			for v in values:
				try:
					# 将每个值转换为 float 类型
					numeric_values.append(float(v))
				except ValueError:
					# 发出警告，提示无法转换的值和其对应的 key
					warnings.warn(f"Warning: Key '{key}' contains a non-numeric value '{v}', which was skipped.")
					continue  # 跳过无法转换的值
			
			if numeric_values:
				averages[key] = sum(numeric_values) / len(numeric_values)
			else:
				averages[key] = None  # 没有可计算的数值，均值为 None
		
		# 将记录字典转换为 DataFrame
		record_df = pd.DataFrame.from_dict(self.record_dict, orient='index').reset_index()
		record_df.columns = ['Key'] + [f'Value_{i + 1}' for i in range(record_df.shape[1] - 1)]
		
		# 将均值转换为 DataFrame
		averages_df = pd.DataFrame(list(averages.items()))
		
		# 保存记录和均值到同一个 Excel 文件中的不同 sheet
		excel_path = os.path.join(path, 'record.xlsx')
		
		with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
			# 保存记录到 "record" sheet
			record_df.to_excel(writer, sheet_name='record', index=False, header=False)
			# 保存均值到 "average" sheet
			averages_df.to_excel(writer, sheet_name='average', index=False, header=False)
	
	def save_to_txt(self, path):
		# 将属性写入到txt文件
		file_path = os.path.join(path, 'attribute.txt')
		with open(file_path, 'w', newline='') as file:
			file.write(f"Threshold Drop Bad Channels: {self.threshold_drop_bad_channels}\n")
			file.write(f"Threshold Annotate Muscle Z-Score: {self.threshold_annotate_muscle_zscore}\n")
			file.write(f"Threshold Find Bads Muscle: {self.threshold_find_bads_muscle}\n")
			file.write(f"Threshold Pro ICALabel: {self.threshold_pro_icalabel}\n")
			file.write(f"If Drop Bad Channels: {self.if_drop_bad_channels}\n")
			file.write(f"If Annotate Muscle Z-Score: {self.if_annotate_muscle_zscore}\n")
			file.write(f"If Find Bads Muscle: {self.if_find_bads_muscle}\n")
			file.write(f"If Find Bads ECGs: {self.if_find_bads_ecgs}\n")
			file.write(f"If Find Bads EOGs: {self.if_find_bads_eogs}\n")
			file.write(f"If Baseline: {self.if_baseline}\n")
			file.write(f"Base Directory: {self.base_dir}\n")
			file.write(f"Task Keywords: {self.task_keywords}\n")
			file.write(f"Channels Drop: {self.channels_drop}\n")
			file.write(f"Save Folder Name: {self.save_folder_name}\n")
			file.write(f"EEG Frequencies: {self.eeg_freqs}\n")
			file.write(f"Notch Frequencies: {self.notch_freqs}\n")
			file.write(f"ECG Frequency: {self.ecg_freq}\n")
			file.write(f"Process:{self.process}\n")
	
	def set_threshold(self):
		# 默认1.5
		self.threshold_drop_bad_channels = 3
		# 默认4
		self.threshold_annotate_muscle_zscore = 4
		# 默认0.5
		self.threshold_find_bads_muscle = 0.5
		self.threshold_pro_icalabel = 0.95
	
	def if_repair_artifact(self):
		self.if_drop_bad_channels = True
		self.if_annotate_muscle_zscore = True
		self.if_find_bads_muscle = True
		self.if_find_bads_ecgs = True
		self.if_find_bads_eogs = True
	
	def append_string(self, input_string):
		# 拼接 self.process 和输入字符串，使用 '->' 连接
		if self.first_call:
			self.process = self.process+'->'+input_string

	
	def epochs_to_raw(self, epochs):
		warnings.warn("此函数还未验证正确性", UserWarning)
		# 提取 Epochs 的数据和信息
		data = epochs.get_data()  # (n_epochs, n_channels, n_samples)
		info = epochs.info.copy()
		
		# 将 Epochs 数据重新构造成 Raw 数据
		n_epochs, n_channels, n_samples = data.shape
		raw_data = np.reshape(data, (n_channels, n_epochs * n_samples))
		# 创建 Raw 对象
		raw = mne.io.RawArray(raw_data, info)
		return raw
	
	def get_dir(self):
		base_dir = self.base_dir
		task_keywords = self.task_keywords
		dir = {}
		for sub_number in os.listdir(base_dir):
			sub_dir = os.path.join(base_dir, sub_number)
			for session in os.listdir(sub_dir):
				eeg_dir = os.path.join(sub_dir, session, 'eeg')
				eeg_task_dir_lists = find_folders_containing_string(eeg_dir, task_keywords)
				for eeg_task_dir in eeg_task_dir_lists:
					eeg_task_dir = os.path.normpath(eeg_task_dir)
					eeg_task_dir_name = eeg_task_dir.split("\\")[-1][:-4]
					dir[f"{sub_number}_{session}_{eeg_task_dir_name}"] = eeg_task_dir
		return dir
	
	def drop_channel(self, raw):
		"""注意检查通道大小写"""
		channels = self.channels_drop
		for channel in channels:
			if channel in raw.info['ch_names']:
				raw.drop_channels('Cz')
		return raw
	
	def filter_data(self, raw):
		eeg_freqs = self.eeg_freqs
		notch_freqs = self.notch_freqs
		ecg_freq = self.ecg_freq
		raw.filter(l_freq=eeg_freqs[0], h_freq=eeg_freqs[1], picks='eeg')
		raw.notch_filter(freqs=notch_freqs, picks='eeg')
		raw.filter(l_freq=ecg_freq[0], h_freq=ecg_freq[1], picks='ecg')
		return raw
	
	def drop_bad_channels(self, raw):
		if self.if_drop_bad_channels:
			# 选取坏导
			bad_channels = find_bad_channels_lof(raw, threshold=self.threshold_drop_bad_channels, picks='eeg')
			raw.info['bads'].extend(bad_channels)
			raw.interpolate_bads()
			
			self.update_record('bad_channels', len(bad_channels))
		return raw
	
	def drop_bad_period(self, raw):
		if self.if_annotate_muscle_zscore:
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
			bad_durations = round(np.sum(annot.duration), 2)
			self.update_record('total_bad_duration', str(bad_durations))
		
		return raw_clean
	
	def reference_data(self, raw):
		return raw.set_eeg_reference(ref_channels='average', ch_type='eeg')
	
	def ica(self, raw):
		# method='picard',
		
		ica = ICA(method='infomax', fit_params=dict(extended=True), max_iter="auto", random_state=1019)
		ica.fit(raw)
		if self.if_find_bads_muscle:
			# scores = slope_score * focus_score * smoothness_score
			muscle_idx, muscle_idx_scores = ica.find_bads_muscle(raw, threshold=0.9)
		if self.if_find_bads_ecgs:
			ecg_idx, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG1')
		if self.if_find_bads_eogs:
			ic_labels = label_components(raw, ica, method="iclabel")
			eog_idx = [
				idx for idx, (prob, lbl) in enumerate(zip(ic_labels['y_pred_proba'], ic_labels['labels']))
				if lbl in ["eye blink"] and prob > self.threshold_pro_icalabel
			]
		combined_set = set()
		if muscle_idx is not None:
			combined_set.update(muscle_idx)
		if ecg_idx is not None:
			combined_set.update(ecg_idx)
		if eog_idx is not None:
			combined_set.update(eog_idx)
		
		# 去除重复索引并排除伪迹
		ica.exclude = list(combined_set)
		# 应用 ICA 去除伪迹成分
		ica.apply(raw)
		
		# 更新记录
		self.update_record('muscle_idx', len(muscle_idx) if muscle_idx is not None else 0)
		self.update_record('ecg_idx', len(ecg_idx) if ecg_idx is not None else 0)
		self.update_record('eog_idx', len(eog_idx) if eog_idx is not None else 0)
		self.update_record('total_idx', len(combined_set))
		
		print('*' * 50)
		print('muscle_idx: ', len(muscle_idx))
		print('ecg_idx: ', len(ecg_idx))
		print('eog_idx: ', len(eog_idx))
		print('total:', len(combined_set))
		print('*' * 50)
		return raw
	
	def get_event_id(self, file_name):
		if file_name.startswith('zero'):
			event_ids = ["6021", "6022"]
		if file_name.startswith('one'):
			event_ids = ["6121", "6122"]
		if file_name.startswith('two'):
			event_ids = ["6221", "6222", '6223']
		return event_ids
	
	def extract_data(self, raw, file_name):
		event_ids = self.get_event_id(file_name)
		# 从注释中提取事件
		events, event_id_map = mne.events_from_annotations(raw)
		include_id = [event_id_map.get(id) for id in event_ids]
		events_taril = mne.pick_events(events, include=include_id)
		
		if self.if_baseline:
			epochs = mne.Epochs(raw, events_taril, tmin=-0.2, tmax=2.3, baseline=(-0.2, 0), preload=True)
		else:
			epochs = mne.Epochs(raw, events_taril, tmin=0, tmax=2.5, baseline=None, preload=True)
		
		return epochs
	
	def process_file_path(self, file_dir):
		raw = mne.io.read_raw_eeglab(file_dir, preload=True)
		save_path = os.path.join("..", 'Per_Process', self.save_folder_name, os.path.dirname(file_dir)[3:].lstrip('\\'))
		return raw, save_path
	
	def check_the_path(self, *paths):
		for path in paths:
			if not os.path.exists(path):
				os.makedirs(path)
	
	def process_raw(self, raw, save_path, file_name):
		
		raw.set_channel_types({'ECG1': 'ecg'})  # 设置通道类型
		# 去除Cz
		raw = self.drop_channel(raw)
		self.append_string('drop_channel')
		# 滤波1-100，50Hz陷波
		raw = self.filter_data(raw)
		self.append_string('filter')
		# 重参考
		raw = self.reference_data(raw)
		self.append_string('reference')
		# 去除坏导
		raw = self.drop_bad_channels(raw)
		self.append_string('drop_bad_channels')
		# 去除坏段
		# if self.task == 'matb':
		# 	raw = self.drop_bad_period(raw)
		# 	self.append_string('drop_bad_period')
		# 降采样
		raw.resample(sfreq=200)
		self.append_string('resample')
		# ica去眼电与心电
		raw = self.ica(raw)
		self.append_string('ica')
		
		# 提取事件，去基线
		# if self.task == 'back':
		# 	raw = self.extract_data(raw, file_name)
		# 	self.append_string('extract_data')
		
		raw_ecg = raw.copy().pick(picks='ecg')
		raw_eeg = raw.copy().pick_types(eeg=True)
		
		save_path_matlab = os.path.join(save_path, 'matlab')
		self.check_the_path(save_path, save_path.replace('eeg', 'ecg'), save_path_matlab,
		                    save_path_matlab.replace('eeg', 'ecg'))
		
		raw_eeg.save(os.path.join(save_path, file_name + '_eeg.fif'), overwrite=True)
		raw_ecg.save(os.path.join(save_path, file_name + '_ecg.fif').replace('eeg', 'ecg'), overwrite=True)
		
		raw_eeg.export(os.path.join(save_path_matlab, file_name + '_eeg.set'), overwrite=True)
		raw_ecg.export(os.path.join(save_path_matlab, file_name + '_ecg.set').replace('eeg', 'ecg'), overwrite=True)
		
		if self.first_call:
			# 只在第一次调用时执行
			self.first_call = False
	
	def adjust_task(self, value):
		file_dir = value
		file_name = (value.split('\\')[-1]).split('.')[0]
		# 判断任务类型
		if file_name.startswith("MATB"):
			self.task = 'matb'
		elif file_name.endswith("BACK"):
			self.task = 'back'
		else:
			raise ValueError("task type error")
	
	def run(self):
		# 文件的储存路径
		documentation_path = os.path.join("..", 'Per_Process', self.save_folder_name)
		self.check_the_path(documentation_path)
		
		# get_dir已经集成到EEGUtils.get_dir(base_dir)
		data_dir = self.get_dir()
		for _, value in data_dir.items():
			file_name = value.split('\\')[-1].split('.')[0]
			self.adjust_task(file_name)
			raw, save_path = self.process_file_path(value)
			self.process_raw(raw, save_path, file_name)
		
		self.save_to_csv(documentation_path)
		self.save_to_txt(documentation_path)


process = Per_Process()
process.run()
