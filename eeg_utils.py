import os
import mne
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
from scipy.stats import differential_entropy

class Signal_Utils:
	"""
	这个类包含信号的处理方法，都是静态方法
	"""
	
	@staticmethod
	def create_fold_iterator(data, n_splits=10, axis=1):
		"""
		按指定轴进行顺序划分，并逐步返回训练集和测试集，用于时序数据的十折交叉验证。

		参数:
		- data: 输入数据，形状为 (n_channels, n_samples) 或类似
		- n_splits: 折数，默认 10 折
		- axis: 要进行划分的轴，默认为 1（即按采样点进行划分）

		返回:
		- 生成器，逐步返回训练集和测试集数据 (train_data, test_data)
		"""
		# 获取需要划分的轴的长度
		n_samples = data.shape[axis]
		
		# 初始化 KFold，不打乱顺序
		kf = KFold(n_splits=n_splits, shuffle=False)
		
		# 生成按指定轴划分的索引
		for train_idx, test_idx in kf.split(np.arange(n_samples)):
			if axis == 0:
				# 如果按第一个维度划分（如通道），返回相应的行数据
				train_data = data[train_idx, :]
				test_data = data[test_idx, :]
			else:
				# 如果按第二个维度（如采样点），返回相应的列数据
				train_data = data[:, train_idx]
				test_data = data[:, test_idx]
			
			# 返回每一折的训练集和测试集
			yield train_data, test_data
	
	@staticmethod
	def windowed_signal_segments(signal_data, window_duration, step_duration):
		"""
		对输入信号进行窗口化处理，并根据指定的窗宽和步长分段，丢弃最后多余的部分。

		参数:
		- signal_data: 输入信号，形状为 (n_channels, n_samples) 的 NumPy 数组
		- fs: 采样率 (Hz)
		- window_duration: 窗宽（秒）
		- step_duration: 步长（秒）

		返回:
		- windowed_segments: 包含每个窗口片段的数组列表，形状为 (num_windows, n_channels, window_samples)
		"""
		# 计算窗宽和步长对应的采样点数
		window_samples = window_duration  # 窗宽对应的采样点数
		step_samples = step_duration  # 步长对应的采样点数
		
		# 分段信号处理
		windowed_segments = []
		
		# 滑窗选取信号
		for start in range(0, signal_data.shape[1] - window_samples + 1, step_samples):
			segment = signal_data[:, start:start + window_samples]  # 取出当前窗口的信号段
			windowed_segments.append(segment)  # 存储结果
		
		# 转换为 NumPy 数组并返回
		return np.array(windowed_segments)
	
	@staticmethod
	def epochs_to_raw(epochs):
		"""
		将 mne.Epochs 对象转换为 mne.io.RawArray 对象。
		:param epochs: mne.Epochs 实例
		:return: mne.io.RawArray 实例
		"""
		# 提取 epochs 的数据并转换为 2D 格式
		data = epochs.get_data()
		n_epochs, n_channels, n_times = data.shape
		data_transposed = data.transpose(1, 0, 2)  # 转置为 (n_channels, n_epochs, n_times)
		data_2d = data_transposed.reshape(n_channels, n_times * n_epochs)  # 重新调整为 2D 数据
		
		# 使用 epochs 的 info 属性创建 raw 数据
		info = epochs.info
		raw = mne.io.RawArray(data_2d, info)
		
		return raw
	
	@staticmethod
	def concatenate_dict_values(data_dict):
		"""
		接受一个字典，将字典中每个键的值按列拼接。
		每个键的值都应该是形状相同的 NumPy 数组。

		参数:
		- data_dict: 一个字典，键为字符串，值为 NumPy 数组

		返回:
		- 拼接后的 NumPy 数组
		"""
		# 确保字典非空
		if not data_dict:
			raise ValueError("输入字典不能为空")
		
		# 提取字典中的所有值，并按列进行拼接
		values = list(data_dict.values())
		
		# 检查所有数组的行数是否一致
		first_shape = values[0].shape[0]
		if not all(v.shape[0] == first_shape for v in values):
			raise ValueError("所有数组的行数（第一个维度）必须一致")
		
		# 按列（第二维度）拼接
		concatenated = np.vstack(values)
		
		return concatenated
	
	@staticmethod
	def create_fold_iterator(data, n_splits=10, axis=1):
		"""
	    按指定轴进行顺序划分，并逐步返回训练集和测试集，用于时序数据的十折交叉验证。

	    参数:
	    - data: 输入数据，形状为 (n_channels, n_samples) 或类似
	    - n_splits: 折数，默认 10 折
	    - axis: 要进行划分的轴，默认为 1（即按采样点进行划分）

	    返回:
	    - 生成器，逐步返回训练集和测试集数据 (train_data, test_data)
	    """
		# 获取需要划分的轴的长度
		n_samples = data.shape[axis]
		
		# 初始化 KFold，不打乱顺序
		kf = KFold(n_splits=n_splits, shuffle=False)
		
		# 生成按指定轴划分的索引
		for train_idx, test_idx in kf.split(np.arange(n_samples)):
			if axis == 0:
				# 如果按第一个维度划分（如通道），返回相应的行数据
				train_data = data[train_idx, :]
				test_data = data[test_idx, :]
			else:
				# 如果按第二个维度（如采样点），返回相应的列数据
				train_data = data[:, train_idx]
				test_data = data[:, test_idx]
			
			# 返回每一折的训练集和测试集
			yield train_data, test_data
	
	@staticmethod
	def save_pkl(folds_data, save_path='./all_folds.pkl'):
		"""
		保存所有折的训练集和测试集特征到一个 .pkl 文件中。

		参数：
		- folds_data (dict): 包含所有折的数据，每个折是一个字典，包含 'train_data_wind_de' 和 'test_data_wind_de'。
		- save_path (str): 保存路径。
		"""
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		with open(save_path, 'wb') as f:
			pickle.dump(folds_data, f)
		print(f"files have saved to {save_path}")
	
	@staticmethod
	def load_fold_data(save_path, fold_number=None):
		"""
		加载指定折的数据，或者作为迭代器逐个返回折的数据。

		参数：
		- save_path (str): 保存路径。
		- fold_number (int, optional): 折的编号，从1开始。如果提供，则返回该折数据。

		返回：
		- 如果提供了 fold_number，则返回指定折的数据 (train_data_wind_de, test_data_wind_de)。
		- 如果没有提供 fold_number，则作为迭代器逐个返回每个折的数据。
		"""
		with open(save_path, 'rb') as f:
			folds_data = pickle.load(f)
		
		# 定义内部函数用于加载特定折的数据
		def _load_specific_fold(folds_data, fold_number):
			fold_key = f'fold_{fold_number}'
			if fold_key not in folds_data:
				raise ValueError(f"Fold {fold_number} not found in the saved data.")
			
			fold_data = folds_data[fold_key]
			if fold_data.get('train_data_wind_de') is None or fold_data.get('test_data_wind_de') is None:
				raise ValueError(f"Missing 'train_data_wind_de' or 'test_data_wind_de' in {fold_key}.")
			
			return fold_data.get('train_data_wind_de'), fold_data.get('test_data_wind_de')
		
		# 定义内部函数用于迭代所有折的数据
		def _iter_all_folds(folds_data):
			for fold_key, fold_data in folds_data.items():
				yield fold_key, fold_data.get('train_data_wind_de'), fold_data.get('test_data_wind_de')
		
		# 根据是否提供 fold_number 来调用相应的内部函数
		if fold_number is not None:
			return _load_specific_fold(folds_data, fold_number)
		else:
			return _iter_all_folds(folds_data)
	
	@staticmethod
	def load_all_data(save_path):
		with open(save_path, 'rb') as f:
			all_data = pickle.load(f)
		return all_data

class File_path_Get:
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


class EEG_Filt:
	"""
	这个类负责加载和滤波EEG数据
	"""
	
	FREQ_BANDS = {
		"Delta": (1, 4),
		"Theta": (4, 8),
		"Alpha": (8, 13),
		"Beta": (13, 30),
		"Gamma": (30, 100)
	}
	
	def __init__(self, file_path):
		"""
		初始化类，file_path是具体的文件路径
		"""
		self.file_path = file_path
		self.raw_data = None
		self.filtered_data = {}
	
	def load_data(self):
		"""
		读取EEG数据。如果读取raw_fif失败，则尝试读取epochs。
		"""
		try:
			self.raw_data = mne.io.read_raw_fif(self.file_path, preload=True)
		except:
			try:
				epochs = mne.read_epochs(self.file_path, preload=True)
				print("成功读取 epochs 数据，正在转换为 raw 数据...")
				
				# 将 epochs 转换为 raw 数据
				self.raw_data = Signal_Utils.epochs_to_raw(epochs)
				print("已成功将 epochs 数据转换为 raw 格式。")
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
			self.filtered_data[band] = self.raw_data.copy().filter(l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.1,
			                                                       h_trans_bandwidth=0.1).get_data()
			print(f"已成功滤波 {band} 频带，范围为: {l_freq} Hz - {h_freq} Hz")
	
	def get_filtered_data(self, band_name=None, freq_bands=None):
		"""
		获取指定频带的滤波后的数据。如果数据尚未加载或滤波，则自动执行。
		:param band_name: 频带名称
		:param freq_bands: 频带字典，包含多个频段的上下限（如果需要滤波）
		:return: 返回滤波后的EEG数据
		"""
		# 如果尚未加载数据，先加载数据
		if self.raw_data is None:
			print("数据未加载，正在加载...")
			self.load_data()
		
		# 如果 band_name 和 freq_bands 都为 None，使用默认的 FREQ_BANDS
		if band_name is None and freq_bands is None:
			print("未指定 band_name 和 freq_bands，默认使用 FREQ_BANDS 进行滤波...")
			freq_bands = self.FREQ_BANDS
			self.filter_data(freq_bands)
			return self.filtered_data
		
		# 如果只给出了 band_name，但没有提供 freq_bands
		if band_name and freq_bands is None:
			if band_name in self.FREQ_BANDS:
				# 使用默认的 FREQ_BANDS 中的 band_name 对应频带进行滤波
				print(f"使用 FREQ_BANDS 中的 {band_name} 频带进行滤波...")
				self.filter_data({band_name: self.FREQ_BANDS[band_name]})
			else:
				print(f"频带 {band_name} 不存在于 FREQ_BANDS 中，请提供有效的频带或 freq_bands 参数。")
				return None
			return self.filtered_data.get(band_name, None)
		
		# 如果同时提供了 band_name 和 freq_bands
		if band_name and freq_bands is not None:
			print(f"使用提供的 freq_bands 中的 {band_name} 频带进行滤波...")
			self.filter_data({band_name: freq_bands})
			return self.filtered_data.get(band_name, None)

def split_path(path):
	return Path(path).parts
class EEG_Feature_Extraction:
	def __init__(self, config):
		self.config = config
	
	def calculate_feature(self, feature_name, data):
		"""
		根据特征名称计算相应的特征。

		参数：
		- feature_name: 'DE' 或 'PSD' 特征名称。
		- data: EEG 数据。

		返回：
		- 计算后的特征数据。
		"""
		if feature_name == 'DE':
			return differential_entropy(data, axis=-1)
		elif feature_name == 'PSD':
			return self._calculate_psd(data)
		else:
			raise ValueError(f"Unsupported feature: {feature_name}")
	
	def _calculate_psd(self, data_raw):
		"""
		计算功率谱密度 (PSD)。

		参数：
		- data_raw: 原始EEG数据，形状为 [n_example, n_channel, n_sample]。

		返回：
		- 计算后的 PSD 数据。
		"""
		psd_all = []
		for i, (_, freq_range) in enumerate(self.config.FREQ_BANDS.items()):
			data_band = data_raw[:, i * 62:(i + 1) * 62, :]
			psd = mne.time_frequency.psd_array_welch(data_band, sfreq=self.config.SAMPLE_RATE,
			                                         fmin=freq_range[0], fmax=freq_range[1])
			psd_all.append(np.mean(psd[0], axis=2))
		return np.concatenate(psd_all, axis=1)
	
	def process_folds(self, feature_name, data_filt, n_splits=10):
		"""
		处理每个折的数据，计算特征并生成结果。

		参数：
		- feature_name: 'DE' 或 'PSD' 特征名称。
		- data_filt: 经过滤波的 EEG 数据。
		- n_splits: 折数，默认 10。

		返回：
		- 处理后的折数据。
		"""
		fold_iterator = Signal_Utils.create_fold_iterator(data_filt, n_splits=n_splits, axis=1)
		folds_data = {}
		
		for i, (train_data, test_data) in enumerate(fold_iterator):
			train_data_wind = Signal_Utils.windowed_signal_segments(train_data,
			                                                        window_duration=self.config.window_length,
			                                                        step_duration=self.config.step_length)
			test_data_wind = Signal_Utils.windowed_signal_segments(test_data,
			                                                       window_duration=self.config.window_length,
			                                                       step_duration=self.config.step_length)
			
			train_data_wind_feature = self.calculate_feature(feature_name, train_data_wind)
			test_data_wind_feature = self.calculate_feature(feature_name, test_data_wind)
			
			folds_data[f'fold_{i}'] = {
				'train_data_wind_feature': train_data_wind_feature,
				'test_data_wind_feature': test_data_wind_feature
			}
		
		return folds_data
	
	def get_save_path(self, file_path, feature_name, suffix=""):
		"""
		生成保存路径。

		参数：
		- file_path: 原始文件路径。
		- feature_name: 特征名称。
		- suffix: 保存路径的后缀。

		返回：
		- 生成的保存路径。
		"""
		split_parts = split_path(file_path)
		if feature_name == 'DE':
			return os.path.join(self.config.save_path, split_parts[-4], split_parts[-3], split_parts[-2],
			                    split_parts[-1].split('.')[0] + f'_{feature_name}{suffix}.pkl')
		else:
			return os.path.join(self.config.save_path.replace('DE', feature_name), split_parts[-4], split_parts[-3], split_parts[-2],
			                    split_parts[-1].split('.')[0] + f'_{feature_name}{suffix}.pkl')
	
	def save_data(self, file_path, feature_name, folds_data, all_data, only_all_data=False):
		"""
		保存折数据和所有数据。

		参数：
		- file_path: 原始文件路径。
		- feature_name: 特征名称。
		- folds_data: 处理后的折数据。
		- all_data: 处理后的所有数据。
		- only_all_data: 是否只保存所有数据而不保存折数据，默认 False。
		"""
		if not only_all_data:
			save_path = self.get_save_path(file_path, feature_name)
			Signal_Utils.save_pkl(folds_data, save_path)
		
		save_path_all_data = self.get_save_path(file_path, feature_name, "_all_data")
		Signal_Utils.save_pkl(all_data, save_path_all_data)
	
	def process_and_save(self, feature_name, data_filt, file_path, n_splits=10, only_all_data=False):
		"""
		主方法：处理 EEG 数据并保存。

		参数：
		- feature_name: 'DE' 或 'PSD' 特征名称。
		- data_filt: 经过滤波的 EEG 数据。
		- file_path: 文件路径。
		- n_splits: 折数，默认 10。
		- only_all_data: 是否只保存所有数据而不保存折数据，默认 False。
		"""
		if not only_all_data:
			folds_data = self.process_folds(feature_name, data_filt, n_splits)
		else:
			folds_data = None
		
		all_data = self.calculate_feature(feature_name,
		                                  Signal_Utils.windowed_signal_segments(data_filt,
		                                                                        window_duration=self.config.window_length,
		                                                                        step_duration=self.config.step_length))
		
		self.save_data(file_path, feature_name, folds_data, all_data, only_all_data)


class EEGDataset(Dataset):
	def __init__(self, eeg_data, target_labels):
		"""
		初始化 EEG 数据集。

		参数:
		- eeg_data: 包含 (样本数, 特征维度) 的 ndarray 数据。
		- target_labels: 对应的标签，形状为 (样本数,)。
		"""
		self.eeg_data = eeg_data
		self.target_labels = target_labels
	
	def __len__(self):
		return len(self.target_labels)
	
	def __getitem__(self, idx):
		eeg = torch.as_tensor(self.eeg_data[idx], dtype=torch.float32)
		label = torch.as_tensor(self.target_labels[idx], dtype=torch.long)
		return eeg, label

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
        """
        初始化训练器。

        参数：
        - model: 要训练的模型。
        - train_loader: 训练数据集的 DataLoader。
        - test_loader: 测试数据集的 DataLoader。
        - criterion: 损失函数。
        - optimizer: 优化器。
        - device: 设备 (CPU 或 GPU)。
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        """
        训练一个 epoch。
        返回：
        - epoch_loss: 当前 epoch 的损失。
        - epoch_acc: 当前 epoch 的准确率。
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for eeg, target_labels in self.train_loader:
            eeg = eeg.to(self.device)
            target_labels = target_labels.to(self.device)

            self.optimizer.zero_grad()

            try:
                outputs = self.model(eeg=eeg, eye=None, alpha_=0.0)
                loss = self.criterion(outputs, target_labels)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue

            running_loss += loss.item() * target_labels.size(0)

            _, predicted = torch.max(outputs, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()

            try:
                loss.backward()
                self.optimizer.step()
            except Exception as e:
                print(f"Error during backpropagation: {e}")
                continue

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self):
        """
        评估模型性能。
        返回：
        - test_loss: 测试集的损失。
        - test_acc: 测试集的准确率。
        """
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for eeg, target_labels in self.test_loader:
                eeg = eeg.to(self.device)
                target_labels = target_labels.to(self.device)

                try:
                    outputs = self.model(eeg=eeg, eye=None, alpha_=0.0)
                    loss = self.criterion(outputs, target_labels)
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue

                test_loss += loss.item() * target_labels.size(0)

                _, predicted = torch.max(outputs, 1)
                test_total += target_labels.size(0)
                test_correct += (predicted == target_labels).sum().item()

        test_epoch_loss = test_loss / len(self.test_loader.dataset)
        test_epoch_acc = test_correct / test_total
        return test_epoch_loss, test_epoch_acc

    def train(self, num_epochs):
        """
        训练指定轮数，并在每轮后进行评估。

        参数：
        - num_epochs: 训练的 epoch 数。

        返回：
        - 每个 epoch 的训练和测试损失以及准确率。
        """
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}")

            test_loss, test_acc = self.evaluate()
            print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_acc:.4f}")
