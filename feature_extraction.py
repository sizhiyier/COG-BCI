import os
import pickle
import numpy as np
from scipy.stats import differential_entropy
from scipy.signal import ShortTimeFFT, get_window
from eeg_utils import EEGUtils, EEGProcessor
from temp_utils import calculate_mean_and_variance

# 递归地分解路径中的每一级目录
def split_path(path):
	parts = []
	while True:
		path, tail = os.path.split(path)
		if tail:
			parts.append(tail)
		else:
			if path:
				parts.append(path)
			break
	parts.reverse()
	return parts


class EEGConfig:
	FREQ_BANDS = {
		"Theta": (4, 8),
		"Alpha": (8, 13),
		"Beta": (13, 30),
		"Gamma": (30, 100)
	}
	BASE_DIR = '../Per_Process/Per_Process_1/sub'
	SAMPLE_RATE = 500
	STFT_WINDOW = 1500
	HOP_LENGTH = STFT_WINDOW
	SAVE_PATH = '../Result_DE/DE_'


class Embedding:
	
	def __init__(self, signal, file_path):
		self.fs = EEGConfig.SAMPLE_RATE
		self.stft_window = EEGConfig.STFT_WINDOW
		self.hop_length = EEGConfig.HOP_LENGTH
		self.signal = signal
		self.file_path = file_path
	
	def compute_stft(self, data, win_size, hop):
		win = get_window('hann', win_size)
		SFT = ShortTimeFFT(win, hop, self.fs)
		Sx = SFT.stft(data)
		return Sx
	
	def compute_differential_entropy(self):
		de_results = {}
		data = self.signal.get_data()
		if len(data.shape) == 3:
			n_epochs, n_channels, n_times = data.shape
			data_transposed = data.transpose(1, 0, 2)
			data_2d = data_transposed.reshape(n_channels, n_times * n_epochs)
			data = data_2d
		stft_magnitude = self.compute_stft(data, win_size=self.stft_window, hop=self.hop_length)
		for band, sfreq in EEGConfig.FREQ_BANDS.items():
			l_freq = sfreq[0]
			h_freq = sfreq[1]
			frequencies = np.fft.rfftfreq(self.stft_window, 1 / self.fs)
			freq_mask = (frequencies >= l_freq) & (frequencies < h_freq)
			Sx_filtered = stft_magnitude[:, freq_mask, :]
			# 函数内部使用了 np.sort(values, axis=-1)，该操作在遇到复数时，默认按复数的实部排序，而不考虑虚部。这可能导致排序和熵计算结果不符合预期。
			# 熵的定义通常是基于实数数据，尤其是概率密度函数，因此输入复数数据时，取其绝对值（幅值）通常是合理的预处理步骤。
			de_results[band] = differential_entropy(np.abs(Sx_filtered), axis=1)
		
		self.save_de_results(de_results)
		return de_results
	
	def save_de_results(self, de_results):
		file_name_with_extension = os.path.basename(self.file_path)
		file_name, _ = os.path.splitext(file_name_with_extension)
		file_path_to_save = os.path.join(EEGConfig.SAVE_PATH, *split_path(self.file_path)[-5:-1])
		if not os.path.exists(file_path_to_save):
			os.makedirs(file_path_to_save)
		with open(os.path.join(file_path_to_save, file_name + '.pkl'), 'wb') as f:
			pickle.dump(de_results, f)
		print(f"差分熵结果已成功保存到: {os.path.join(file_path_to_save, file_name + '.pkl')}")


def main():
	# 获取所有 EEG 文件目录
	base_dir = EEGConfig.BASE_DIR
	dir_eeg = EEGUtils.get_dir(base_dir)
	
	# 处理每个文件，计算差分熵并保存
	for _, file_path in dir_eeg.items():
		processor = EEGProcessor(file_path)
		embedding = Embedding(processor.raw_data, file_path)
		de_results = embedding.compute_differential_entropy()
	
	result = calculate_mean_and_variance(os.path.join(EEGConfig.SAVE_PATH, 'sub'), 'MATB', "BACK")
	with open(os.path.join(EEGConfig.SAVE_PATH, 'attribute.txt'), 'w', newline='') as file:
		file.write(f"SAMPLE_RATE: {EEGConfig.SAMPLE_RATE}\n")
		file.write(f"STFT_WINDOW: {EEGConfig.STFT_WINDOW}\n")
		file.write(f"HOP_LENGTH: {EEGConfig.HOP_LENGTH}\n")
		file.write(f"MATB: {result['MATB']}\n")
		file.write(f"BACK: {result['BACK']}\n")


if __name__ == "__main__":
	main()
