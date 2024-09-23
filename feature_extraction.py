import os
import pickle
from pathlib import Path
import mne
import numpy as np
from scipy.stats import differential_entropy
from eeg_utils import File_path_Get, EEG_Filt, Signal_Utils,EEG_Feature_Extraction


# 递归地分解路径中的每一级目录
def split_path(path):
	return Path(path).parts


class EEGConfig:
	FREQ_BANDS = EEG_Filt.FREQ_BANDS
	BASE_DIR = '../Per_Process/Per_Process_1/sub_test'
	SAMPLE_RATE = 500
	
	# 将 presets 作为类属性
	PRESETS = {
		1: (1000, 500, '../Result_DE/DE_1'),
		2: (1000, 750, '../Result_DE/DE_2'),
		3: (1500, 500, '../Result_DE/DE_3'),
		4: (1500, 1000, '../Result_DE/DE_4'),
		5: (2000, 500, '../Result_DE/DE_5'),
		6: (2000, 1000, '../Result_DE/DE_6'),
		7: (2000, 1500, '../Result_DE/DE_7'),
	}
	
	def __init__(self, window_length, step_length, save_path):
		self.window_length = window_length
		self.step_length = step_length
		self.save_path = save_path
	
	@classmethod
	def from_preset(cls, preset):
		if preset in cls.PRESETS:
			return cls(*cls.PRESETS[preset])
		else:
			raise ValueError(f"Preset {preset} is not available.")





def main():
	for n in range(len(EEGConfig.PRESETS)):
		# 获取所有 EEG 文件目录
		config = EEGConfig.from_preset(n + 1)
		base_dir = EEGConfig.BASE_DIR
		dir_eeg = File_path_Get.get_dir(base_dir)
		
		# 处理每个文件，计算差分熵并保存
		for file_name, file_path in dir_eeg.items():
			eeg_filt_data = EEG_Filt(file_path)
			data = Signal_Utils.concatenate_dict_values(eeg_filt_data.get_filtered_data())
			eeg_feature_extractor = EEG_Feature_Extraction(config)
			eeg_feature_extractor.process_and_save('PSD', data, file_path,)
			eeg_feature_extractor.process_and_save('DE', data, file_path)

		
		with open(os.path.join(config.save_path, 'attribute.txt'), 'w', newline='') as file:
			file.write(f"BASE_DIR: {EEGConfig.BASE_DIR}\n")
			file.write(f"SAMPLE_RATE: {EEGConfig.SAMPLE_RATE}\n")
			file.write(f"window_length: {config.window_length}\n")
			file.write(f"step_length: {config.step_length}\n")


if __name__ == "__main__":
	main()
