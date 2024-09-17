import os
import pickle
import numpy as np
from eeg_utils import *

def calculate_mean_and_variance(base_dir, *keywords):
	"""
	这个函数根据提供的关键字遍历字典的每个键，
	并从每个键对应的值字典中获取第一个值的 shape 的第二个元素的值，
	最终返回每个关键字对应的均值和方差。

	:param data_dict: 一个包含字典的字典
	:param keywords: 若干个关键字，用来查找字典中的键（支持部分匹配）
	:return: 一个字典，键是关键字，值是均值和方差的元组
	"""

	data_loader = DEDataLoader(base_dir)  # 实例化并加载所有 .pkl 文件
	data_dict = data_loader.de_data
	
	results = {}
	
	for key in data_dict:
		# 遍历每个字典的键，检查是否与输入的关键字部分匹配
		for keyword in keywords:
			if keyword in key:
				sub_dict = data_dict[key]
				
				# 获取子字典中的第一个值
				first_value = next(iter(sub_dict.values()))
				
				# 获取该值的 shape 的第二个元素
				if isinstance(first_value, np.ndarray) and len(first_value.shape) > 1:
					second_dim = first_value.shape[1]
					
					# 如果关键字在结果中不存在，则初始化为一个列表
					if keyword not in results:
						results[keyword] = []
					
					# 保存该关键字匹配到的第二个维度的值
					results[keyword].append(second_dim)
	
	# 计算均值和方差并更新结果字典
	final_results = {}
	for keyword, values in results.items():
		mean_value = np.mean(values)
		variance_value = np.var(values)
		final_results[keyword] = (mean_value, variance_value)
	
	return final_results