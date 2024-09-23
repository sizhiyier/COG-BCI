
from model import MAET
from eeg_utils import *
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import random
import os

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_combine_files(file_paths, labels, fold_number):
	"""
	加载并组合多个文件的指定折数据，给不同文件赋予不同的标签。

	参数：
	- file_paths: 文件路径列表，列表中的每个路径会被逐一加载。
	- labels: 每个文件对应的标签，列表的长度应与 file_paths 一致。
	- fold_number: 指定要加载的折号。

	返回：
	- 组合后的 train_data, test_data 和相应的 labels。
	"""
	combined_train_data = []
	combined_test_data = []
	combined_train_labels = []
	combined_test_labels = []
	
	for file_path, label in zip(file_paths, labels):
		# 使用 Signal_Utils.load_fold_data 加载指定折的数据
		train_data_wind_de, test_data_wind_de = Signal_Utils.load_fold_data(file_path, fold_number)
		
		# print(f"Loading data from {file_path} for fold {fold_number}")
		# print(f"Train data mean: {np.mean(train_data_wind_de)}, std: {np.std(train_data_wind_de)}")
		# print(f"Test data mean: {np.mean(test_data_wind_de)}, std: {np.std(test_data_wind_de)}")
		# 组合 train_data 和 test_data
		combined_train_data.append(train_data_wind_de)
		combined_test_data.append(test_data_wind_de)
		
		# 给每个样本赋予对应的标签
		combined_train_labels.extend([label] * len(train_data_wind_de))
		combined_test_labels.extend([label] * len(test_data_wind_de))
	
	# 将列表中的数据合并为 ndarray
	combined_train_data = np.vstack(combined_train_data)
	combined_test_data = np.vstack(combined_test_data)
	
	return combined_train_data, combined_train_labels, combined_test_data, combined_test_labels

# 假设有多个文件要加载，并为它们分别赋予标签
file_paths = [
	r"C:\COG-BCI\Per_Process\Per_Process_\DE\DE_1\sub-01\ses-S1\eeg\MATBdiff_eeg_DE.pkl",
	r"C:\COG-BCI\Per_Process\Per_Process_\DE\DE_1\sub-01\ses-S1\eeg\MATBeasy_eeg_DE.pkl",
	r"C:\COG-BCI\Per_Process\Per_Process_\DE\DE_1\sub-01\ses-S1\eeg\MATBmed_eeg_DE.pkl"
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label = [0, 1, 2]  # 为每个文件指定一个标签，比如 0 表示 MATBdiff, 1 表示 MATBeasy, 2 表示 MATBmed
# 十折循环
for fold_number in range(4):

	set_seed(42)
	
	print(f"Processing fold {fold_number}...")
	
	# 加载并组合当前折的数据
	train_data, train_labels, test_data, test_labels = load_and_combine_files(file_paths, label, fold_number)
	
	# 创建训练集和测试集的 Dataset
	train_dataset = EEGDataset(train_data, train_labels)
	test_dataset = EEGDataset(test_data, test_labels)
	#
	# print(f"Train data shape: {train_data.shape}, Train labels length: {len(train_labels)}")
	# print(f"Test data shape: {test_data.shape}, Test labels length: {len(test_labels)}")
	
	# 使用 DataLoader 加载数据集
	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
	
	# 模型参数示例（根据实际需求调整）
	
	
	model = MAET(num_classes=3)
	model.to(device)
	
	# 定义损失函数
	criterion = nn.CrossEntropyLoss()
	# 定义优化器
	optimizer = optim.Adam(model.parameters(), lr=1e-4)
	
	num_epochs = 10
	for epoch in range(num_epochs):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		
		# 使用 tqdm 显示进度条
		
		
		for eeg, labels in train_loader:
			eeg = eeg.to(device)
			labels = labels.to(device)
			
			optimizer.zero_grad()
			
			outputs = model(eeg=eeg, eye=None, alpha_=0.0)  # eye=None 已正确处理
			loss = criterion(outputs, labels)
			running_loss += loss.item() * labels.size(0)
			
			_, predicted = torch.max(outputs, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			
			loss.backward()
			optimizer.step()
			

			
		epoch_loss = running_loss / len(train_loader.dataset)
		epoch_acc = correct / total
		print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
		
	model.eval()
	test_loss = 0.0
	test_correct = 0
	test_total = 0
	
	with torch.no_grad():
		for eeg, labels in test_loader:
			eeg = eeg.to(device)
			labels = labels.to(device)
			
			outputs = model(eeg=eeg, eye=None, alpha_=0.0)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * labels.size(0)
			
			_, predicted = torch.max(outputs, 1)
			test_total += labels.size(0)
			test_correct += (predicted == labels).sum().item()
	
	test_epoch_loss = test_loss / len(test_loader.dataset)
	test_epoch_acc = test_correct / test_total

