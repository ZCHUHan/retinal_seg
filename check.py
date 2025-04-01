import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import models
from ruamel.yaml import YAML
from bunch import Bunch
from utils.helpers import get_instance


# 1. 加载 .npz 文件
def load_npz_file(file_path):
    # 使用 numpy 加载 .npz 文件
    data = np.load(file_path)
    
    print("Available keys in the .npz file:", data.files)
    # 你需要根据实际的数组名称调整这些键
    inputs = data['inputs']  # 输入数据
    #inputs = data[data.files[0]]

    
    # 将 numpy 数组转换为 PyTorch 张量
    inputs_tensor = torch.from_numpy(inputs).float()  # 转为 float 类型
    
    return inputs_tensor

# 2. 创建数据集和数据加载器
def prepare_data(inputs_tensor, labels_tensor, batch_size=32):
    # 创建 TensorDataset
    dataset = TensorDataset(inputs_tensor, labels_tensor)
    
    # 创建 DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader

# 4. 主流程
def main():
    yaml = YAML(typ='safe', pure=True)
    with open('config_quan.yaml', 'r') as file:
        CFG = Bunch(yaml.load(file))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("CUDA is not available.")
        device = torch.device('cpu')
        
    model = get_instance(models, 'model', CFG).to(device)
    pth_path = '/home/dongz/FR-UNet_dysample/saved/FR_UNet_Quan_BiLinear/DRIVE/250313142101/checkpoint-epoch20.pth'

    # 加载模型的状态
    pretrained_dict = torch.load(pth_path, map_location=device)

    # 获取 state_dict
    state_dict = pretrained_dict['state_dict']

    # 去除 "module." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]  # 去掉 "module." 前缀
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # 加载修改后的状态字典到目标模型
    model.load_state_dict(new_state_dict)
    model.eval()
    
    
    # # 文件路径
    # file_path = 'outputs/log/4.npz'
    # file_me = 'npz_logging/8_conv.npz'
    
    # # 加载数据
    # # 使用 numpy 加载 .npz 文件
    # data_hard = np.load(file_path)
    # data_me = np.load(file_me)

    # print("Available keys in the hardware .npz file:", data_hard.files)
    # print("Available keys in the local .npz file:", data_me.files)

    
    # tensor_1 = torch.from_numpy(data_hard['out']).float()
    # print("hardware tensor size", tensor_1.size(), tensor_1)
    # #tensor_1 = tensor_1[:, :1, :, :]

    # tensor_2 = torch.from_numpy(data_me['output']).float()  
    # print("local tensor size", tensor_2.size(), tensor_2)
    
    # are_all_equal = torch.allclose(tensor_1, tensor_2, rtol=1e-5, atol=1e-8) 
    # print("Are all tensors equal (using torch.equal)?:", are_all_equal)
    
    # 文件路径
    file_path = 'outputs/log/0.npz'
    file_me = 'npz_logging/input.npy'
    
    # 加载数据
    # 使用 numpy 加载 .npz 文件
    data_hard = np.load(file_path)
    data_me = np.load(file_me)

    print("Available keys in the hardware .npz file:", data_hard.files)
    print("scale of input", data_hard['scales'])
    #print("Available keys in the local .npz file:", data_me.files)

    
    tensor_1 = torch.from_numpy(data_hard['in']).float()
    print("hardware tensor size", tensor_1.size())
    tensor_1 = tensor_1[:, :1, :, :]
    print("hardware tensor", tensor_1)

    tensor_2 = torch.from_numpy(data_me).float() 
    print("local tensor size", tensor_2.size())
    print("local tensor", tensor_2)
    
    are_all_equal = torch.allclose(tensor_1, tensor_2, rtol=1e-5, atol=1e-8) 
    print("Are all tensors equal (using torch.equal)?:", are_all_equal)
    
    #dummy_input = torch.rand(1, 1, 64, 64).to(device)

    #进行推理
    #with torch.no_grad():
    #   dummy_output = model(inputs_tensor_0)
       
    #print("dummy_output size", dummy_output.size())
       
    
    

if __name__ == "__main__":
    main()