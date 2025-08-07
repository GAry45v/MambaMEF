#!/usr/local/bin/python
import torch
import torch.nn as nn
import time
import os
import numpy as np
import argparse
from model.MambaMEF import MambaMEF
from config import config_1
from torch.utils.data import DataLoader
import myUtils
from thop import profile
from PIL import Image
from model.IAT_main import IAT
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save(img, path, suf, flag, mode, ckp):
    """
    保存融合后的图像
    参数:
    img: 融合后的图像张量
    path: 图像名称
    suf: 图像后缀
    flag: 标识符
    mode: 图像模式
    ckp: 检查点名称
    """
    # 确保输出目录存在
    output_dir = f"images/fused/MEFB"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理图像数据
    img = img.permute([0, 2, 3, 1]).cpu().detach().numpy()
    img = np.squeeze(img)
    img = np.uint8(img)
    img = img.astype(np.uint8)
    
    # 创建PIL图像并保存
    save_img = Image.fromarray(img, mode)
    output_path = os.path.join(output_dir, f"{path}_{flag}.{suf}")
    save_img.save(output_path)
    print(f"已保存融合结果: {output_path}")

def count_parameters(model):
    """计算模型的总参数量"""
    return sum(p.numel() for p in model.parameters())

def run(args, ckp):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------main_model--------------------------------
    # 初始化MambaMEF模型
    model = MambaMEF()
    checkpoint = torch.load("/nvme0/shared_data/MambaMEF/output/job1/log/run_20250528_013122/checkpoints/checkpoint_0039.pth")
    net = checkpoint['model']
    net = {key.replace("module.", ""): val for key, val in net.items()}
    model.load_state_dict(net, strict=False)
    model.to(device)
    
    # 初始化IAT曝光校正模型
    exposure_correction_model = IAT()
    exposure_correction_model.load_state_dict(
        torch.load("/nvme0/shared_data/MambaMEF/IAT_enhance/workdirs/snapshots_folder_exposure/best_Epoch.pth", 
                   map_location="cuda"))
    exposure_correction_model.to(device)
    
    # 计算模型参数和计算量
    model.eval()
    exposure_correction_model.eval()
    
    # 计算MambaMEF模型的参数量和计算量
    test_Y = torch.randn(2, 1, 256, 256).cuda()
    flops, params = profile(model, (test_Y, test_Y))
    
    # 计算IAT模型的参数量
    iat_params = count_parameters(exposure_correction_model)
    
    print('\n===== 模型参数量统计 =====')
    print(f'MambaMEF模型:')
    print(f'  FLOPs: {(flops/1024/1024/1024):.4f}G')
    print(f'  参数量: {(params/1024/1024):.4f}M')
    print(f'IAT模型参数量: {(iat_params/1024/1024):.4f}M')
    print(f'总参数量: {((params + iat_params)/1024/1024):.4f}M')
    print('=========================\n')
    
    # 加载测试数据集
    dataset_test = myUtils.build_dataset(args, mode='test')
    n = len(dataset_test)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_val = DataLoader(dataset_test, batch_size=1, sampler=sampler_test, num_workers=1, pin_memory=True)
    
    # 初始化计时变量
    total_time = 0.0
    processed_images = 0
    
    with torch.no_grad():
        for _, data in enumerate(dataloader_val):
            # 解包数据
            samples, folder_name, suffix = data
            
            # 处理文件夹名称和后缀
            if isinstance(folder_name, (torch.Tensor, list)):
                folder_name = folder_name[0]
            if not isinstance(folder_name, str):
                folder_name = str(folder_name)
                
            if isinstance(suffix, (torch.Tensor, list)):
                suffix = suffix[0]
            if not isinstance(suffix, str):
                suffix = str(suffix)
            
            # 将图像数据移动到GPU
            img0_RGB = samples['img0_RGB'].cuda()
            img0_gra = samples['img0_gra'].cuda()
            img1_RGB = samples['img1_RGB'].cuda()
            img1_gra = samples['img1_gra'].cuda()
            img1_Y = samples['img1_Y'].cuda()
            img1_Cb = samples['img1_Cb'].cuda()
            img1_Cr = samples['img1_Cr'].cuda()
            img0_Y = samples['img0_Y'].cuda()
            img0_Cb = samples['img0_Cb'].cuda()
            img0_Cr = samples['img0_Cr'].cuda()
            
            # 开始计时
            start_time = time.time()
            
            # 执行曝光校正
            _, _, img0_Y_enhance = exposure_correction_model(img0_Y)
            _, _, img1_Y_enhance = exposure_correction_model(img1_Y)
            
            # 执行融合
            fus_Y = model(img0_Y_enhance, img1_Y_enhance)
            fus_Y = fus_Y.clamp(0, 1)
            fus_Cb = myUtils.colorCombine_torch([img0_Cb*255, img1_Cb*255])
            fus_Cr = myUtils.colorCombine_torch([img0_Cr*255, img1_Cr*255])
            fus_RGB = myUtils.YCbCr2RGB_torch(fus_Y*255, fus_Cb, fus_Cr)
            
            # 结束计时
            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            processed_images += 1
            
            # 保存融合结果
            save(fus_RGB, folder_name, suffix, "First", "RGB", ckp)
            
            print(f"处理图片 {folder_name}, 耗时: {processing_time:.4f}秒")
    
    # 打印统计信息
    if processed_images > 0:
        avg_time = total_time / processed_images * 1000  # 转换为毫秒
        print('\n===== 处理时间统计 =====')
        print(f'总处理图片数量: {processed_images}')
        print(f'总处理时间: {total_time:.4f}秒')
        print(f'平均每张图片处理时间: {avg_time:.4f}毫秒')
        print('=========================')
    else:
        print('没有处理任何图片')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config", default=1, type=int)
    parser.add_argument("--ckp", default='checkpoint_0039.pth', type=str)
    args = parser.parse_args()
    run(config_1.get_config(), args.ckp)
