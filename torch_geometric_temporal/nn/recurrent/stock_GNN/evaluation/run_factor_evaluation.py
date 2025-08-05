"""
使用已训练模型进行因子线性模型评估的示例脚本

这个脚本展示了如何:
1. 加载已训练的GNN模型
2. 提取训练/验证/测试集的32个因子预测
3. 使用训练集数据拟合线性模型
4. 在验证集和测试集上评估线性模型性能

使用方法:
1. 修改main()函数中的checkpoint_path为你的模型路径
2. 运行: python run_factor_evaluation.py

或者在代码中调用:
    results = load_trained_model_and_evaluate(
        checkpoint_path="path/to/your/model.ckpt",
        device='cuda'
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from pytorch_factor_evaluation import evaluate_factor_linear_model_pytorch, print_evaluation_summary

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset import StockDataModule
from torch_geometric_temporal.nn.recurrent.stock_GNN.training_module import DynamicGraphLightning as TrainingModule


def load_config(config_path: str = None) -> DictConfig:
    """Load configuration from YAML file - same as predict.py"""
    if config_path and os.path.exists(config_path):
        return OmegaConf.load(config_path)
    else:
        # Default configuration for evaluation
        return OmegaConf.create({
            "data": {
                "_target_": "torch_geometric_temporal.nn.recurrent.stock_GNN.dataset.stock_dataset.StockDataModule",
                "data_dir": "/home/xu/clean_data_unaligned",
                "use_factors": True,
                "sequence_length": 30,
                "prediction_horizons": [1],
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "batch_size": 1,
                "num_workers": 8,
                "normalize_features": True,
                "normalize_targets": True,
                "debug": False
            }
        })


def load_model_from_checkpoint(checkpoint_path: str) -> pl.LightningModule:
    """Load the trained model from checkpoint - same as predict.py"""
    try:
        # Load the training module (which contains the model)
        model = TrainingModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        print(f"✓ Loaded model from checkpoint: {checkpoint_path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from checkpoint: {e}")
        raise


def load_trained_model_and_evaluate(checkpoint_path, data_module=None, config_path=None, device='cuda'):
    """
    加载训练好的模型并进行因子线性评估
    
    Args:
        checkpoint_path: 模型检查点路径
        data_module: 数据模块实例 (可选，会自动创建)
        config_path: 配置文件路径 (可选)
        device: 计算设备
    """
    
    print("=== 加载模型和数据 ===")
    
    # 1. 加载训练好的模型
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件 {checkpoint_path}")
        return None
    
    # 使用和predict.py相同的加载方式
    model = load_model_from_checkpoint(checkpoint_path)
    model.to(device)
    
    # 2. 设置数据模块 (如果没有提供则创建)
    if data_module is None:
        print("创建数据模块...")
        cfg = load_config(config_path)
        data_module = instantiate(cfg.data)
    
    data_module.setup()
    
    # 3. 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    print(f"✓ 数据加载器准备完成")
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader)}")
    print(f"   测试批次数: {len(test_loader)}")
    
    # 4. 提取因子预测
    def extract_factors_from_dataloader(model, dataloader, max_batches=None, take_last=False):
        """从数据加载器中提取因子预测"""
        model.eval()
        all_factors = []
        all_returns = []
        
        # 如果需要取最后几个batch，先将所有数据收集起来
        if take_last and max_batches:
            batches_to_process = []
            total_batches = 0
            for batch in dataloader:
                batches_to_process.append(batch)
                total_batches += 1
            
            # 取最后max_batches个
            if len(batches_to_process) > max_batches:
                batches_to_process = batches_to_process[-max_batches:]
                print(f"   从总共{total_batches}个batch中取最后{max_batches}个")
            
        else:
            batches_to_process = dataloader
        
        with torch.no_grad():
            for i, batch in enumerate(batches_to_process):
                if not take_last and max_batches and i >= max_batches:
                    break
                    
                if len(batch) == 2:
                    features, targets = batch
                elif len(batch) == 3:
                    features, targets, metadata = batch
                else:
                    print(f"警告: 未知的batch格式，长度为 {len(batch)}")
                    continue
                
                # 将数据移到正确的设备
                features = features.to(device)
                targets = targets.to(device)
                
                # 获取模型预测的因子 [B, N, K]
                # 使用和predict.py相同的方式调用模型
                try:
                    factors = model(features)  # Shape: [batch_size, num_stocks, output_dim]
                    batch_idx = i if not take_last else len(batches_to_process) - max_batches + i
                    print(f'batch {batch_idx}: factor shape {factors.shape}, return shape {targets.shape}')
                    
                    # 重要: 需要转置targets以匹配因子的维度
                    # factors: [B, N, K], targets: [B, T, N] -> 需要转为 [B, N, T]
                    targets_reshaped = targets.transpose(1, 2)  # [B, N, T]
                    
                    # 沿着股票维度展平: [B, N, K] -> [B*N, K]
                    factors_flat = factors.view(-1, factors.shape[-1])  # [B*N, K]
                    targets_flat = targets_reshaped.view(-1, targets_reshaped.shape[-1])  # [B*N, T]
                    
                    all_factors.append(factors_flat.cpu())  # [B*N, K]
                    all_returns.append(targets_flat.cpu())  # [B*N, T]
                        
                except Exception as e:
                    print(f"模型预测失败: {e}")
                    continue
        
        if all_factors:
            # 现在可以直接concatenate，因为所有tensor都是[N_i, K]和[N_i, T]的格式
            all_factors_tensor = torch.cat(all_factors, dim=0)  # [total_stocks, K]
            all_returns_tensor = torch.cat(all_returns, dim=0)  # [total_stocks, T]
            print(f"✓ 成功拼接: 因子 {all_factors_tensor.shape}, 收益 {all_returns_tensor.shape}")
            return all_factors_tensor, all_returns_tensor
        else:
            return None, None
    
    # 提取训练集因子 - 取最后100个batch（最新的数据）
    print("提取训练集因子（最后100个batch）...")
    train_factors, train_returns = extract_factors_from_dataloader(
        model, train_loader, max_batches=200, take_last=True
    )
    
    # 提取验证集因子  
    print("提取验证集因子...")
    val_factors, val_returns = extract_factors_from_dataloader(model, val_loader)
    
    # 提取测试集因子
    print("提取测试集因子...")
    test_factors, test_returns = extract_factors_from_dataloader(model, test_loader)
    
    # 5. 检查数据形状
    if train_factors is None:
        print("错误: 无法提取训练集数据")
        return None
    
    print(f"✓ 数据提取完成")
    print(f"   训练集因子形状: {train_factors.shape}")
    print(f"   训练集收益形状: {train_returns.shape}")
    print(f"   验证集因子形状: {val_factors.shape if val_factors is not None else 'None'}")
    print(f"   测试集因子形状: {test_factors.shape if test_factors is not None else 'None'}")
    
    # 6. 进行因子线性模型评估
    print("\n" + "="*60)
    # 现在我们需要重新组织数据格式以匹配原始evaluation函数的期望
    # 原函数期望: train_factors [B, N, K], train_returns [B, T, N]
    # 但我们现在有: train_factors [total_stocks, K], train_returns [total_stocks, T]
    
    # 为了适配原函数，我们添加一个batch维度
    train_factors_batched = train_factors.unsqueeze(0)  # [1, total_stocks, K]
    train_returns_batched = train_returns.transpose(0, 1).unsqueeze(0)  # [1, T, total_stocks]
    
    val_factors_batched = val_factors.unsqueeze(0) if val_factors is not None else None
    val_returns_batched = val_returns.transpose(0, 1).unsqueeze(0) if val_returns is not None else None
    
    test_factors_batched = test_factors.unsqueeze(0) if test_factors is not None else None
    test_returns_batched = test_returns.transpose(0, 1).unsqueeze(0) if test_returns is not None else None
    
    results = evaluate_factor_linear_model_pytorch(
        train_factors_batched.to(device), train_returns_batched.to(device),
        val_factors_batched.to(device), val_returns_batched.to(device), 
        test_factors_batched.to(device), test_returns_batched.to(device)
    )
    
    # 7. 打印结果摘要
    print_evaluation_summary(results)
    
    return results

def main():
    """主函数 - 修改这里的路径和参数"""
    
    # 配置参数 - 请根据实际情况修改
    checkpoint_path = "/home/xu/pytorch_geometric_temporal/torch_geometric_temporal/nn/recurrent/stock_GNN/logs/gat_experiment/version_3/checkpoints/stock-gnn-42--0.0553.ckpt"  # 模型检查点路径
    config_path = None  # 配置文件路径 (可选)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # 运行评估 - 数据模块会自动创建
    try:
        results = load_trained_model_and_evaluate(
            checkpoint_path=checkpoint_path,
            data_module=None,  # 会自动创建
            config_path=config_path,
            device=device
        )
        
        if results:
            print("\n🎉 评估成功完成！")
            
            # 可以在这里添加更多分析，比如保存结果到文件
            # torch.save(results, 'factor_evaluation_results.pt')
            
        else:
            print("❌ 评估失败")
            
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
