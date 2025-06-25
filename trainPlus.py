from ultralytics import YOLO
import os
import torch
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # 日志文件
        logging.StreamHandler()  # 控制台输出
    ]
)

def validate_environment():
    """验证GPU环境和依赖"""
    try:
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，请检查GPU驱动和PyTorch安装")
        
        # 显存预清理
        torch.cuda.empty_cache()
        
        # 显存监控设置
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logging.info(f"GPU显存总量: {total_mem/1024**3:.2f} GB")
        
    except Exception as e:
        logging.error(f"环境验证失败: {str(e)}")
        raise

def train_model():
    """执行模型训练"""
    try:
        # --- 配置参数 ---
        config = {
            "dataset_config": "garbage_dataset.yaml",
            "base_model": "yolov8n.pt",
            "epochs": 300,
            "device": 0,           # GPU设备ID
            "batch_size": 8,       # 根据显存调整
            "imgsz": 640,          # 图像尺寸
            "workers": 2,          # 数据加载线程数（Windows建议≤2）
            "amp": True,           # 自动混合精度
            "patience": 50,        # 早停轮数
            "optimizer": "AdamW",  # 优化器选择
            "lr0": 0.001,          # 初始学习率
        }

        # --- 环境验证 ---
        logging.info("开始环境验证...")
        validate_environment()

        # --- 文件检查 ---
        required_files = [
            config["dataset_config"],
            config["base_model"]
        ]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"关键文件缺失: {file}")

        # --- 模型初始化 ---
        logging.info("初始化模型...")
        model = YOLO(config["base_model"])
        
        # --- 训练配置 ---
        logging.info(f"启动训练，设备: {'GPU' if isinstance(config['device'], int) else 'CPU'}")
        results = model.train(
            data=config["dataset_config"],
            epochs=config["epochs"],
            batch=config["batch_size"],
            imgsz=config["imgsz"],
            device=config["device"],
            workers=config["workers"],
            amp=config["amp"],
            patience=config["patience"],
            optimizer=config["optimizer"],
            lr0=config["lr0"],
            verbose=False  # 禁用冗余输出
        )

        # --- 结果处理 ---
        logging.info("\n--- 训练完成 ---")
        logging.info(f"最佳模型保存路径: {results.save_dir}")
        logging.info(f"验证集mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")

    except Exception as e:
        logging.error(f"训练过程异常终止: {str(e)}", exc_info=True)
        raise
    finally:
        # 显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("显存已清理")

if __name__ == '__main__':
    try:
        train_model()
    except KeyboardInterrupt:
        logging.warning("用户中断训练")
    finally:
        logging.info("脚本执行结束")