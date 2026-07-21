from ultralytics import YOLO
import os
import torch
import logging

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("training.log"),  # 日志文件
        logging.StreamHandler(),  # 控制台输出
    ],
)


def validate_base_model(model_path):
    """验证基础模型文件"""
    if not os.path.exists(model_path):
        try:
            logging.info(f"本地未找到基础模型，尝试下载: {model_path}")
            YOLO(model_path)  # YOLO会自动下载官方模型
            return True
        except Exception as e:
            logging.error(f"模型下载失败: {str(e)}")
            return False
    return True


def validate_dataset(yaml_path):
    """验证数据集配置和路径"""
    try:
        # 确保配置文件存在
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"数据集配置文件不存在: {yaml_path}")

        # 获取配置文件所在目录作为基础路径
        base_dir = os.path.dirname(os.path.abspath(yaml_path))

        # 检查数据集文件夹结构
        required_dirs = [
            os.path.join(base_dir, "dataset/train/images"),
            os.path.join(base_dir, "dataset/train/labels"),
            os.path.join(base_dir, "dataset/test/images"),
            os.path.join(base_dir, "dataset/test/labels"),
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"数据集目录不存在: {dir_path}")
            if not os.listdir(dir_path):
                raise ValueError(f"数据集目录为空: {dir_path}")

        logging.info("数据集验证通过")
        return True

    except Exception as e:
        logging.error(f"数据集验证失败: {str(e)}")
        return False


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

        return True
    except Exception as e:
        logging.error(f"环境验证失败: {str(e)}")
        return False


def train_model():
    """执行模型训练"""
    try:
        # --- 配置参数 ---
        config = {
            "dataset_config": "garbage_dataset.yaml",
            "base_model": "yolov8n.pt",
            "epochs": 300,
            "device": 0,  # GPU设备ID
            "batch_size": 8,  # 根据显存调整
            "imgsz": 640,  # 图像尺寸
            "workers": 2,  # 数据加载线程数（Windows建议≤2）
            "amp": True,  # 自动混合精度
            "patience": 50,  # 早停轮数
            "optimizer": "AdamW",  # 优化器选择
            "lr0": 0.001,  # 初始学习率
        }

        # --- 环境验证 ---
        logging.info("开始环境验证...")
        validate_environment()

        # --- 文件检查 ---
        # 验证数据集
        if not validate_dataset(config["dataset_config"]):
            raise RuntimeError("数据集验证失败，请检查目录结构和文件完整性")

        # 验证基础模型
        if not validate_base_model(config["base_model"]):
            raise FileNotFoundError(f"基础模型无法获取: {config['base_model']}")

        # --- 模型初始化 ---
        logging.info("初始化模型...")
        model = YOLO(config["base_model"])

        # --- 训练配置 ---
        logging.info(
            f"启动训练，设备: {'GPU' if isinstance(config['device'], int) else 'CPU'}"
        )
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
            verbose=False,  # 禁用冗余输出
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


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        logging.warning("用户中断训练")
    finally:
        logging.info("脚本执行结束")
