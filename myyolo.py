from ultralytics import YOLO
import os

if __name__ == '__main__':
    # --- 配置 ---
    dataset_config_path = "garbage_dataset.yaml"
    base_model = "yolov8n.pt"
    epochs = 10
    # device = 0  # 使用哪个 GPU (0) 或 CPU ('cpu')
    device = 'cpu'  # 使用 CPU 进行训练
    batch=8  # 减小 batch_size

    # 检查数据集配置文件是否存在
    if not os.path.exists(dataset_config_path):
        print(f"错误：找不到数据集配置文件 '{dataset_config_path}'。请确保文件存在于正确的位置。")
        exit()

    # 加载模型
    model = YOLO(base_model)

    # 开始训练
    print(f"开始使用数据集 '{dataset_config_path}' 训练模型 '{base_model}'...")
    print(f"训练轮数: {epochs}")

    try:
        results = model.train(
            data=dataset_config_path,
            epochs=epochs,
            device=device,
            batch=batch,
        )

        print("\n--- 训练完成 ---")
        print(f"训练结果保存在: {results.save_dir}")

    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")

    print("脚本执行完毕。")