import os
import random
import shutil
import glob

# --- 配置 ---
source_dir = 'yolo垃圾分类数据集'
output_dir = 'dataset'
test_size = 400  # 测试集大小
image_ext = '.jpg' # 图片文件扩展名 (根据你的实际情况修改)
label_ext = '.txt' # 标签文件扩展名 (根据你的实际情况修改)
# -------------

source_images_dir = os.path.join(source_dir, 'images')
source_labels_dir = os.path.join(source_dir, 'labels')

train_images_dir = os.path.join(output_dir, 'train', 'images')
train_labels_dir = os.path.join(output_dir, 'train', 'labels')
test_images_dir = os.path.join(output_dir, 'test', 'images')
test_labels_dir = os.path.join(output_dir, 'test', 'labels')

# --- 创建目标目录 ---
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)
print(f"创建目录: {train_images_dir}")
print(f"创建目录: {train_labels_dir}")
print(f"创建目录: {test_images_dir}")
print(f"创建目录: {test_labels_dir}")

# --- 获取所有图片文件 (不含扩展名) ---
image_files = glob.glob(os.path.join(source_images_dir, f'*{image_ext}'))
if not image_files:
    print(f"错误：在 '{source_images_dir}' 中未找到任何图片文件 (扩展名: {image_ext})。请检查路径和扩展名。")
    exit()

file_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
total_files = len(file_basenames)
print(f"找到 {total_files} 个图片文件。")

# --- 随机打乱 ---
random.shuffle(file_basenames)

# --- 确定实际的测试集大小 ---
actual_test_size = min(test_size, total_files)
if actual_test_size < test_size:
    print(f"警告：请求的测试集大小 ({test_size}) 大于总文件数 ({total_files})。测试集将包含所有 {total_files} 个文件。")
    actual_test_size = total_files # 如果总数小于400，则全部放入测试集？或者报错？这里选择全部放入测试集，训练集为空

print(f"将分配 {actual_test_size} 个文件到测试集，{total_files - actual_test_size} 个文件到训练集。")

# --- 分割文件 ---
moved_train_count = 0
moved_test_count = 0
missing_labels = []

for i, basename in enumerate(file_basenames):
    source_image_path = os.path.join(source_images_dir, basename + image_ext)
    source_label_path = os.path.join(source_labels_dir, basename + label_ext)

    # 检查标签文件是否存在
    if not os.path.exists(source_label_path):
        missing_labels.append(basename)
        print(f"警告：找不到图片 '{basename}{image_ext}' 对应的标签文件 '{basename}{label_ext}'，将跳过此文件。")
        continue # 跳过没有对应标签的文件

    if i < actual_test_size:
        # 移动到测试集
        dest_image_path = os.path.join(test_images_dir, basename + image_ext)
        dest_label_path = os.path.join(test_labels_dir, basename + label_ext)
        try:
            shutil.move(source_image_path, dest_image_path)
            shutil.move(source_label_path, dest_label_path)
            moved_test_count += 1
        except Exception as e:
            print(f"移动文件 '{basename}' 到测试集时出错: {e}")
    else:
        # 移动到训练集
        dest_image_path = os.path.join(train_images_dir, basename + image_ext)
        dest_label_path = os.path.join(train_labels_dir, basename + label_ext)
        try:
            shutil.move(source_image_path, dest_image_path)
            shutil.move(source_label_path, dest_label_path)
            moved_train_count += 1
        except Exception as e:
            print(f"移动文件 '{basename}' 到训练集时出错: {e}")


print("\n--- 分割完成 ---")
print(f"成功移动 {moved_train_count} 个文件到训练集 ({train_images_dir}, {train_labels_dir})")
print(f"成功移动 {moved_test_count} 个文件到测试集 ({test_images_dir}, {test_labels_dir})")
if missing_labels:
    print(f"\n警告：以下 {len(missing_labels)} 个图片文件因为缺少对应的标签文件而被跳过:")
    # for label in missing_labels:
    #     print(f"- {label}") # 如果列表太长，可以注释掉这行

# --- (可选) 清理空的源目录 ---
# try:
#     if not os.listdir(source_images_dir):
#         os.rmdir(source_images_dir)
#         print(f"已删除空的源图片目录: {source_images_dir}")
#     if not os.listdir(source_labels_dir):
#         os.rmdir(source_labels_dir)
#         print(f"已删除空的源标签目录: {source_labels_dir}")
#     # 可以考虑是否删除 yolo垃圾分类数据集 根目录，如果它也空了
#     # if not os.listdir(source_dir):
#     #     os.rmdir(source_dir)
#     #     print(f"已删除空的源数据根目录: {source_dir}")
# except OSError as e:
#     print(f"清理源目录时出错: {e}")

print("脚本执行完毕。")
