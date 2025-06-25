from ultralytics import YOLO

def get_classify_id(result_list, class_name: str):
    if not result_list:
        print("错误：预测结果列表为空。")
        return None
    names_map = result_list[0].names
    if not names_map:
        print("错误：在预测结果中找不到类别名称映射 (.names)。")
        return None

    # 遍历类别 ID 和名称的映射
    for class_id, name in names_map.items():
        if name == class_name:
            return class_id  # 找到匹配的名称，返回 ID

    # 如果遍历完所有名称都没有找到匹配项
    print(f"警告：未在模型的类别中找到名称 '{class_name}'。")
    return None


def main():        
    model = YOLO(model="best.pt", task="detect")
    result = model.predict(source="yoloDetect.jpg", save=True, conf=0.2)
    # class_id = get_classify_id(result, "other waste")
    # print(class_id)

if __name__ == "__main__":
    main()