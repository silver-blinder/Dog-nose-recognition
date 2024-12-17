import os
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import SiameseNetwork

# 设置模型和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = SiameseNetwork().cuda()
model.load_state_dict(torch.load('siamese_network.pth', weights_only=True))
model.eval()

# 推理函数
def infer(model, img1, img2):
    with torch.no_grad():
        img1 = img1.cuda()
        img2 = img2.cuda()
        output = model(img1, img2)
        return (output > 0.5).float().item()

# 加载测试集
def load_dataset(dir_train):
    dataset = {}
    for subdir in os.listdir(dir_train):
        subfolder_path = os.path.join(dir_train, subdir)
        if os.path.isdir(subfolder_path):
            dataset[subdir] = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(('.jpg', '.png'))]
    return dataset

# 测试函数
def evaluate_model_random_100(model, dataset, total_pairs=100):
    """
    测试随机选择的 100 对样本，并统计正负样本对及其正确数量。
    :param model: Siamese network model
    :param dataset: Dictionary containing dataset paths
    :param total_pairs: Total number of pairs to test
    :return: accuracy, true_labels, predictions, stats
    """
    true_labels = []
    predictions = []
    pairs_generated = 0  # 已生成样本对计数

    labels = list(dataset.keys())  # 获取所有类别

    if len(labels) < 2:
        raise ValueError("Dataset must contain at least two categories to generate negative samples.")

    # 初始化统计数据
    num_positive_samples = 0
    num_negative_samples = 0
    correct_positive_samples = 0
    correct_negative_samples = 0

    while pairs_generated < total_pairs:
        # 随机选择正样本对或负样本对
        is_positive = random.choice([True, False])

        if is_positive:
            # 生成正样本对
            label = random.choice(labels)
            img_paths = dataset[label]
            if len(img_paths) < 2:
                continue  # 跳过不足两张图像的类别
            img1_path, img2_path = random.sample(img_paths, 2)
            true_labels.append(1)  # 正样本标签
            num_positive_samples += 1
        else:
            # 生成负样本对
            label1, label2 = random.sample(labels, 2)
            img_paths1 = dataset[label1]
            img_paths2 = dataset[label2]
            if len(img_paths1) < 1 or len(img_paths2) < 1:
                continue  # 跳过不足样本的类别
            img1_path = random.choice(img_paths1)
            img2_path = random.choice(img_paths2)
            true_labels.append(0)  # 负样本标签
            num_negative_samples += 1

        # 处理图像并进行推理
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0)
        prediction = infer(model, img1, img2)
        predictions.append(prediction)

        # 更新正确预测数量
        if is_positive and prediction == 1:
            correct_positive_samples += 1
        elif not is_positive and prediction == 0:
            correct_negative_samples += 1

        pairs_generated += 1

    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)

    # 返回统计信息
    stats = {
        "num_positive_samples": num_positive_samples,
        "num_negative_samples": num_negative_samples,
        "correct_positive_samples": correct_positive_samples,
        "correct_negative_samples": correct_negative_samples,
    }
    return accuracy, true_labels, predictions, stats


# 主程序
if __name__ == "__main__":
    dir_train = 'dir_train'

    # 加载数据集
    dataset = load_dataset(dir_train)

    # 调用测试函数
    total_pairs = 100  # 测试总对数
    accuracy, true_labels, predictions, stats = evaluate_model_random_100(model, dataset, total_pairs)


    # 绘制图表
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Siamese Network Testing Accuracy (100 Pairs)')
    plt.show()

    # 输出测试结果
    print("\n=== Testing Results ===")
    print(f"Total pairs tested: {total_pairs}")
    print(f"Positive samples: {stats['num_positive_samples']}, Correct: {stats['correct_positive_samples']}")
    print(f"Negative samples: {stats['num_negative_samples']}, Correct: {stats['correct_negative_samples']}")
    print(f"Model Accuracy: {accuracy:.2f}")
