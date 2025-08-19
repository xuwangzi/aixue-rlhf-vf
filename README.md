# AIXue RLHF Value Function

基于强化学习人类反馈(RLHF)的价值函数，用于预测学生在当前策略模型下的学习表现。

## 📋 项目概述

该项目实现了一个价值函数模型，通过分析学生的历史学习数据（包括发音、拼读、游戏成绩等）和个人属性（年龄、性别、CAT评级等），预测学生在当前课程中的拼读成绩表现。

## 🗂️ 项目结构

```
aixue-rlhf-vf/
├── aixue_value_function.ipynb          # 主要的价值函数模型实现
├── aixue_value_function_filter.ipynb   # 数据过滤和预处理
├── datasets/                           # 数据集
│   ├── student-por_aixue.parquet      # AIXue真实学生数据集
│   └── student-por_kaggle.csv         # Kaggle公开数据集
└── README.md                          # 项目说明
```

## 🎯 核心功能

### 数据特征
- **学生画像**: 性别、年龄、CAT评级等个人属性
- **历史成绩**: 过去1-3节课的发音、拼读、游戏成绩
- **预测目标**: 当前课程的拼读成绩(spell_score)

### 模型对比
项目实现了多种回归模型并进行了性能对比：

| 模型 | MSE | RMSE | R² | 推荐度 |
|------|-----|------|----|----|
| **Support Vector Regression** | 62.67 | 7.92 | 0.886 | ⭐⭐⭐⭐⭐ |
| **XGBoost Regression** | 67.26 | 8.20 | 0.877 | ⭐⭐⭐⭐ |
| **Random Forest Regression** | 73.64 | 8.58 | 0.866 | ⭐⭐⭐⭐ |
| **Decision Tree Regression** | 80.90 | 8.99 | 0.852 | ⭐⭐⭐ |
| **Linear Regression** | 87.07 | 9.33 | 0.841 | ⭐⭐⭐ |
| **K-Nearest Neighbors** | 92.91 | 9.64 | 0.830 | ⭐⭐ |
| **Multi-Layer Perceptron** | 99.22 | 9.96 | 0.819 | ⭐⭐ |

## 🚀 快速开始

### 环境要求
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch
```

### 运行步骤
1. 克隆项目到本地
2. 确保数据集在 `datasets/` 目录下
3. 运行 `aixue_value_function.ipynb` 进行模型训练和评估
4. 运行 `aixue_value_function_filter.ipynb` 进行数据过滤分析

## 📊 数据说明

### 数据集来源（不公开）
- **student-por_aixue.parquet**: AIXue平台的真实学生学习数据
- **student-por_kaggle.csv**: 来自Kaggle的公开教育数据集，用于对比分析

### 主要特征
- 学生基本信息：性别、年龄
- CAT评级：听力、阅读、口语水平
- 历史成绩：过去1-3节课的各项得分
- 预测目标：当前课程拼读成绩(0-100分)

## 🔧 技术特点

- **数据预处理**: 智能缺失值填充、特征工程
- **模型集成**: 多种机器学习算法对比
- **超参数优化**: GridSearchCV自动调参
- **性能评估**: MSE、RMSE、R²多维度评估

## 📈 应用场景

- 个性化学习路径推荐
- 学习效果预测与干预
- 教学策略优化
- 学生学习能力评估
