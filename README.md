# AIXue RLHF Value Function

基于强化学习人类反馈(RLHF)的价值函数，用于预测学生在当前策略模型下的学习表现。

## 📋 项目概述

该项目实现了一个价值函数模型，通过分析学生的历史学习数据（包括发音、拼读、游戏成绩等）和个人属性（年龄、性别、CAT评级等），预测学生在当前课程中的拼读成绩表现。

## 🗂️ 项目结构

```
aixue-rlhf-vf/
├── aixue_value_function_v1.ipynb         # 价值函数模型_v1
├── aixue_value_function_v2.ipynb         # 价值函数模型_v2
├── datasets/                             # 数据集
│   ├── student-por_aixue.parquet        # AIXue真实学生数据集
│   └── student-por_kaggle.csv           # Kaggle公开数据集
└── README.md                            # 项目说明
```

## 🧾 版本说明

### v1 - 基础版本
基础实现，包含完整的数据处理与多模型对比

### v2 - 优化版本
在 v1 基础上进行以下优化改进：

#### 🔧 数据预处理优化
- **年龄计算精度提升**: 从整数年龄改为保留小数的年龄计算(`dt.days / 365`)，提高特征精度
- **新增年龄平方根特征**: 引入`age_sqrt = np.sqrt(age)`，增强模型对年龄非线性关系的捕捉能力  
- **新增课程ID对数特征**: 引入`lesson_id_log = np.log1p(lesson_id)`，处理课程序号的非线性影响
- **改进缺失值填充策略**: 使用更精确的平均值填充方法，避免数据类型转换导致的精度损失

#### 📈 模型性能提升
- **XGBoost参数精细化调优**: 增加learning_rate和gamma等关键参数的网格搜索，围绕最优值进行精确调参
- **整体性能改善**: 通过特征工程和参数优化，多个模型的预测精度都有不同程度的提升

#### 📊 评估体系完善  
- **指标说明详细化**: 为MSE、RMSE、R²等评估指标添加详细的含义解释和评估标准

### v3 - todo

- [ ] MLP 网络结构优化
- [ ] 异常数据的分析和处理



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
3. 打开并运行 `aixue_value_function.ipynb`

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
