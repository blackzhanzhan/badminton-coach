# 羽毛球动作分析系统 - 机器学习版本

## 项目概述

这是羽毛球动作分析系统的机器学习增强版本，专注于使用深度学习和机器学习技术来分析和改进羽毛球动作。

## 主要特性

### 🤖 机器学习核心模块
- **HybridActionAdvisor**: 混合动作顾问，结合规则分析和ML预测
- **ActionQualityPredictor**: 动作质量预测器，使用随机森林等算法
- **EnhancedPoseAnalyzer**: 增强姿态分析器，深度学习姿态识别
- **OnlineLearningManager**: 在线学习管理器，支持模型持续学习
- **MLFeatureExtractor**: 机器学习特征提取器
- **VideoProcessor**: 视频处理模块

### 🎯 核心功能
1. **智能动作分析**: 基于机器学习的动作质量评估
2. **实时姿态检测**: 使用深度学习模型进行姿态识别
3. **个性化建议**: 根据用户数据提供定制化改进建议
4. **持续学习**: 系统可以从用户反馈中学习和改进
5. **混合分析**: 结合传统规则和ML预测的双重分析

## 技术栈

- **深度学习**: MediaPipe, TensorFlow/PyTorch
- **机器学习**: scikit-learn, numpy, pandas
- **计算机视觉**: OpenCV, PIL
- **用户界面**: tkinter
- **数据处理**: JSON, 特征工程

## 安装和使用

### 环境要求
- Python 3.8+
- 依赖包：见requirements.txt

### 快速开始
1. 克隆仓库并切换到machine-learning分支
2. 安装依赖：`pip install -r requirements.txt`
3. 配置API密钥（config.ini）
4. 运行：`python main.py`

### 使用步骤
1. 启动应用程序
2. 配置火山引擎API密钥
3. 启用机器学习模式
4. 选择视频文件进行分析
5. 查看AI生成的分析报告和建议

## 项目结构

```
羽毛球项目/
├── modules/                    # 核心模块
│   ├── hybrid_action_advisor.py      # 混合动作顾问
│   ├── action_quality_predictor.py   # 动作质量预测
│   ├── enhanced_pose_analyzer.py     # 增强姿态分析
│   ├── online_learning_manager.py    # 在线学习管理
│   ├── ml_feature_extractor.py       # 特征提取
│   └── video_processor.py            # 视频处理
├── ui/                         # 用户界面
│   └── main_window_tk.py            # 主窗口
├── data/                       # 数据目录
│   ├── templates/                   # 标准模板
│   ├── staged_templates/            # 用户数据
│   └── feedback/                    # 用户反馈
├── models/                     # 机器学习模型
└── main.py                     # 主程序入口
```

## 机器学习特性

### 动作质量预测
- 使用随机森林算法预测动作质量
- 支持多维特征分析
- 提供置信度评估

### 在线学习
- 从用户反馈中持续学习
- 模型自动更新和优化
- 个性化推荐系统

### 混合分析
- 结合传统规则分析和ML预测
- 可调节的权重配置
- 多层次分析结果

## 开发说明

### 分支说明
- `master`: 原始版本（已废弃传统分析模块）
- `machine-learning`: 当前机器学习增强版本

### 贡献指南
1. Fork项目
2. 创建特性分支
3. 提交更改
4. 发起Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。
