#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作质量预测器
使用机器学习模型预测动作质量和错误类型
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

from .ml_feature_extractor import MLFeatureExtractor

class ActionQualityPredictor:
    """
    动作质量预测器
    使用机器学习模型预测羽毛球动作的质量分数和错误类型
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化动作质量预测器
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 特征提取器
        self.feature_extractor = MLFeatureExtractor()
        
        # 模型组件
        self.quality_regressor = None      # 质量分数回归模型
        self.error_classifier = None       # 错误类型分类模型
        self.scaler = None                 # 特征标准化器
        self.label_encoder = None          # 标签编码器
        
        # 模型配置
        self.model_config = {
            'quality_model': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'error_model': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            }
        }
        
        # 错误类型定义
        self.error_types = [
            'normal',           # 正常动作
            'timing_early',     # 时机过早
            'timing_late',      # 时机过晚
            'angle_elbow',      # 肘部角度问题
            'angle_shoulder',   # 肩部角度问题
            'trajectory_high',  # 轨迹过高
            'trajectory_low',   # 轨迹过低
            'speed_fast',       # 速度过快
            'speed_slow',       # 速度过慢
            'coordination'      # 协调性问题
        ]
        
        # 质量分数范围
        self.quality_range = (0, 100)
        
        # 加载已有模型
        self._load_models()
        
        print("动作质量预测器初始化完成")
    
    def predict_quality(self, landmarks_timeline: List[Dict]) -> Dict[str, Any]:
        """
        预测动作质量
        
        Args:
            landmarks_timeline: 姿态时间线数据
            
        Returns:
            预测结果字典
        """
        if not landmarks_timeline:
            return self._get_default_prediction()
        
        # 提取特征
        features = self.feature_extractor.extract_features(landmarks_timeline)
        
        if len(features) == 0:
            return self._get_default_prediction()
        
        # 特征标准化
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        prediction_result = {
            'quality_score': 50,  # 默认分数
            'quality_level': '一般',
            'error_type': 'normal',
            'error_probability': 0.5,
            'confidence': 0.5,
            'detailed_scores': {},
            'recommendations': []
        }
        
        # 质量分数预测
        if self.quality_regressor is not None:
            try:
                quality_score = self.quality_regressor.predict(features_scaled)[0]
                quality_score = np.clip(quality_score, self.quality_range[0], self.quality_range[1])
                prediction_result['quality_score'] = float(quality_score)
                prediction_result['quality_level'] = self._score_to_level(quality_score)
                
                # 如果是随机森林，计算预测置信度
                if hasattr(self.quality_regressor, 'estimators_'):
                    predictions = [tree.predict(features_scaled)[0] for tree in self.quality_regressor.estimators_]
                    prediction_result['confidence'] = 1.0 - (np.std(predictions) / 100.0)
                    prediction_result['confidence'] = np.clip(prediction_result['confidence'], 0, 1)
                
            except Exception as e:
                print(f"质量预测错误: {e}")
        
        # 错误类型预测
        if self.error_classifier is not None:
            try:
                error_probs = self.error_classifier.predict_proba(features_scaled)[0]
                error_pred = self.error_classifier.predict(features_scaled)[0]
                
                # 解码错误类型
                if self.label_encoder is not None:
                    error_type = self.label_encoder.inverse_transform([error_pred])[0]
                else:
                    error_type = self.error_types[error_pred] if error_pred < len(self.error_types) else 'normal'
                
                prediction_result['error_type'] = error_type
                prediction_result['error_probability'] = float(np.max(error_probs))
                
                # 详细的错误概率
                if self.label_encoder is not None:
                    error_classes = self.label_encoder.classes_
                else:
                    error_classes = self.error_types[:len(error_probs)]
                
                prediction_result['detailed_scores'] = {
                    error_classes[i]: float(prob) for i, prob in enumerate(error_probs)
                }
                
            except Exception as e:
                print(f"错误类型预测错误: {e}")
        
        # 生成建议
        prediction_result['recommendations'] = self._generate_recommendations(
            prediction_result['quality_score'],
            prediction_result['error_type'],
            prediction_result.get('detailed_scores', {})
        )
        
        return prediction_result
    
    def train_models(self, training_data: List[Dict[str, Any]], 
                    validation_split: float = 0.2) -> Dict[str, Any]:
        """
        训练机器学习模型
        
        Args:
            training_data: 训练数据列表，每个元素包含 'landmarks', 'quality_score', 'error_type'
            validation_split: 验证集比例
            
        Returns:
            训练结果字典
        """
        if len(training_data) < 10:
            return {'success': False, 'message': '训练数据不足，至少需要10个样本'}
        
        print(f"开始训练模型，数据量: {len(training_data)}")
        
        # 提取特征和标签
        features_list = []
        quality_scores = []
        error_types = []
        
        for data_item in training_data:
            landmarks = data_item.get('landmarks', [])
            quality_score = data_item.get('quality_score', 50)
            error_type = data_item.get('error_type', 'normal')
            
            if landmarks:
                features = self.feature_extractor.extract_features(landmarks)
                if len(features) > 0:
                    features_list.append(features)
                    quality_scores.append(quality_score)
                    error_types.append(error_type)
        
        if len(features_list) < 5:
            return {'success': False, 'message': '有效特征数据不足'}
        
        # 转换为numpy数组
        X = np.array(features_list)
        y_quality = np.array(quality_scores)
        y_error = np.array(error_types)
        
        print(f"特征维度: {X.shape}, 质量标签: {len(y_quality)}, 错误标签: {len(y_error)}")
        
        # 特征标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 错误类型编码
        self.label_encoder = LabelEncoder()
        y_error_encoded = self.label_encoder.fit_transform(y_error)
        
        # 分割训练和验证集
        X_train, X_val, y_quality_train, y_quality_val, y_error_train, y_error_val = train_test_split(
            X_scaled, y_quality, y_error_encoded, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_error_encoded
        )
        
        training_results = {
            'success': True,
            'data_info': {
                'total_samples': len(X),
                'feature_dimension': X.shape[1],
                'train_samples': len(X_train),
                'val_samples': len(X_val)
            },
            'quality_model': {},
            'error_model': {}
        }
        
        # 训练质量回归模型
        try:
            print("训练质量回归模型...")
            self.quality_regressor = RandomForestRegressor(**self.model_config['quality_model']['params'])
            self.quality_regressor.fit(X_train, y_quality_train)
            
            # 验证质量模型
            y_quality_pred = self.quality_regressor.predict(X_val)
            quality_mse = mean_squared_error(y_quality_val, y_quality_pred)
            quality_rmse = np.sqrt(quality_mse)
            
            training_results['quality_model'] = {
                'mse': float(quality_mse),
                'rmse': float(quality_rmse),
                'feature_importance': self.quality_regressor.feature_importances_.tolist()
            }
            
            print(f"质量模型 RMSE: {quality_rmse:.2f}")
            
        except Exception as e:
            print(f"质量模型训练失败: {e}")
            training_results['quality_model']['error'] = str(e)
        
        # 训练错误分类模型
        try:
            print("训练错误分类模型...")
            self.error_classifier = RandomForestClassifier(**self.model_config['error_model']['params'])
            self.error_classifier.fit(X_train, y_error_train)
            
            # 验证错误模型
            y_error_pred = self.error_classifier.predict(X_val)
            error_accuracy = accuracy_score(y_error_val, y_error_pred)
            
            training_results['error_model'] = {
                'accuracy': float(error_accuracy),
                'feature_importance': self.error_classifier.feature_importances_.tolist(),
                'classes': self.label_encoder.classes_.tolist()
            }
            
            print(f"错误分类模型准确率: {error_accuracy:.3f}")
            
        except Exception as e:
            print(f"错误分类模型训练失败: {e}")
            training_results['error_model']['error'] = str(e)
        
        # 保存模型
        try:
            self._save_models()
            training_results['model_saved'] = True
            print("模型保存成功")
        except Exception as e:
            print(f"模型保存失败: {e}")
            training_results['model_saved'] = False
            training_results['save_error'] = str(e)
        
        return training_results
    
    def update_model_online(self, new_data: Dict[str, Any]) -> bool:
        """
        在线更新模型（增量学习）
        
        Args:
            new_data: 新的训练数据
            
        Returns:
            是否更新成功
        """
        # 简化的在线学习实现
        # 实际应用中可以使用更复杂的增量学习算法
        
        landmarks = new_data.get('landmarks', [])
        quality_score = new_data.get('quality_score')
        error_type = new_data.get('error_type')
        
        if not landmarks or quality_score is None or not error_type:
            return False
        
        try:
            # 提取特征
            features = self.feature_extractor.extract_features(landmarks)
            if len(features) == 0:
                return False
            
            # 标准化特征
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                return False
            
            # 编码错误类型
            if self.label_encoder is not None and error_type in self.label_encoder.classes_:
                error_encoded = self.label_encoder.transform([error_type])[0]
            else:
                return False
            
            # 更新模型（这里使用简单的重新训练一个样本）
            # 实际应用中应该累积多个样本后批量更新
            if (self.quality_regressor is not None and 
                hasattr(self.quality_regressor, 'n_estimators')):
                
                # 对于随机森林，可以添加新的树（简化实现）
                # 这里只是示例，实际需要更复杂的增量学习策略
                pass
            
            return True
            
        except Exception as e:
            print(f"在线更新失败: {e}")
            return False
    
    def evaluate_model(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据
            
        Returns:
            评估结果
        """
        if not test_data or len(test_data) < 5:
            return {'success': False, 'message': '测试数据不足'}
        
        # 提取测试特征和标签
        features_list = []
        quality_scores = []
        error_types = []
        
        for data_item in test_data:
            landmarks = data_item.get('landmarks', [])
            quality_score = data_item.get('quality_score', 50)
            error_type = data_item.get('error_type', 'normal')
            
            if landmarks:
                features = self.feature_extractor.extract_features(landmarks)
                if len(features) > 0:
                    features_list.append(features)
                    quality_scores.append(quality_score)
                    error_types.append(error_type)
        
        if len(features_list) < 3:
            return {'success': False, 'message': '有效测试数据不足'}
        
        X_test = np.array(features_list)
        y_quality_test = np.array(quality_scores)
        y_error_test = np.array(error_types)
        
        evaluation_results = {
            'success': True,
            'test_samples': len(X_test),
            'quality_metrics': {},
            'error_metrics': {}
        }
        
        # 评估质量回归模型
        if self.quality_regressor is not None and self.scaler is not None:
            try:
                X_test_scaled = self.scaler.transform(X_test)
                y_quality_pred = self.quality_regressor.predict(X_test_scaled)
                
                quality_mse = mean_squared_error(y_quality_test, y_quality_pred)
                quality_rmse = np.sqrt(quality_mse)
                quality_mae = np.mean(np.abs(y_quality_test - y_quality_pred))
                
                evaluation_results['quality_metrics'] = {
                    'mse': float(quality_mse),
                    'rmse': float(quality_rmse),
                    'mae': float(quality_mae)
                }
                
            except Exception as e:
                evaluation_results['quality_metrics']['error'] = str(e)
        
        # 评估错误分类模型
        if (self.error_classifier is not None and 
            self.scaler is not None and 
            self.label_encoder is not None):
            try:
                X_test_scaled = self.scaler.transform(X_test)
                
                # 过滤测试数据中的未知错误类型
                valid_indices = []
                y_error_test_filtered = []
                
                for i, error_type in enumerate(y_error_test):
                    if error_type in self.label_encoder.classes_:
                        valid_indices.append(i)
                        y_error_test_filtered.append(error_type)
                
                if len(valid_indices) > 0:
                    X_test_filtered = X_test_scaled[valid_indices]
                    y_error_test_encoded = self.label_encoder.transform(y_error_test_filtered)
                    y_error_pred = self.error_classifier.predict(X_test_filtered)
                    
                    error_accuracy = accuracy_score(y_error_test_encoded, y_error_pred)
                    
                    evaluation_results['error_metrics'] = {
                        'accuracy': float(error_accuracy),
                        'valid_samples': len(valid_indices)
                    }
                
            except Exception as e:
                evaluation_results['error_metrics']['error'] = str(e)
        
        return evaluation_results
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """
        获取默认预测结果
        
        Returns:
            默认预测结果
        """
        return {
            'quality_score': 50,
            'quality_level': '无法评估',
            'error_type': 'normal',
            'error_probability': 0.5,
            'confidence': 0.0,
            'detailed_scores': {},
            'recommendations': ['数据不足，无法进行准确评估']
        }
    
    def _score_to_level(self, score: float) -> str:
        """
        将分数转换为等级
        
        Args:
            score: 质量分数
            
        Returns:
            质量等级
        """
        if score >= 85:
            return '优秀'
        elif score >= 75:
            return '良好'
        elif score >= 60:
            return '一般'
        else:
            return '需改进'
    
    def _generate_recommendations(self, quality_score: float, 
                                error_type: str, 
                                detailed_scores: Dict[str, float]) -> List[str]:
        """
        生成改进建议
        
        Args:
            quality_score: 质量分数
            error_type: 主要错误类型
            detailed_scores: 详细错误概率
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于质量分数的建议
        if quality_score < 60:
            recommendations.append("整体动作质量需要显著改进")
        elif quality_score < 75:
            recommendations.append("动作基本正确，但仍有改进空间")
        elif quality_score < 85:
            recommendations.append("动作质量良好，继续保持并精进细节")
        else:
            recommendations.append("动作质量优秀，保持当前水平")
        
        # 基于错误类型的具体建议
        error_recommendations = {
            'timing_early': "注意击球时机，避免过早出手",
            'timing_late': "提高反应速度，避免击球时机过晚",
            'angle_elbow': "调整肘部角度，保持合适的弯曲度",
            'angle_shoulder': "注意肩部姿态，保持正确的角度",
            'trajectory_high': "降低击球轨迹，控制球的高度",
            'trajectory_low': "提高击球轨迹，增加球的高度",
            'speed_fast': "控制击球速度，避免过于急躁",
            'speed_slow': "增加击球速度，提高动作的爆发力",
            'coordination': "加强身体协调性训练，提高动作流畅度"
        }
        
        if error_type in error_recommendations:
            recommendations.append(error_recommendations[error_type])
        
        # 基于详细错误概率的建议
        if detailed_scores:
            high_prob_errors = [(error, prob) for error, prob in detailed_scores.items() 
                              if prob > 0.3 and error != 'normal']
            
            for error, prob in sorted(high_prob_errors, key=lambda x: x[1], reverse=True)[:2]:
                if error in error_recommendations:
                    recommendations.append(f"注意{error_recommendations[error]}（置信度: {prob:.1%}）")
        
        # 通用建议
        if len(recommendations) == 1:  # 只有质量分数建议
            recommendations.extend([
                "建议多练习基本动作，注意动作的标准性",
                "可以录制视频进行自我分析和改进"
            ])
        
        return recommendations[:5]  # 最多返回5条建议
    
    def _save_models(self):
        """
        保存训练好的模型
        """
        # 保存质量回归模型
        if self.quality_regressor is not None:
            joblib.dump(self.quality_regressor, self.model_dir / 'quality_regressor.pkl')
        
        # 保存错误分类模型
        if self.error_classifier is not None:
            joblib.dump(self.error_classifier, self.model_dir / 'error_classifier.pkl')
        
        # 保存标准化器
        if self.scaler is not None:
            joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        
        # 保存标签编码器
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.pkl')
        
        # 保存配置信息
        config_info = {
            'error_types': self.error_types,
            'quality_range': self.quality_range,
            'feature_dimension': self.feature_extractor.get_feature_dimension(),
            'model_config': self.model_config
        }
        
        with open(self.model_dir / 'config.pkl', 'wb') as f:
            pickle.dump(config_info, f)
    
    def _load_models(self):
        """
        加载已保存的模型
        """
        try:
            # 加载质量回归模型
            quality_model_path = self.model_dir / 'quality_regressor.pkl'
            if quality_model_path.exists():
                self.quality_regressor = joblib.load(quality_model_path)
                print("质量回归模型加载成功")
            
            # 加载错误分类模型
            error_model_path = self.model_dir / 'error_classifier.pkl'
            if error_model_path.exists():
                self.error_classifier = joblib.load(error_model_path)
                print("错误分类模型加载成功")
            
            # 加载标准化器
            scaler_path = self.model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("特征标准化器加载成功")
            
            # 加载标签编码器
            encoder_path = self.model_dir / 'label_encoder.pkl'
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                print("标签编码器加载成功")
            
            # 加载配置信息
            config_path = self.model_dir / 'config.pkl'
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    config_info = pickle.load(f)
                    self.error_types = config_info.get('error_types', self.error_types)
                    self.quality_range = config_info.get('quality_range', self.quality_range)
                print("配置信息加载成功")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'feature_dimension': self.feature_extractor.get_feature_dimension(),
            'error_types': self.error_types,
            'quality_range': self.quality_range,
            'models_loaded': {
                'quality_regressor': self.quality_regressor is not None,
                'error_classifier': self.error_classifier is not None,
                'scaler': self.scaler is not None,
                'label_encoder': self.label_encoder is not None
            }
        }
        
        # 添加模型具体信息
        if self.quality_regressor is not None:
            info['quality_model_info'] = {
                'type': type(self.quality_regressor).__name__,
                'n_features': getattr(self.quality_regressor, 'n_features_in_', 'unknown')
            }
        
        if self.error_classifier is not None:
            info['error_model_info'] = {
                'type': type(self.error_classifier).__name__,
                'n_classes': getattr(self.error_classifier, 'n_classes_', 'unknown'),
                'n_features': getattr(self.error_classifier, 'n_features_in_', 'unknown')
            }
        
        if self.label_encoder is not None:
            info['label_encoder_classes'] = self.label_encoder.classes_.tolist()
        
        return info

# 导出主要类
__all__ = ['ActionQualityPredictor']