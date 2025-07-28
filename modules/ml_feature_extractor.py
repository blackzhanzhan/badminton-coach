#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习特征提取器
从姿态数据中提取用于机器学习的特征向量
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy import signal, stats
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

class MLFeatureExtractor:
    """
    机器学习特征提取器
    从姿态序列数据中提取多维特征向量，用于机器学习模型训练和预测
    """
    
    def __init__(self):
        """
        初始化特征提取器
        """
        # 关键点映射（与MediaPipe一致）
        self.landmark_indices = {
            'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7, 'RIGHT_EAR': 8, 'MOUTH_LEFT': 9, 'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12, 'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14, 'LEFT_WRIST': 15, 'RIGHT_WRIST': 16,
            'LEFT_PINKY': 17, 'RIGHT_PINKY': 18, 'LEFT_INDEX': 19,
            'RIGHT_INDEX': 20, 'LEFT_THUMB': 21, 'RIGHT_THUMB': 22,
            'LEFT_HIP': 23, 'RIGHT_HIP': 24, 'LEFT_KNEE': 25, 'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27, 'RIGHT_ANKLE': 28, 'LEFT_HEEL': 29,
            'RIGHT_HEEL': 30, 'LEFT_FOOT_INDEX': 31, 'RIGHT_FOOT_INDEX': 32
        }
        
        # 特征配置
        self.feature_config = {
            'angle_features': True,
            'distance_features': True,
            'velocity_features': True,
            'acceleration_features': True,
            'temporal_features': True,
            'statistical_features': True,
            'frequency_features': True,
            'coordination_features': True
        }
        
        # 关键角度定义
        self.key_angles = {
            'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            'left_elbow': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_shoulder': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            'left_shoulder': ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
            'right_hip': ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
            'left_hip': ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
            'right_knee': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            'left_knee': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'torso_lean': ('LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            'arm_span': ('LEFT_WRIST', 'LEFT_SHOULDER', 'RIGHT_WRIST')
        }
        
        # 关键距离定义
        self.key_distances = {
            'shoulder_width': ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            'hip_width': ('LEFT_HIP', 'RIGHT_HIP'),
            'arm_length_right': ('RIGHT_SHOULDER', 'RIGHT_WRIST'),
            'arm_length_left': ('LEFT_SHOULDER', 'LEFT_WRIST'),
            'torso_length': ('LEFT_SHOULDER', 'LEFT_HIP'),
            'leg_length_right': ('RIGHT_HIP', 'RIGHT_ANKLE'),
            'leg_length_left': ('LEFT_HIP', 'LEFT_ANKLE'),
            'hand_separation': ('LEFT_WRIST', 'RIGHT_WRIST')
        }
        
        # 关键点用于速度计算
        self.velocity_points = [
            'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_ELBOW', 'LEFT_ELBOW',
            'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_HIP', 'LEFT_HIP'
        ]
        
        print("机器学习特征提取器初始化完成")
    
    def extract_features(self, landmarks_timeline: List[Dict], 
                        feature_types: Optional[List[str]] = None) -> np.ndarray:
        """
        从姿态时间线提取特征向量
        
        Args:
            landmarks_timeline: 姿态时间线数据
            feature_types: 要提取的特征类型列表，None表示提取所有特征
            
        Returns:
            特征向量 (numpy array)
        """
        if not landmarks_timeline:
            return np.array([])
        
        # 预处理数据
        processed_data = self._preprocess_data(landmarks_timeline)
        
        if len(processed_data) < 2:
            return np.array([])
        
        # 提取各类特征
        all_features = []
        
        if feature_types is None:
            feature_types = list(self.feature_config.keys())
        
        for feature_type in feature_types:
            if self.feature_config.get(feature_type, False):
                features = self._extract_feature_type(processed_data, feature_type)
                if len(features) > 0:
                    all_features.extend(features)
        
        return np.array(all_features) if all_features else np.array([])
    
    def extract_features_with_labels(self, landmarks_timeline: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        提取特征向量并返回特征标签
        
        Args:
            landmarks_timeline: 姿态时间线数据
            
        Returns:
            (特征向量, 特征标签列表)
        """
        if not landmarks_timeline:
            return np.array([]), []
        
        processed_data = self._preprocess_data(landmarks_timeline)
        
        if len(processed_data) < 2:
            return np.array([]), []
        
        all_features = []
        all_labels = []
        
        for feature_type in self.feature_config.keys():
            if self.feature_config[feature_type]:
                features, labels = self._extract_feature_type_with_labels(processed_data, feature_type)
                all_features.extend(features)
                all_labels.extend(labels)
        
        return np.array(all_features), all_labels
    
    def _preprocess_data(self, landmarks_timeline: List[Dict]) -> List[Dict]:
        """
        预处理姿态数据
        
        Args:
            landmarks_timeline: 原始姿态时间线
            
        Returns:
            预处理后的数据
        """
        processed_data = []
        
        for frame_data in landmarks_timeline:
            landmarks = frame_data.get('landmarks', {})
            
            # 检查关键点是否存在
            if not self._has_required_landmarks(landmarks):
                continue
            
            # 标准化坐标（相对于肩膀中点）
            normalized_landmarks = self._normalize_landmarks(landmarks)
            
            processed_frame = {
                'timestamp_ms': frame_data.get('timestamp_ms', 0),
                'landmarks': normalized_landmarks,
                'original_landmarks': landmarks
            }
            
            processed_data.append(processed_frame)
        
        return processed_data
    
    def _has_required_landmarks(self, landmarks: Dict) -> bool:
        """
        检查是否包含必需的关键点
        
        Args:
            landmarks: 关键点数据
            
        Returns:
            是否包含必需关键点
        """
        required_points = [
            'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW',
            'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP'
        ]
        
        for point_name in required_points:
            point = self._get_landmark(landmarks, point_name)
            if not point or point.get('visibility', 0) < 0.5:
                return False
        
        return True
    
    def _normalize_landmarks(self, landmarks: Dict) -> Dict:
        """
        标准化关键点坐标
        
        Args:
            landmarks: 原始关键点
            
        Returns:
            标准化后的关键点
        """
        # 计算肩膀中点作为参考点
        left_shoulder = self._get_landmark(landmarks, 'LEFT_SHOULDER')
        right_shoulder = self._get_landmark(landmarks, 'RIGHT_SHOULDER')
        
        if not left_shoulder or not right_shoulder:
            return landmarks
        
        center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        
        # 计算肩膀宽度作为缩放因子
        shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])
        scale_factor = shoulder_width if shoulder_width > 0 else 1.0
        
        normalized_landmarks = {}
        
        for name, point in landmarks.items():
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                normalized_landmarks[name] = {
                    'x': (point['x'] - center_x) / scale_factor,
                    'y': (point['y'] - center_y) / scale_factor,
                    'z': point.get('z', 0) / scale_factor,
                    'visibility': point.get('visibility', 1.0)
                }
            else:
                normalized_landmarks[name] = point
        
        return normalized_landmarks
    
    def _get_landmark(self, landmarks: Dict, name: str) -> Optional[Dict]:
        """
        获取指定名称的关键点
        
        Args:
            landmarks: 关键点字典
            name: 关键点名称
            
        Returns:
            关键点数据或None
        """
        return landmarks.get(name)
    
    def _extract_feature_type(self, data: List[Dict], feature_type: str) -> List[float]:
        """
        提取指定类型的特征
        
        Args:
            data: 预处理后的数据
            feature_type: 特征类型
            
        Returns:
            特征值列表
        """
        if feature_type == 'angle_features':
            return self._extract_angle_features(data)
        elif feature_type == 'distance_features':
            return self._extract_distance_features(data)
        elif feature_type == 'velocity_features':
            return self._extract_velocity_features(data)
        elif feature_type == 'acceleration_features':
            return self._extract_acceleration_features(data)
        elif feature_type == 'temporal_features':
            return self._extract_temporal_features(data)
        elif feature_type == 'statistical_features':
            return self._extract_statistical_features(data)
        elif feature_type == 'frequency_features':
            return self._extract_frequency_features(data)
        elif feature_type == 'coordination_features':
            return self._extract_coordination_features(data)
        else:
            return []
    
    def _extract_feature_type_with_labels(self, data: List[Dict], 
                                        feature_type: str) -> Tuple[List[float], List[str]]:
        """
        提取指定类型的特征并返回标签
        
        Args:
            data: 预处理后的数据
            feature_type: 特征类型
            
        Returns:
            (特征值列表, 特征标签列表)
        """
        if feature_type == 'angle_features':
            return self._extract_angle_features_with_labels(data)
        elif feature_type == 'distance_features':
            return self._extract_distance_features_with_labels(data)
        elif feature_type == 'velocity_features':
            return self._extract_velocity_features_with_labels(data)
        elif feature_type == 'acceleration_features':
            return self._extract_acceleration_features_with_labels(data)
        elif feature_type == 'temporal_features':
            return self._extract_temporal_features_with_labels(data)
        elif feature_type == 'statistical_features':
            return self._extract_statistical_features_with_labels(data)
        elif feature_type == 'frequency_features':
            return self._extract_frequency_features_with_labels(data)
        elif feature_type == 'coordination_features':
            return self._extract_coordination_features_with_labels(data)
        else:
            return [], []
    
    def _extract_angle_features(self, data: List[Dict]) -> List[float]:
        """
        提取角度特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            角度特征列表
        """
        features = []
        
        # 收集所有帧的角度数据
        angle_sequences = {angle_name: [] for angle_name in self.key_angles.keys()}
        
        for frame in data:
            landmarks = frame['landmarks']
            
            for angle_name, (p1_name, p2_name, p3_name) in self.key_angles.items():
                p1 = self._get_landmark(landmarks, p1_name)
                p2 = self._get_landmark(landmarks, p2_name)
                p3 = self._get_landmark(landmarks, p3_name)
                
                if p1 and p2 and p3:
                    angle = self._calculate_angle(p1, p2, p3)
                    if angle is not None:
                        angle_sequences[angle_name].append(angle)
        
        # 计算每个角度的统计特征
        for angle_name, angles in angle_sequences.items():
            if len(angles) > 0:
                features.extend([
                    np.mean(angles),      # 平均角度
                    np.std(angles),       # 角度标准差
                    np.min(angles),       # 最小角度
                    np.max(angles),       # 最大角度
                    np.max(angles) - np.min(angles)  # 角度范围
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
        
        return features
    
    def _extract_angle_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取角度特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (角度特征列表, 特征标签列表)
        """
        features = []
        labels = []
        
        # 收集所有帧的角度数据
        angle_sequences = {angle_name: [] for angle_name in self.key_angles.keys()}
        
        for frame in data:
            landmarks = frame['landmarks']
            
            for angle_name, (p1_name, p2_name, p3_name) in self.key_angles.items():
                p1 = self._get_landmark(landmarks, p1_name)
                p2 = self._get_landmark(landmarks, p2_name)
                p3 = self._get_landmark(landmarks, p3_name)
                
                if p1 and p2 and p3:
                    angle = self._calculate_angle(p1, p2, p3)
                    if angle is not None:
                        angle_sequences[angle_name].append(angle)
        
        # 计算每个角度的统计特征
        for angle_name, angles in angle_sequences.items():
            if len(angles) > 0:
                features.extend([
                    np.mean(angles),
                    np.std(angles),
                    np.min(angles),
                    np.max(angles),
                    np.max(angles) - np.min(angles)
                ])
                labels.extend([
                    f'{angle_name}_mean',
                    f'{angle_name}_std',
                    f'{angle_name}_min',
                    f'{angle_name}_max',
                    f'{angle_name}_range'
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
                labels.extend([
                    f'{angle_name}_mean',
                    f'{angle_name}_std',
                    f'{angle_name}_min',
                    f'{angle_name}_max',
                    f'{angle_name}_range'
                ])
        
        return features, labels
    
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> Optional[float]:
        """
        计算三点构成的角度
        
        Args:
            p1, p2, p3: 三个点的坐标字典
            
        Returns:
            角度值（度）或None
        """
        try:
            # 创建向量
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            # 计算角度
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            
            return float(angle)
        except:
            return None
    
    def _extract_distance_features(self, data: List[Dict]) -> List[float]:
        """
        提取距离特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            距离特征列表
        """
        features = []
        
        # 收集所有帧的距离数据
        distance_sequences = {dist_name: [] for dist_name in self.key_distances.keys()}
        
        for frame in data:
            landmarks = frame['landmarks']
            
            for dist_name, (p1_name, p2_name) in self.key_distances.items():
                p1 = self._get_landmark(landmarks, p1_name)
                p2 = self._get_landmark(landmarks, p2_name)
                
                if p1 and p2:
                    distance = self._calculate_distance(p1, p2)
                    distance_sequences[dist_name].append(distance)
        
        # 计算每个距离的统计特征
        for dist_name, distances in distance_sequences.items():
            if len(distances) > 0:
                features.extend([
                    np.mean(distances),   # 平均距离
                    np.std(distances),    # 距离标准差
                    np.min(distances),    # 最小距离
                    np.max(distances)     # 最大距离
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_distance_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取距离特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (距离特征列表, 特征标签列表)
        """
        features = []
        labels = []
        
        # 收集所有帧的距离数据
        distance_sequences = {dist_name: [] for dist_name in self.key_distances.keys()}
        
        for frame in data:
            landmarks = frame['landmarks']
            
            for dist_name, (p1_name, p2_name) in self.key_distances.items():
                p1 = self._get_landmark(landmarks, p1_name)
                p2 = self._get_landmark(landmarks, p2_name)
                
                if p1 and p2:
                    distance = self._calculate_distance(p1, p2)
                    distance_sequences[dist_name].append(distance)
        
        # 计算每个距离的统计特征
        for dist_name, distances in distance_sequences.items():
            if len(distances) > 0:
                features.extend([
                    np.mean(distances),
                    np.std(distances),
                    np.min(distances),
                    np.max(distances)
                ])
                labels.extend([
                    f'{dist_name}_mean',
                    f'{dist_name}_std',
                    f'{dist_name}_min',
                    f'{dist_name}_max'
                ])
            else:
                features.extend([0, 0, 0, 0])
                labels.extend([
                    f'{dist_name}_mean',
                    f'{dist_name}_std',
                    f'{dist_name}_min',
                    f'{dist_name}_max'
                ])
        
        return features, labels
    
    def _calculate_distance(self, p1: Dict, p2: Dict) -> float:
        """
        计算两点间距离
        
        Args:
            p1, p2: 两个点的坐标字典
            
        Returns:
            距离值
        """
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        return np.sqrt(dx**2 + dy**2)
    
    def _extract_velocity_features(self, data: List[Dict]) -> List[float]:
        """
        提取速度特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            速度特征列表
        """
        features = []
        
        if len(data) < 2:
            return [0] * (len(self.velocity_points) * 4)  # 每个点4个统计特征
        
        # 计算每个关键点的速度序列
        velocity_sequences = {point: [] for point in self.velocity_points}
        
        for i in range(1, len(data)):
            prev_landmarks = data[i-1]['landmarks']
            curr_landmarks = data[i]['landmarks']
            
            for point_name in self.velocity_points:
                prev_point = self._get_landmark(prev_landmarks, point_name)
                curr_point = self._get_landmark(curr_landmarks, point_name)
                
                if prev_point and curr_point:
                    velocity = self._calculate_velocity(prev_point, curr_point)
                    velocity_sequences[point_name].append(velocity)
        
        # 计算每个点的速度统计特征
        for point_name in self.velocity_points:
            velocities = velocity_sequences[point_name]
            if len(velocities) > 0:
                features.extend([
                    np.mean(velocities),  # 平均速度
                    np.std(velocities),   # 速度标准差
                    np.max(velocities),   # 最大速度
                    len([v for v in velocities if v > np.mean(velocities)]) / len(velocities)  # 高速度比例
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        return features
    
    def _extract_velocity_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取速度特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (速度特征列表, 特征标签列表)
        """
        features = []
        labels = []
        
        if len(data) < 2:
            for point_name in self.velocity_points:
                features.extend([0, 0, 0, 0])
                labels.extend([
                    f'{point_name.lower()}_velocity_mean',
                    f'{point_name.lower()}_velocity_std',
                    f'{point_name.lower()}_velocity_max',
                    f'{point_name.lower()}_velocity_high_ratio'
                ])
            return features, labels
        
        # 计算每个关键点的速度序列
        velocity_sequences = {point: [] for point in self.velocity_points}
        
        for i in range(1, len(data)):
            prev_landmarks = data[i-1]['landmarks']
            curr_landmarks = data[i]['landmarks']
            
            for point_name in self.velocity_points:
                prev_point = self._get_landmark(prev_landmarks, point_name)
                curr_point = self._get_landmark(curr_landmarks, point_name)
                
                if prev_point and curr_point:
                    velocity = self._calculate_velocity(prev_point, curr_point)
                    velocity_sequences[point_name].append(velocity)
        
        # 计算每个点的速度统计特征
        for point_name in self.velocity_points:
            velocities = velocity_sequences[point_name]
            if len(velocities) > 0:
                features.extend([
                    np.mean(velocities),
                    np.std(velocities),
                    np.max(velocities),
                    len([v for v in velocities if v > np.mean(velocities)]) / len(velocities)
                ])
                labels.extend([
                    f'{point_name.lower()}_velocity_mean',
                    f'{point_name.lower()}_velocity_std',
                    f'{point_name.lower()}_velocity_max',
                    f'{point_name.lower()}_velocity_high_ratio'
                ])
            else:
                features.extend([0, 0, 0, 0])
                labels.extend([
                    f'{point_name.lower()}_velocity_mean',
                    f'{point_name.lower()}_velocity_std',
                    f'{point_name.lower()}_velocity_max',
                    f'{point_name.lower()}_velocity_high_ratio'
                ])
        
        return features, labels
    
    def _calculate_velocity(self, prev_point: Dict, curr_point: Dict) -> float:
        """
        计算点的速度
        
        Args:
            prev_point: 前一帧的点
            curr_point: 当前帧的点
            
        Returns:
            速度值
        """
        dx = curr_point['x'] - prev_point['x']
        dy = curr_point['y'] - prev_point['y']
        return np.sqrt(dx**2 + dy**2)
    
    def _extract_acceleration_features(self, data: List[Dict]) -> List[float]:
        """
        提取加速度特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            加速度特征列表
        """
        features = []
        
        if len(data) < 3:
            return [0] * (len(self.velocity_points) * 3)  # 每个点3个统计特征
        
        # 计算加速度序列
        acceleration_sequences = {point: [] for point in self.velocity_points}
        
        for i in range(2, len(data)):
            prev2_landmarks = data[i-2]['landmarks']
            prev_landmarks = data[i-1]['landmarks']
            curr_landmarks = data[i]['landmarks']
            
            for point_name in self.velocity_points:
                prev2_point = self._get_landmark(prev2_landmarks, point_name)
                prev_point = self._get_landmark(prev_landmarks, point_name)
                curr_point = self._get_landmark(curr_landmarks, point_name)
                
                if prev2_point and prev_point and curr_point:
                    acceleration = self._calculate_acceleration(prev2_point, prev_point, curr_point)
                    acceleration_sequences[point_name].append(acceleration)
        
        # 计算加速度统计特征
        for point_name in self.velocity_points:
            accelerations = acceleration_sequences[point_name]
            if len(accelerations) > 0:
                features.extend([
                    np.mean(np.abs(accelerations)),  # 平均加速度幅值
                    np.std(accelerations),           # 加速度标准差
                    np.max(np.abs(accelerations))    # 最大加速度幅值
                ])
            else:
                features.extend([0, 0, 0])
        
        return features
    
    def _extract_acceleration_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取加速度特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (加速度特征列表, 特征标签列表)
        """
        features = []
        labels = []
        
        if len(data) < 3:
            for point_name in self.velocity_points:
                features.extend([0, 0, 0])
                labels.extend([
                    f'{point_name.lower()}_acceleration_mean',
                    f'{point_name.lower()}_acceleration_std',
                    f'{point_name.lower()}_acceleration_max'
                ])
            return features, labels
        
        # 计算加速度序列
        acceleration_sequences = {point: [] for point in self.velocity_points}
        
        for i in range(2, len(data)):
            prev2_landmarks = data[i-2]['landmarks']
            prev_landmarks = data[i-1]['landmarks']
            curr_landmarks = data[i]['landmarks']
            
            for point_name in self.velocity_points:
                prev2_point = self._get_landmark(prev2_landmarks, point_name)
                prev_point = self._get_landmark(prev_landmarks, point_name)
                curr_point = self._get_landmark(curr_landmarks, point_name)
                
                if prev2_point and prev_point and curr_point:
                    acceleration = self._calculate_acceleration(prev2_point, prev_point, curr_point)
                    acceleration_sequences[point_name].append(acceleration)
        
        # 计算加速度统计特征
        for point_name in self.velocity_points:
            accelerations = acceleration_sequences[point_name]
            if len(accelerations) > 0:
                features.extend([
                    np.mean(np.abs(accelerations)),
                    np.std(accelerations),
                    np.max(np.abs(accelerations))
                ])
                labels.extend([
                    f'{point_name.lower()}_acceleration_mean',
                    f'{point_name.lower()}_acceleration_std',
                    f'{point_name.lower()}_acceleration_max'
                ])
            else:
                features.extend([0, 0, 0])
                labels.extend([
                    f'{point_name.lower()}_acceleration_mean',
                    f'{point_name.lower()}_acceleration_std',
                    f'{point_name.lower()}_acceleration_max'
                ])
        
        return features, labels
    
    def _calculate_acceleration(self, prev2_point: Dict, prev_point: Dict, curr_point: Dict) -> float:
        """
        计算点的加速度
        
        Args:
            prev2_point: 前两帧的点
            prev_point: 前一帧的点
            curr_point: 当前帧的点
            
        Returns:
            加速度值
        """
        # 计算前后两个速度
        v1 = self._calculate_velocity(prev2_point, prev_point)
        v2 = self._calculate_velocity(prev_point, curr_point)
        
        # 加速度为速度变化
        return v2 - v1
    
    def _extract_temporal_features(self, data: List[Dict]) -> List[float]:
        """
        提取时间特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            时间特征列表
        """
        features = []
        
        if len(data) < 2:
            return [0, 0, 0, 0]
        
        # 总时长
        total_duration = data[-1]['timestamp_ms'] - data[0]['timestamp_ms']
        features.append(total_duration)
        
        # 帧间隔统计
        intervals = []
        for i in range(1, len(data)):
            interval = data[i]['timestamp_ms'] - data[i-1]['timestamp_ms']
            intervals.append(interval)
        
        if intervals:
            features.extend([
                np.mean(intervals),  # 平均帧间隔
                np.std(intervals),   # 帧间隔标准差
                len(data)            # 总帧数
            ])
        else:
            features.extend([0, 0, len(data)])
        
        return features
    
    def _extract_temporal_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取时间特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (时间特征列表, 特征标签列表)
        """
        features = []
        labels = ['total_duration', 'avg_frame_interval', 'frame_interval_std', 'total_frames']
        
        if len(data) < 2:
            return [0, 0, 0, 0], labels
        
        # 总时长
        total_duration = data[-1]['timestamp_ms'] - data[0]['timestamp_ms']
        features.append(total_duration)
        
        # 帧间隔统计
        intervals = []
        for i in range(1, len(data)):
            interval = data[i]['timestamp_ms'] - data[i-1]['timestamp_ms']
            intervals.append(interval)
        
        if intervals:
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                len(data)
            ])
        else:
            features.extend([0, 0, len(data)])
        
        return features, labels
    
    def _extract_statistical_features(self, data: List[Dict]) -> List[float]:
        """
        提取统计特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            统计特征列表
        """
        features = []
        
        if not data:
            return [0] * 12  # 预定义的统计特征数量
        
        # 提取右手腕轨迹进行统计分析
        wrist_x = []
        wrist_y = []
        
        for frame in data:
            landmarks = frame['landmarks']
            wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
            if wrist:
                wrist_x.append(wrist['x'])
                wrist_y.append(wrist['y'])
        
        if wrist_x and wrist_y:
            # X方向统计
            features.extend([
                np.mean(wrist_x),
                np.std(wrist_x),
                stats.skew(wrist_x),      # 偏度
                stats.kurtosis(wrist_x)   # 峰度
            ])
            
            # Y方向统计
            features.extend([
                np.mean(wrist_y),
                np.std(wrist_y),
                stats.skew(wrist_y),
                stats.kurtosis(wrist_y)
            ])
            
            # 轨迹复杂度
            trajectory_length = 0
            for i in range(1, len(wrist_x)):
                dx = wrist_x[i] - wrist_x[i-1]
                dy = wrist_y[i] - wrist_y[i-1]
                trajectory_length += np.sqrt(dx**2 + dy**2)
            
            # 直线距离
            direct_distance = np.sqrt((wrist_x[-1] - wrist_x[0])**2 + (wrist_y[-1] - wrist_y[0])**2)
            
            # 轨迹效率
            efficiency = direct_distance / trajectory_length if trajectory_length > 0 else 0
            
            features.extend([
                trajectory_length,
                direct_distance,
                efficiency,
                len(wrist_x)  # 有效点数
            ])
        else:
            features.extend([0] * 12)
        
        return features
    
    def _extract_statistical_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取统计特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (统计特征列表, 特征标签列表)
        """
        features = []
        labels = [
            'wrist_x_mean', 'wrist_x_std', 'wrist_x_skew', 'wrist_x_kurtosis',
            'wrist_y_mean', 'wrist_y_std', 'wrist_y_skew', 'wrist_y_kurtosis',
            'trajectory_length', 'direct_distance', 'trajectory_efficiency', 'valid_points'
        ]
        
        if not data:
            return [0] * 12, labels
        
        # 提取右手腕轨迹进行统计分析
        wrist_x = []
        wrist_y = []
        
        for frame in data:
            landmarks = frame['landmarks']
            wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
            if wrist:
                wrist_x.append(wrist['x'])
                wrist_y.append(wrist['y'])
        
        if wrist_x and wrist_y:
            # X方向统计
            features.extend([
                np.mean(wrist_x),
                np.std(wrist_x),
                stats.skew(wrist_x),
                stats.kurtosis(wrist_x)
            ])
            
            # Y方向统计
            features.extend([
                np.mean(wrist_y),
                np.std(wrist_y),
                stats.skew(wrist_y),
                stats.kurtosis(wrist_y)
            ])
            
            # 轨迹复杂度
            trajectory_length = 0
            for i in range(1, len(wrist_x)):
                dx = wrist_x[i] - wrist_x[i-1]
                dy = wrist_y[i] - wrist_y[i-1]
                trajectory_length += np.sqrt(dx**2 + dy**2)
            
            # 直线距离
            direct_distance = np.sqrt((wrist_x[-1] - wrist_x[0])**2 + (wrist_y[-1] - wrist_y[0])**2)
            
            # 轨迹效率
            efficiency = direct_distance / trajectory_length if trajectory_length > 0 else 0
            
            features.extend([
                trajectory_length,
                direct_distance,
                efficiency,
                len(wrist_x)
            ])
        else:
            features.extend([0] * 12)
        
        return features, labels
    
    def _extract_frequency_features(self, data: List[Dict]) -> List[float]:
        """
        提取频域特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            频域特征列表
        """
        features = []
        
        if len(data) < 8:  # FFT需要足够的数据点
            return [0] * 6
        
        # 提取右手腕的X和Y坐标序列
        wrist_x = []
        wrist_y = []
        
        for frame in data:
            landmarks = frame['landmarks']
            wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
            if wrist:
                wrist_x.append(wrist['x'])
                wrist_y.append(wrist['y'])
        
        if len(wrist_x) >= 8:
            # 对X和Y坐标分别进行FFT分析
            for signal_data in [wrist_x, wrist_y]:
                # 去除直流分量
                signal_data = np.array(signal_data) - np.mean(signal_data)
                
                # FFT
                fft_result = np.fft.fft(signal_data)
                power_spectrum = np.abs(fft_result[:len(fft_result)//2])
                
                if len(power_spectrum) > 0:
                    # 主频率分量
                    dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # 排除直流分量
                    
                    # 频域特征
                    features.extend([
                        dominant_freq_idx,                    # 主频率索引
                        np.max(power_spectrum),               # 最大功率
                        np.sum(power_spectrum[:3]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0  # 低频能量比
                    ])
                else:
                    features.extend([0, 0, 0])
        else:
            features.extend([0] * 6)
        
        return features
    
    def _extract_frequency_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取频域特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (频域特征列表, 特征标签列表)
        """
        features = []
        labels = [
            'wrist_x_dominant_freq', 'wrist_x_max_power', 'wrist_x_low_freq_ratio',
            'wrist_y_dominant_freq', 'wrist_y_max_power', 'wrist_y_low_freq_ratio'
        ]
        
        if len(data) < 8:
            return [0] * 6, labels
        
        # 提取右手腕的X和Y坐标序列
        wrist_x = []
        wrist_y = []
        
        for frame in data:
            landmarks = frame['landmarks']
            wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
            if wrist:
                wrist_x.append(wrist['x'])
                wrist_y.append(wrist['y'])
        
        if len(wrist_x) >= 8:
            # 对X和Y坐标分别进行FFT分析
            for signal_data in [wrist_x, wrist_y]:
                # 去除直流分量
                signal_data = np.array(signal_data) - np.mean(signal_data)
                
                # FFT
                fft_result = np.fft.fft(signal_data)
                power_spectrum = np.abs(fft_result[:len(fft_result)//2])
                
                if len(power_spectrum) > 0:
                    # 主频率分量
                    dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
                    
                    # 频域特征
                    features.extend([
                        dominant_freq_idx,
                        np.max(power_spectrum),
                        np.sum(power_spectrum[:3]) / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else 0
                    ])
                else:
                    features.extend([0, 0, 0])
        else:
            features.extend([0] * 6)
        
        return features, labels
    
    def _extract_coordination_features(self, data: List[Dict]) -> List[float]:
        """
        提取协调性特征
        
        Args:
            data: 预处理后的数据
            
        Returns:
            协调性特征列表
        """
        features = []
        
        if len(data) < 3:
            return [0] * 8
        
        # 收集关键角度序列
        right_elbow_angles = []
        right_shoulder_angles = []
        left_elbow_angles = []
        left_shoulder_angles = []
        
        for frame in data:
            landmarks = frame['landmarks']
            
            # 计算各关节角度
            angles = {}
            for angle_name, (p1_name, p2_name, p3_name) in self.key_angles.items():
                p1 = self._get_landmark(landmarks, p1_name)
                p2 = self._get_landmark(landmarks, p2_name)
                p3 = self._get_landmark(landmarks, p3_name)
                
                if p1 and p2 and p3:
                    angle = self._calculate_angle(p1, p2, p3)
                    if angle is not None:
                        angles[angle_name] = angle
            
            # 收集角度数据
            if 'right_elbow' in angles:
                right_elbow_angles.append(angles['right_elbow'])
            if 'right_shoulder' in angles:
                right_shoulder_angles.append(angles['right_shoulder'])
            if 'left_elbow' in angles:
                left_elbow_angles.append(angles['left_elbow'])
            if 'left_shoulder' in angles:
                left_shoulder_angles.append(angles['left_shoulder'])
        
        # 计算协调性特征
        # 1. 右侧肢体协调性（肘部-肩部）
        if len(right_elbow_angles) == len(right_shoulder_angles) and len(right_elbow_angles) >= 3:
            correlation = np.corrcoef(right_elbow_angles, right_shoulder_angles)[0, 1]
            features.append(abs(correlation) if not np.isnan(correlation) else 0)
            
            # 角度变化的同步性
            elbow_changes = np.diff(right_elbow_angles)
            shoulder_changes = np.diff(right_shoulder_angles)
            sync_correlation = np.corrcoef(elbow_changes, shoulder_changes)[0, 1]
            features.append(abs(sync_correlation) if not np.isnan(sync_correlation) else 0)
        else:
            features.extend([0, 0])
        
        # 2. 左侧肢体协调性
        if len(left_elbow_angles) == len(left_shoulder_angles) and len(left_elbow_angles) >= 3:
            correlation = np.corrcoef(left_elbow_angles, left_shoulder_angles)[0, 1]
            features.append(abs(correlation) if not np.isnan(correlation) else 0)
        else:
            features.append(0)
        
        # 3. 左右对称性
        if (len(right_elbow_angles) == len(left_elbow_angles) and 
            len(right_shoulder_angles) == len(left_shoulder_angles) and
            len(right_elbow_angles) >= 3):
            
            # 肘部对称性
            elbow_symmetry = np.corrcoef(right_elbow_angles, left_elbow_angles)[0, 1]
            features.append(abs(elbow_symmetry) if not np.isnan(elbow_symmetry) else 0)
            
            # 肩部对称性
            shoulder_symmetry = np.corrcoef(right_shoulder_angles, left_shoulder_angles)[0, 1]
            features.append(abs(shoulder_symmetry) if not np.isnan(shoulder_symmetry) else 0)
        else:
            features.extend([0, 0])
        
        # 4. 动作平滑性
        if len(right_elbow_angles) >= 3:
            # 角度变化的平滑性（二阶差分的标准差）
            elbow_smoothness = 1.0 / (1.0 + np.std(np.diff(right_elbow_angles, n=2)))
            features.append(elbow_smoothness)
        else:
            features.append(0)
        
        # 5. 整体协调性评分
        if len(features) >= 5:
            overall_coordination = np.mean(features[-5:])
            features.append(overall_coordination)
        else:
            features.append(0)
        
        # 6. 节奏一致性
        if len(right_elbow_angles) >= 5:
            # 使用峰值检测评估节奏
            peaks, _ = signal.find_peaks(right_elbow_angles)
            if len(peaks) >= 2:
                peak_intervals = np.diff(peaks)
                rhythm_consistency = 1.0 / (1.0 + np.std(peak_intervals))
                features.append(rhythm_consistency)
            else:
                features.append(0)
        else:
            features.append(0)
        
        return features[:8]  # 确保返回8个特征
    
    def _extract_coordination_features_with_labels(self, data: List[Dict]) -> Tuple[List[float], List[str]]:
        """
        提取协调性特征并返回标签
        
        Args:
            data: 预处理后的数据
            
        Returns:
            (协调性特征列表, 特征标签列表)
        """
        labels = [
            'right_elbow_shoulder_coordination',
            'right_elbow_shoulder_sync',
            'left_elbow_shoulder_coordination',
            'elbow_symmetry',
            'shoulder_symmetry',
            'movement_smoothness',
            'overall_coordination',
            'rhythm_consistency'
        ]
        
        features = self._extract_coordination_features(data)
        
        # 确保特征和标签数量一致
        while len(features) < len(labels):
            features.append(0)
        
        return features[:len(labels)], labels
    
    def get_feature_dimension(self) -> int:
        """
        获取特征向量的维度
        
        Returns:
            特征向量维度
        """
        # 计算各类特征的维度
        dimensions = {
            'angle_features': len(self.key_angles) * 5,      # 每个角度5个统计特征
            'distance_features': len(self.key_distances) * 4, # 每个距离4个统计特征
            'velocity_features': len(self.velocity_points) * 4, # 每个点4个速度特征
            'acceleration_features': len(self.velocity_points) * 3, # 每个点3个加速度特征
            'temporal_features': 4,                          # 4个时间特征
            'statistical_features': 12,                      # 12个统计特征
            'frequency_features': 6,                         # 6个频域特征
            'coordination_features': 8                       # 8个协调性特征
        }
        
        total_dim = 0
        for feature_type, enabled in self.feature_config.items():
            if enabled:
                total_dim += dimensions.get(feature_type, 0)
        
        return total_dim
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有特征的名称
        
        Returns:
            特征名称列表
        """
        all_labels = []
        
        # 创建空数据进行测试
        dummy_data = [{
            'timestamp_ms': i * 33,
            'landmarks': {
                name: {'x': 0.5, 'y': 0.5, 'z': 0, 'visibility': 1.0}
                for name in self.landmark_indices.keys()
            }
        } for i in range(10)]
        
        for feature_type in self.feature_config.keys():
            if self.feature_config[feature_type]:
                _, labels = self._extract_feature_type_with_labels(dummy_data, feature_type)
                all_labels.extend(labels)
        
        return all_labels

# 导出主要类
__all__ = ['MLFeatureExtractor']