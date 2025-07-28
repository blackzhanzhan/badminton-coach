#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版姿态分析器
在原有PoseAnalyzer基础上增加机器学习特征提取和高级分析功能
"""

import json
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from scipy import signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import cv2

# 导入原有的PoseAnalyzer
from .pose_analyzer import PoseAnalyzer

class EnhancedPoseAnalyzer(PoseAnalyzer):
    """
    增强版姿态分析器
    继承原有PoseAnalyzer，增加高级特征提取和分析功能
    """
    
    def __init__(self):
        """
        初始化增强版姿态分析器
        """
        super().__init__()
        
        # 增强版配置
        self.smoothing_window = 5  # 平滑窗口大小
        self.velocity_threshold = 0.1  # 速度阈值
        self.acceleration_threshold = 0.05  # 加速度阈值
        
        # 关键阶段检测参数
        self.stage_detection_params = {
            'min_stage_duration': 200,  # 最小阶段持续时间(ms)
            'velocity_change_threshold': 0.15,  # 速度变化阈值
            'angle_change_threshold': 10  # 角度变化阈值(度)
        }
        
        print("增强版姿态分析器初始化完成")
    
    def analyze_pose_sequence_enhanced(self, landmarks_timeline: List[Dict]) -> Dict[str, Any]:
        """
        增强版姿态序列分析
        
        Args:
            landmarks_timeline: 姿态时间线数据
            
        Returns:
            增强版分析结果
        """
        if not landmarks_timeline:
            return self._get_empty_analysis_result()
        
        # 1. 基础分析（调用父类方法）
        basic_analysis = self.analyze_pose_sequence(landmarks_timeline)
        
        # 2. 数据预处理
        processed_data = self._preprocess_landmarks_data(landmarks_timeline)
        
        # 3. 高级特征提取
        advanced_features = self._extract_advanced_features(processed_data)
        
        # 4. 动作阶段自动检测
        detected_stages = self._detect_action_stages(processed_data)
        
        # 5. 运动学分析
        kinematics_analysis = self._analyze_kinematics(processed_data)
        
        # 6. 动作质量评估
        quality_assessment = self._assess_action_quality(advanced_features, kinematics_analysis)
        
        # 7. 合并所有分析结果
        enhanced_result = {
            **basic_analysis,
            'enhanced_features': advanced_features,
            'detected_stages': detected_stages,
            'kinematics_analysis': kinematics_analysis,
            'quality_assessment': quality_assessment,
            'processing_info': {
                'smoothed_frames': len(processed_data),
                'analysis_version': 'enhanced_v1.0'
            }
        }
        
        return enhanced_result
    
    def _get_empty_analysis_result(self) -> Dict[str, Any]:
        """
        获取空的分析结果
        
        Returns:
            空分析结果字典
        """
        return {
            'total_frames': 0,
            'valid_frames': 0,
            'analysis_summary': '未检测到有效的姿势数据',
            'key_metrics': {},
            'suggestions': ['无法分析：未检测到姿势数据'],
            'enhanced_features': {},
            'detected_stages': [],
            'kinematics_analysis': {},
            'quality_assessment': {'overall_score': 0, 'quality_level': '无法评估'}
        }
    
    def _preprocess_landmarks_data(self, landmarks_timeline: List[Dict]) -> List[Dict]:
        """
        预处理关键点数据
        
        Args:
            landmarks_timeline: 原始关键点时间线
            
        Returns:
            预处理后的数据
        """
        processed_data = []
        
        # 提取有效帧
        valid_frames = []
        for frame_data in landmarks_timeline:
            landmarks = frame_data.get('landmarks', {})
            if landmarks and self._is_valid_frame(landmarks):
                valid_frames.append(frame_data)
        
        if len(valid_frames) < 3:
            return valid_frames
        
        # 平滑处理
        smoothed_frames = self._smooth_landmarks_sequence(valid_frames)
        
        # 添加计算字段
        for i, frame in enumerate(smoothed_frames):
            enhanced_frame = frame.copy()
            
            # 计算关键角度
            landmarks = frame.get('landmarks', {})
            enhanced_frame['computed_angles'] = self._compute_all_angles(landmarks)
            
            # 计算速度和加速度（需要前后帧）
            if i > 0:
                enhanced_frame['velocities'] = self._compute_velocities(
                    smoothed_frames[i-1]['landmarks'], landmarks
                )
            
            if i > 1:
                enhanced_frame['accelerations'] = self._compute_accelerations(
                    smoothed_frames[i-2]['landmarks'], 
                    smoothed_frames[i-1]['landmarks'], 
                    landmarks
                )
            
            processed_data.append(enhanced_frame)
        
        return processed_data
    
    def _is_valid_frame(self, landmarks: Dict) -> bool:
        """
        检查帧是否有效
        
        Args:
            landmarks: 关键点数据
            
        Returns:
            是否为有效帧
        """
        # 检查关键关节点是否存在
        key_points = ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'LEFT_SHOULDER']
        
        for point_name in key_points:
            point = self._get_landmark(landmarks, point_name)
            if not point or point.get('visibility', 0) < 0.5:
                return False
        
        return True
    
    def _smooth_landmarks_sequence(self, frames: List[Dict]) -> List[Dict]:
        """
        平滑关键点序列
        
        Args:
            frames: 帧序列
            
        Returns:
            平滑后的帧序列
        """
        if len(frames) < self.smoothing_window:
            return frames
        
        smoothed_frames = []
        
        for i in range(len(frames)):
            # 确定平滑窗口
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(frames), i + self.smoothing_window // 2 + 1)
            window_frames = frames[start_idx:end_idx]
            
            # 平滑当前帧
            smoothed_frame = self._smooth_single_frame(window_frames, frames[i])
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def _smooth_single_frame(self, window_frames: List[Dict], center_frame: Dict) -> Dict:
        """
        平滑单个帧
        
        Args:
            window_frames: 窗口内的帧
            center_frame: 中心帧
            
        Returns:
            平滑后的帧
        """
        smoothed_frame = center_frame.copy()
        landmarks = center_frame.get('landmarks', {})
        
        if not landmarks:
            return smoothed_frame
        
        smoothed_landmarks = {}
        
        # 对每个关键点进行平滑
        for landmark_name in landmarks.keys():
            positions = []
            for frame in window_frames:
                frame_landmarks = frame.get('landmarks', {})
                if landmark_name in frame_landmarks:
                    point = frame_landmarks[landmark_name]
                    positions.append([point.get('x', 0), point.get('y', 0), point.get('z', 0)])
            
            if positions:
                # 计算平均位置
                avg_position = np.mean(positions, axis=0)
                smoothed_landmarks[landmark_name] = {
                    'x': float(avg_position[0]),
                    'y': float(avg_position[1]),
                    'z': float(avg_position[2]),
                    'visibility': landmarks[landmark_name].get('visibility', 1.0)
                }
            else:
                smoothed_landmarks[landmark_name] = landmarks[landmark_name]
        
        smoothed_frame['landmarks'] = smoothed_landmarks
        return smoothed_frame
    
    def _compute_all_angles(self, landmarks: Dict) -> Dict[str, float]:
        """
        计算所有关键角度
        
        Args:
            landmarks: 关键点数据
            
        Returns:
            角度字典
        """
        angles = {}
        
        # 定义角度计算配置
        angle_configs = {
            'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            'left_elbow': ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            'right_shoulder': ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            'left_shoulder': ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
            'right_hip': ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
            'left_hip': ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
            'right_knee': ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            'left_knee': ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            'torso_lean': ('LEFT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'),  # 身体倾斜
            'arm_extension': ('LEFT_SHOULDER', 'RIGHT_SHOULDER', 'RIGHT_WRIST')  # 手臂伸展
        }
        
        for angle_name, (p1_name, p2_name, p3_name) in angle_configs.items():
            p1 = self._get_landmark(landmarks, p1_name)
            p2 = self._get_landmark(landmarks, p2_name)
            p3 = self._get_landmark(landmarks, p3_name)
            
            if p1 and p2 and p3:
                angle = self._calculate_angle(p1, p2, p3)
                angles[angle_name] = angle if angle is not None else 0.0
            else:
                angles[angle_name] = 0.0
        
        return angles
    
    def _compute_velocities(self, prev_landmarks: Dict, curr_landmarks: Dict) -> Dict[str, float]:
        """
        计算关键点速度
        
        Args:
            prev_landmarks: 前一帧关键点
            curr_landmarks: 当前帧关键点
            
        Returns:
            速度字典
        """
        velocities = {}
        
        key_points = ['RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_ELBOW', 'LEFT_ELBOW']
        
        for point_name in key_points:
            prev_point = self._get_landmark(prev_landmarks, point_name)
            curr_point = self._get_landmark(curr_landmarks, point_name)
            
            if prev_point and curr_point:
                # 计算2D速度
                dx = curr_point['x'] - prev_point['x']
                dy = curr_point['y'] - prev_point['y']
                velocity = np.sqrt(dx**2 + dy**2)
                velocities[f'{point_name.lower()}_velocity'] = velocity
            else:
                velocities[f'{point_name.lower()}_velocity'] = 0.0
        
        return velocities
    
    def _compute_accelerations(self, prev2_landmarks: Dict, prev_landmarks: Dict, 
                             curr_landmarks: Dict) -> Dict[str, float]:
        """
        计算关键点加速度
        
        Args:
            prev2_landmarks: 前两帧关键点
            prev_landmarks: 前一帧关键点
            curr_landmarks: 当前帧关键点
            
        Returns:
            加速度字典
        """
        accelerations = {}
        
        # 计算前一帧和当前帧的速度
        prev_velocities = self._compute_velocities(prev2_landmarks, prev_landmarks)
        curr_velocities = self._compute_velocities(prev_landmarks, curr_landmarks)
        
        for key in prev_velocities.keys():
            if key in curr_velocities:
                acceleration = curr_velocities[key] - prev_velocities[key]
                accelerations[key.replace('velocity', 'acceleration')] = acceleration
            else:
                accelerations[key.replace('velocity', 'acceleration')] = 0.0
        
        return accelerations
    
    def _extract_advanced_features(self, processed_data: List[Dict]) -> Dict[str, Any]:
        """
        提取高级特征
        
        Args:
            processed_data: 预处理后的数据
            
        Returns:
            高级特征字典
        """
        if not processed_data:
            return {}
        
        features = {
            'temporal_features': self._extract_temporal_features(processed_data),
            'spatial_features': self._extract_spatial_features(processed_data),
            'kinematic_features': self._extract_kinematic_features(processed_data),
            'coordination_features': self._extract_coordination_features(processed_data)
        }
        
        return features
    
    def _extract_temporal_features(self, data: List[Dict]) -> Dict[str, float]:
        """
        提取时间特征
        
        Args:
            data: 处理后的数据
            
        Returns:
            时间特征字典
        """
        features = {}
        
        # 总时长
        if len(data) >= 2:
            start_time = data[0].get('timestamp_ms', 0)
            end_time = data[-1].get('timestamp_ms', 0)
            features['total_duration'] = end_time - start_time
        else:
            features['total_duration'] = 0
        
        # 帧率稳定性
        if len(data) >= 3:
            intervals = []
            for i in range(1, len(data)):
                interval = data[i].get('timestamp_ms', 0) - data[i-1].get('timestamp_ms', 0)
                intervals.append(interval)
            
            features['frame_rate_stability'] = np.std(intervals) if intervals else 0
            features['average_frame_interval'] = np.mean(intervals) if intervals else 0
        else:
            features['frame_rate_stability'] = 0
            features['average_frame_interval'] = 0
        
        return features
    
    def _extract_spatial_features(self, data: List[Dict]) -> Dict[str, float]:
        """
        提取空间特征
        
        Args:
            data: 处理后的数据
            
        Returns:
            空间特征字典
        """
        features = {}
        
        if not data:
            return features
        
        # 提取右手腕轨迹
        wrist_positions = []
        for frame in data:
            landmarks = frame.get('landmarks', {})
            wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
            if wrist:
                wrist_positions.append([wrist['x'], wrist['y']])
        
        if len(wrist_positions) >= 2:
            wrist_positions = np.array(wrist_positions)
            
            # 轨迹长度
            trajectory_length = np.sum(np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1))
            features['wrist_trajectory_length'] = trajectory_length
            
            # 轨迹范围
            x_range = np.max(wrist_positions[:, 0]) - np.min(wrist_positions[:, 0])
            y_range = np.max(wrist_positions[:, 1]) - np.min(wrist_positions[:, 1])
            features['wrist_x_range'] = x_range
            features['wrist_y_range'] = y_range
            
            # 轨迹复杂度（周长与面积比）
            if x_range > 0 and y_range > 0:
                area = x_range * y_range
                perimeter = 2 * (x_range + y_range)
                features['trajectory_complexity'] = perimeter / area if area > 0 else 0
            else:
                features['trajectory_complexity'] = 0
            
            # 起始和结束位置
            features['start_x'] = wrist_positions[0, 0]
            features['start_y'] = wrist_positions[0, 1]
            features['end_x'] = wrist_positions[-1, 0]
            features['end_y'] = wrist_positions[-1, 1]
            
            # 位移向量
            displacement = wrist_positions[-1] - wrist_positions[0]
            features['net_displacement'] = np.linalg.norm(displacement)
            features['displacement_efficiency'] = features['net_displacement'] / trajectory_length if trajectory_length > 0 else 0
        
        return features
    
    def _extract_kinematic_features(self, data: List[Dict]) -> Dict[str, float]:
        """
        提取运动学特征
        
        Args:
            data: 处理后的数据
            
        Returns:
            运动学特征字典
        """
        features = {}
        
        # 收集所有速度和加速度数据
        velocities = []
        accelerations = []
        
        for frame in data:
            frame_velocities = frame.get('velocities', {})
            frame_accelerations = frame.get('accelerations', {})
            
            # 右手腕速度
            if 'right_wrist_velocity' in frame_velocities:
                velocities.append(frame_velocities['right_wrist_velocity'])
            
            # 右手腕加速度
            if 'right_wrist_acceleration' in frame_accelerations:
                accelerations.append(frame_accelerations['right_wrist_acceleration'])
        
        if velocities:
            features['max_velocity'] = np.max(velocities)
            features['avg_velocity'] = np.mean(velocities)
            features['velocity_std'] = np.std(velocities)
            
            # 速度变化次数（峰值检测）
            peaks, _ = signal.find_peaks(velocities, height=np.mean(velocities))
            features['velocity_peaks'] = len(peaks)
        
        if accelerations:
            features['max_acceleration'] = np.max(np.abs(accelerations))
            features['avg_acceleration'] = np.mean(np.abs(accelerations))
            features['acceleration_std'] = np.std(accelerations)
        
        return features
    
    def _extract_coordination_features(self, data: List[Dict]) -> Dict[str, float]:
        """
        提取协调性特征
        
        Args:
            data: 处理后的数据
            
        Returns:
            协调性特征字典
        """
        features = {}
        
        if len(data) < 3:
            return features
        
        # 收集角度数据
        right_elbow_angles = []
        right_shoulder_angles = []
        
        for frame in data:
            angles = frame.get('computed_angles', {})
            if 'right_elbow' in angles:
                right_elbow_angles.append(angles['right_elbow'])
            if 'right_shoulder' in angles:
                right_shoulder_angles.append(angles['right_shoulder'])
        
        # 角度变化的平滑性
        if len(right_elbow_angles) >= 3:
            elbow_changes = np.diff(right_elbow_angles)
            features['elbow_angle_smoothness'] = 1.0 / (1.0 + np.std(elbow_changes))
        
        if len(right_shoulder_angles) >= 3:
            shoulder_changes = np.diff(right_shoulder_angles)
            features['shoulder_angle_smoothness'] = 1.0 / (1.0 + np.std(shoulder_changes))
        
        # 肘部和肩部协调性（相关性）
        if len(right_elbow_angles) == len(right_shoulder_angles) and len(right_elbow_angles) >= 3:
            correlation = np.corrcoef(right_elbow_angles, right_shoulder_angles)[0, 1]
            features['elbow_shoulder_coordination'] = abs(correlation) if not np.isnan(correlation) else 0
        
        return features
    
    def _detect_action_stages(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """
        自动检测动作阶段
        
        Args:
            data: 处理后的数据
            
        Returns:
            检测到的阶段列表
        """
        if len(data) < 5:
            return []
        
        stages = []
        
        # 提取右手腕速度序列
        velocities = []
        timestamps = []
        
        for frame in data:
            frame_velocities = frame.get('velocities', {})
            if 'right_wrist_velocity' in frame_velocities:
                velocities.append(frame_velocities['right_wrist_velocity'])
                timestamps.append(frame.get('timestamp_ms', 0))
        
        if len(velocities) < 5:
            return stages
        
        velocities = np.array(velocities)
        timestamps = np.array(timestamps)
        
        # 使用速度变化检测阶段边界
        # 1. 平滑速度曲线
        smoothed_velocities = signal.savgol_filter(velocities, 
                                                 window_length=min(5, len(velocities)//2*2+1), 
                                                 polyorder=2)
        
        # 2. 找到速度峰值和谷值
        peaks, _ = signal.find_peaks(smoothed_velocities, 
                                   height=np.mean(smoothed_velocities),
                                   distance=len(velocities)//10)
        valleys, _ = signal.find_peaks(-smoothed_velocities,
                                     height=-np.mean(smoothed_velocities),
                                     distance=len(velocities)//10)
        
        # 3. 合并并排序关键点
        key_points = sorted(list(peaks) + list(valleys))
        
        # 4. 生成阶段
        stage_names = ['准备', '后摆', '前挥', '击球', '收势']
        
        if len(key_points) >= 2:
            # 根据关键点划分阶段
            stage_boundaries = [0] + key_points + [len(data)-1]
            
            for i in range(len(stage_boundaries)-1):
                start_idx = stage_boundaries[i]
                end_idx = stage_boundaries[i+1]
                
                if end_idx > start_idx:
                    stage_name = stage_names[i] if i < len(stage_names) else f'阶段{i+1}'
                    
                    stage = {
                        'stage_name': stage_name,
                        'start_frame': start_idx,
                        'end_frame': end_idx,
                        'start_time': timestamps[start_idx] if start_idx < len(timestamps) else 0,
                        'end_time': timestamps[end_idx-1] if end_idx-1 < len(timestamps) else 0,
                        'duration': timestamps[end_idx-1] - timestamps[start_idx] if end_idx-1 < len(timestamps) and start_idx < len(timestamps) else 0,
                        'avg_velocity': np.mean(velocities[start_idx:end_idx]),
                        'max_velocity': np.max(velocities[start_idx:end_idx]),
                        'confidence': 0.8  # 简化的置信度
                    }
                    
                    stages.append(stage)
        
        return stages[:5]  # 最多返回5个阶段
    
    def _analyze_kinematics(self, data: List[Dict]) -> Dict[str, Any]:
        """
        运动学分析
        
        Args:
            data: 处理后的数据
            
        Returns:
            运动学分析结果
        """
        analysis = {
            'velocity_profile': {},
            'acceleration_profile': {},
            'angle_analysis': {},
            'movement_efficiency': {}
        }
        
        if not data:
            return analysis
        
        # 速度分析
        wrist_velocities = []
        elbow_velocities = []
        
        for frame in data:
            velocities = frame.get('velocities', {})
            if 'right_wrist_velocity' in velocities:
                wrist_velocities.append(velocities['right_wrist_velocity'])
            if 'right_elbow_velocity' in velocities:
                elbow_velocities.append(velocities['right_elbow_velocity'])
        
        if wrist_velocities:
            analysis['velocity_profile']['wrist'] = {
                'max': np.max(wrist_velocities),
                'mean': np.mean(wrist_velocities),
                'std': np.std(wrist_velocities),
                'peak_count': len(signal.find_peaks(wrist_velocities)[0])
            }
        
        if elbow_velocities:
            analysis['velocity_profile']['elbow'] = {
                'max': np.max(elbow_velocities),
                'mean': np.mean(elbow_velocities),
                'std': np.std(elbow_velocities)
            }
        
        # 角度分析
        elbow_angles = []
        shoulder_angles = []
        
        for frame in data:
            angles = frame.get('computed_angles', {})
            if 'right_elbow' in angles:
                elbow_angles.append(angles['right_elbow'])
            if 'right_shoulder' in angles:
                shoulder_angles.append(angles['right_shoulder'])
        
        if elbow_angles:
            analysis['angle_analysis']['elbow'] = {
                'range': np.max(elbow_angles) - np.min(elbow_angles),
                'mean': np.mean(elbow_angles),
                'std': np.std(elbow_angles),
                'min': np.min(elbow_angles),
                'max': np.max(elbow_angles)
            }
        
        if shoulder_angles:
            analysis['angle_analysis']['shoulder'] = {
                'range': np.max(shoulder_angles) - np.min(shoulder_angles),
                'mean': np.mean(shoulder_angles),
                'std': np.std(shoulder_angles)
            }
        
        # 运动效率分析
        if len(data) >= 2:
            # 计算轨迹效率
            wrist_positions = []
            for frame in data:
                landmarks = frame.get('landmarks', {})
                wrist = self._get_landmark(landmarks, 'RIGHT_WRIST')
                if wrist:
                    wrist_positions.append([wrist['x'], wrist['y']])
            
            if len(wrist_positions) >= 2:
                wrist_positions = np.array(wrist_positions)
                trajectory_length = np.sum(np.linalg.norm(np.diff(wrist_positions, axis=0), axis=1))
                direct_distance = np.linalg.norm(wrist_positions[-1] - wrist_positions[0])
                
                efficiency = direct_distance / trajectory_length if trajectory_length > 0 else 0
                analysis['movement_efficiency']['trajectory_efficiency'] = efficiency
                analysis['movement_efficiency']['trajectory_length'] = trajectory_length
                analysis['movement_efficiency']['direct_distance'] = direct_distance
        
        return analysis
    
    def _assess_action_quality(self, features: Dict[str, Any], 
                             kinematics: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估动作质量
        
        Args:
            features: 高级特征
            kinematics: 运动学分析结果
            
        Returns:
            质量评估结果
        """
        assessment = {
            'overall_score': 0,
            'quality_level': '无法评估',
            'component_scores': {},
            'strengths': [],
            'weaknesses': []
        }
        
        scores = []
        
        # 1. 轨迹质量评分
        spatial_features = features.get('spatial_features', {})
        if 'displacement_efficiency' in spatial_features:
            efficiency = spatial_features['displacement_efficiency']
            trajectory_score = min(efficiency * 100, 100)
            scores.append(trajectory_score)
            assessment['component_scores']['trajectory'] = trajectory_score
            
            if efficiency > 0.7:
                assessment['strengths'].append('轨迹效率高')
            elif efficiency < 0.4:
                assessment['weaknesses'].append('轨迹效率低，存在多余动作')
        
        # 2. 速度控制评分
        velocity_profile = kinematics.get('velocity_profile', {})
        if 'wrist' in velocity_profile:
            wrist_vel = velocity_profile['wrist']
            # 基于速度变化的平滑性评分
            if wrist_vel.get('std', 0) > 0:
                smoothness = 1.0 / (1.0 + wrist_vel['std'])
                velocity_score = smoothness * 100
                scores.append(velocity_score)
                assessment['component_scores']['velocity_control'] = velocity_score
                
                if smoothness > 0.8:
                    assessment['strengths'].append('速度控制平滑')
                elif smoothness < 0.5:
                    assessment['weaknesses'].append('速度变化不够平滑')
        
        # 3. 协调性评分
        coordination_features = features.get('coordination_features', {})
        if 'elbow_shoulder_coordination' in coordination_features:
            coordination = coordination_features['elbow_shoulder_coordination']
            coordination_score = coordination * 100
            scores.append(coordination_score)
            assessment['component_scores']['coordination'] = coordination_score
            
            if coordination > 0.7:
                assessment['strengths'].append('肘肩协调性好')
            elif coordination < 0.4:
                assessment['weaknesses'].append('肘肩协调性需要改进')
        
        # 4. 角度控制评分
        angle_analysis = kinematics.get('angle_analysis', {})
        if 'elbow' in angle_analysis:
            elbow_data = angle_analysis['elbow']
            # 基于角度范围的合理性评分
            angle_range = elbow_data.get('range', 0)
            if 30 <= angle_range <= 120:  # 合理的肘部角度变化范围
                angle_score = 90
            elif angle_range < 30:
                angle_score = 60  # 角度变化太小
            else:
                angle_score = 70  # 角度变化太大
            
            scores.append(angle_score)
            assessment['component_scores']['angle_control'] = angle_score
            
            if angle_score >= 85:
                assessment['strengths'].append('角度控制良好')
            elif angle_score < 70:
                assessment['weaknesses'].append('角度控制需要改进')
        
        # 计算总分
        if scores:
            assessment['overall_score'] = np.mean(scores)
        else:
            assessment['overall_score'] = 50  # 默认分数
        
        # 确定质量等级
        score = assessment['overall_score']
        if score >= 85:
            assessment['quality_level'] = '优秀'
        elif score >= 75:
            assessment['quality_level'] = '良好'
        elif score >= 60:
            assessment['quality_level'] = '一般'
        else:
            assessment['quality_level'] = '需改进'
        
        return assessment
    
    def compare_with_template_enhanced(self, user_data: List[Dict], 
                                     template_data: List[Dict]) -> Dict[str, Any]:
        """
        增强版模板对比
        
        Args:
            user_data: 用户数据
            template_data: 模板数据
            
        Returns:
            增强版对比结果
        """
        # 1. 基础对比（调用父类方法）
        basic_comparison = self.analyze_json_difference(
            template_data, user_data  # 注意参数顺序
        )
        
        # 2. 增强版分析
        user_analysis = self.analyze_pose_sequence_enhanced(user_data)
        template_analysis = self.analyze_pose_sequence_enhanced(template_data)
        
        # 3. 特征级对比
        feature_comparison = self._compare_features(
            user_analysis.get('enhanced_features', {}),
            template_analysis.get('enhanced_features', {})
        )
        
        # 4. 阶段级对比
        stage_comparison = self._compare_detected_stages(
            user_analysis.get('detected_stages', []),
            template_analysis.get('detected_stages', [])
        )
        
        # 5. 运动学对比
        kinematics_comparison = self._compare_kinematics(
            user_analysis.get('kinematics_analysis', {}),
            template_analysis.get('kinematics_analysis', {})
        )
        
        enhanced_comparison = {
            'basic_comparison': basic_comparison,
            'user_analysis': user_analysis,
            'template_analysis': template_analysis,
            'feature_comparison': feature_comparison,
            'stage_comparison': stage_comparison,
            'kinematics_comparison': kinematics_comparison,
            'overall_similarity': self._calculate_overall_similarity(
                feature_comparison, stage_comparison, kinematics_comparison
            )
        }
        
        return enhanced_comparison
    
    def _compare_features(self, user_features: Dict, template_features: Dict) -> Dict[str, Any]:
        """
        对比特征
        
        Args:
            user_features: 用户特征
            template_features: 模板特征
            
        Returns:
            特征对比结果
        """
        comparison = {
            'temporal_diff': {},
            'spatial_diff': {},
            'kinematic_diff': {},
            'coordination_diff': {}
        }
        
        # 对比各类特征
        feature_types = ['temporal_features', 'spatial_features', 'kinematic_features', 'coordination_features']
        
        for feature_type in feature_types:
            user_feat = user_features.get(feature_type, {})
            template_feat = template_features.get(feature_type, {})
            
            diff_key = feature_type.replace('_features', '_diff')
            comparison[diff_key] = self._calculate_feature_differences(user_feat, template_feat)
        
        return comparison
    
    def _calculate_feature_differences(self, user_feat: Dict, template_feat: Dict) -> Dict[str, float]:
        """
        计算特征差异
        
        Args:
            user_feat: 用户特征
            template_feat: 模板特征
            
        Returns:
            特征差异字典
        """
        differences = {}
        
        # 找到共同的特征键
        common_keys = set(user_feat.keys()) & set(template_feat.keys())
        
        for key in common_keys:
            user_val = user_feat[key]
            template_val = template_feat[key]
            
            if isinstance(user_val, (int, float)) and isinstance(template_val, (int, float)):
                if template_val != 0:
                    # 计算相对差异
                    relative_diff = abs(user_val - template_val) / abs(template_val)
                    differences[key] = relative_diff
                else:
                    differences[key] = abs(user_val)
        
        return differences
    
    def _compare_detected_stages(self, user_stages: List[Dict], 
                               template_stages: List[Dict]) -> Dict[str, Any]:
        """
        对比检测到的阶段
        
        Args:
            user_stages: 用户阶段
            template_stages: 模板阶段
            
        Returns:
            阶段对比结果
        """
        comparison = {
            'stage_count_diff': len(user_stages) - len(template_stages),
            'duration_comparison': [],
            'velocity_comparison': [],
            'overall_stage_similarity': 0
        }
        
        # 对比对应阶段
        min_stages = min(len(user_stages), len(template_stages))
        
        for i in range(min_stages):
            user_stage = user_stages[i]
            template_stage = template_stages[i]
            
            # 持续时间对比
            user_duration = user_stage.get('duration', 0)
            template_duration = template_stage.get('duration', 0)
            
            if template_duration > 0:
                duration_ratio = user_duration / template_duration
                comparison['duration_comparison'].append({
                    'stage': user_stage.get('stage_name', f'阶段{i+1}'),
                    'user_duration': user_duration,
                    'template_duration': template_duration,
                    'ratio': duration_ratio
                })
            
            # 速度对比
            user_velocity = user_stage.get('avg_velocity', 0)
            template_velocity = template_stage.get('avg_velocity', 0)
            
            comparison['velocity_comparison'].append({
                'stage': user_stage.get('stage_name', f'阶段{i+1}'),
                'user_velocity': user_velocity,
                'template_velocity': template_velocity,
                'difference': abs(user_velocity - template_velocity)
            })
        
        # 计算整体相似度
        if comparison['duration_comparison']:
            duration_similarities = []
            for comp in comparison['duration_comparison']:
                ratio = comp['ratio']
                # 将比率转换为相似度（1.0表示完全相同）
                similarity = 1.0 / (1.0 + abs(ratio - 1.0))
                duration_similarities.append(similarity)
            
            comparison['overall_stage_similarity'] = np.mean(duration_similarities)
        
        return comparison
    
    def _compare_kinematics(self, user_kinematics: Dict, 
                          template_kinematics: Dict) -> Dict[str, Any]:
        """
        对比运动学特征
        
        Args:
            user_kinematics: 用户运动学数据
            template_kinematics: 模板运动学数据
            
        Returns:
            运动学对比结果
        """
        comparison = {
            'velocity_similarity': 0,
            'angle_similarity': 0,
            'efficiency_comparison': {}
        }
        
        # 速度相似度
        user_vel = user_kinematics.get('velocity_profile', {})
        template_vel = template_kinematics.get('velocity_profile', {})
        
        if 'wrist' in user_vel and 'wrist' in template_vel:
            user_max_vel = user_vel['wrist'].get('max', 0)
            template_max_vel = template_vel['wrist'].get('max', 0)
            
            if template_max_vel > 0:
                vel_ratio = user_max_vel / template_max_vel
                comparison['velocity_similarity'] = 1.0 / (1.0 + abs(vel_ratio - 1.0))
        
        # 角度相似度
        user_angles = user_kinematics.get('angle_analysis', {})
        template_angles = template_kinematics.get('angle_analysis', {})
        
        if 'elbow' in user_angles and 'elbow' in template_angles:
            user_range = user_angles['elbow'].get('range', 0)
            template_range = template_angles['elbow'].get('range', 0)
            
            if template_range > 0:
                range_ratio = user_range / template_range
                comparison['angle_similarity'] = 1.0 / (1.0 + abs(range_ratio - 1.0))
        
        # 效率对比
        user_eff = user_kinematics.get('movement_efficiency', {})
        template_eff = template_kinematics.get('movement_efficiency', {})
        
        if 'trajectory_efficiency' in user_eff and 'trajectory_efficiency' in template_eff:
            comparison['efficiency_comparison'] = {
                'user_efficiency': user_eff['trajectory_efficiency'],
                'template_efficiency': template_eff['trajectory_efficiency'],
                'difference': abs(user_eff['trajectory_efficiency'] - template_eff['trajectory_efficiency'])
            }
        
        return comparison
    
    def _calculate_overall_similarity(self, feature_comp: Dict, stage_comp: Dict, 
                                    kinematics_comp: Dict) -> float:
        """
        计算整体相似度
        
        Args:
            feature_comp: 特征对比结果
            stage_comp: 阶段对比结果
            kinematics_comp: 运动学对比结果
            
        Returns:
            整体相似度分数 (0-1)
        """
        similarities = []
        
        # 阶段相似度
        stage_sim = stage_comp.get('overall_stage_similarity', 0)
        if stage_sim > 0:
            similarities.append(stage_sim)
        
        # 运动学相似度
        vel_sim = kinematics_comp.get('velocity_similarity', 0)
        angle_sim = kinematics_comp.get('angle_similarity', 0)
        
        if vel_sim > 0:
            similarities.append(vel_sim)
        if angle_sim > 0:
            similarities.append(angle_sim)
        
        # 特征相似度（简化计算）
        spatial_diffs = feature_comp.get('spatial_diff', {})
        if spatial_diffs:
            # 计算空间特征的平均相似度
            spatial_similarities = [1.0 / (1.0 + diff) for diff in spatial_diffs.values()]
            if spatial_similarities:
                similarities.append(np.mean(spatial_similarities))
        
        # 返回平均相似度
        return np.mean(similarities) if similarities else 0.5

# 导出主要类
__all__ = ['EnhancedPoseAnalyzer']