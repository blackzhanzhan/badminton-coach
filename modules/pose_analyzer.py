#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import requests
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PoseAnalyzer:
    def __init__(self):
        """初始化姿势分析器"""
        self.feedback = []
        
        # MediaPipe关键点索引映射
        self.landmarks_info = {
            "NOSE": 0,
            "LEFT_EYE_INNER": 1,
            "LEFT_EYE": 2,
            "LEFT_EYE_OUTER": 3,
            "RIGHT_EYE_INNER": 4,
            "RIGHT_EYE": 5,
            "RIGHT_EYE_OUTER": 6,
            "LEFT_EAR": 7,
            "RIGHT_EAR": 8,
            "MOUTH_LEFT": 9,
            "MOUTH_RIGHT": 10,
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13,
            "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15,
            "RIGHT_WRIST": 16,
            "LEFT_PINKY": 17,
            "RIGHT_PINKY": 18,
            "LEFT_INDEX": 19,
            "RIGHT_INDEX": 20,
            "LEFT_THUMB": 21,
            "RIGHT_THUMB": 22,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
            "LEFT_HEEL": 29,
            "RIGHT_HEEL": 30,
            "LEFT_FOOT_INDEX": 31,
            "RIGHT_FOOT_INDEX": 32
        }

    def analyze_pose(self, landmarks):
        """
        分析单帧姿势并提供反馈。

        Args:
            landmarks (list): 包含所有关键点的列表。

        Returns:
            list: 包含分析建议的字符串列表。
        """
        if not landmarks:
            return ["画面中未检测到目标。"]

        self.feedback.clear()

        # 简化analyze_pose为：self.feedback.append("实时分析中...")
        self.feedback.append("实时分析中...")
        
        return self.feedback.copy()

    def analyze_json_difference(self, standard_json_path, learner_json_path):
        """比较JSON并用大模型生成建议（带DTW对齐）"""
        try:
            with open(standard_json_path, 'r') as f:
                standard_data = json.load(f)
            with open(learner_json_path, 'r') as f:
                learner_data = json.load(f)
        except Exception as e:
            return [f"加载JSON失败: {e}"]

        if not standard_data or not learner_data:
            return ["JSON数据为空，无法比较。"]

        # 提取特征序列（示例：右肘角度 + 右肩角度，作为多维序列）
        def extract_angle_sequence(data):
            sequence = []
            for frame in data:
                landmarks = frame.get('landmarks', {})
                elbow_angle = self._calculate_angle(
                    self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                    self._get_landmark(landmarks, "RIGHT_ELBOW"),
                    self._get_landmark(landmarks, "RIGHT_WRIST")
                ) or 0
                shoulder_angle = self._calculate_angle(
                    self._get_landmark(landmarks, "RIGHT_HIP"),
                    self._get_landmark(landmarks, "RIGHT_SHOULDER"),
                    self._get_landmark(landmarks, "RIGHT_ELBOW")
                ) or 0
                sequence.append([elbow_angle, shoulder_angle])  # 多维特征
            return np.array(sequence)

        std_seq = extract_angle_sequence(standard_data)
        learn_seq = extract_angle_sequence(learner_data)

        if len(std_seq) == 0 or len(learn_seq) == 0:
            return ["无法提取有效角度序列。"]

        # 应用DTW计算对齐距离和路径
        distance, path = fastdtw(std_seq, learn_seq, dist=euclidean)

        # 计算对齐后的平均偏差
        aligned_diffs = []
        for i, j in path:
            diff = np.linalg.norm(std_seq[i] - learn_seq[j])  # 欧氏距离
            aligned_diffs.append(diff)
        avg_aligned_diff = np.mean(aligned_diffs) if aligned_diffs else 0

        # 节奏差异：路径长度 vs. 序列长度
        duration_ratio = len(learn_seq) / len(std_seq) if len(std_seq) > 0 else 1
        rhythm_suggestion = f"动作节奏{'慢' if duration_ratio > 1.2 else '快'}了 {abs(duration_ratio - 1) * 100:.1f}%"

        # 用大模型生成建议
        prompt = f"作为羽毛球教练，基于以下接球动作差异给出纠正建议：平均对齐偏差 {avg_aligned_diff:.1f}°，{rhythm_suggestion}。重点关注挥拍和恢复阶段。"
        suggestions = ["默认建议: 动作相似，但节奏稍慢，建议加速准备阶段。"]

        try:
            api_key = os.environ.get('VOLCENGINE_API_KEY', '')
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            data = {
                "model": "doubao-seed-1-6-thinking-250715",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 4096,
                "top_p": 0.95
            }
            
            # 禁用代理
            proxies = {
                'http': None,
                'https': None
            }
            
            response = requests.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions", 
                                   headers=headers, json=data, proxies=proxies)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            suggestions = [content]
        except Exception as e:
            suggestions.append(f"API调用失败: {e}")

        return suggestions

    def segment_actions_with_llm(self, json_path, template_path, num_stages=5):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            with open(template_path, 'r') as f:
                template_data = json.load(f)
        except Exception as e:
            return f"加载JSON失败: {str(e)}"
        template_str = json.dumps(template_data)
        batch_size = 500
        staged_json = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            simplified_data = json.dumps(batch)
            prompt = f"你是一个羽毛球动作分析专家。任务: 从用户JSON（time_ms和landmarks）中提取并转化成阶段化JSON。过程: 1. 参考模板{template_str}的格式和expected_values范围。2. 分析用户数据，分成5个阶段（准备、移动/接近、后摆、击球/前挥、收势）。3. 对于每个阶段，计算start_ms/end_ms，写description，设置expected_values（基于模板调整），选择5个代表key_landmarks（精简到关键点0,4,7,8,11）。4. 输出纯JSON数组，确保格式一致，便于后续LLM对比给出改进建议。用户数据批次: {simplified_data}"
            try:
                api_key = os.environ.get('VOLCENGINE_API_KEY', '')
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                
                data = {
                    "model": "doubao-seed-1-6-thinking-250715",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 4096,
                    "top_p": 0.95
                }
                
                # 禁用代理
                proxies = {
                    'http': None,
                    'https': None
                }
                
                response = requests.post("https://ark.cn-beijing.volces.com/api/v3/chat/completions", 
                                       headers=headers, json=data, proxies=proxies)
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                batch_stages = json.loads(content)
                staged_json.extend(batch_stages)
            except Exception as e:
                return f"API调用或JSON解析失败: {str(e)}"
        # 简单合并重叠阶段（可选逻辑）
        return staged_json

    def _get_landmark(self, landmarks, name):
        """通过名称获取关键点坐标"""
        try:
            index = self.landmarks_info[name]
            # 使用 .get() 避免当landmarks中不存在某个索引时引发KeyError
            return landmarks.get(index)
        except KeyError:
            # 当 landmarks_info 中没有这个名字时
            # print(f"警告: 无法在landmarks_info中找到名为 '{name}' 的关键点定义。")
            return None

    def _calculate_angle(self, p1, p2, p3):
        """
        计算由三个点p1, p2, p3组成的角度，p2为顶点。
        点以 {'x': x, 'y': y, 'confidence': c} 的字典形式提供。
        """
        if not all([p1, p2, p3]):
            return None
        
        # 使用置信度进行检查，可以适当降低分析时的阈值
        if p1['confidence'] < 0.3 or p2['confidence'] < 0.3 or p3['confidence'] < 0.3:
            return None

        try:
            # 使用 numpy 进行矢量计算，更高效稳定
            p1_np = np.array([p1['x'], p1['y']])
            p2_np = np.array([p2['x'], p2['y']])
            p3_np = np.array([p3['x'], p3['y']])

            v1 = p1_np - p2_np
            v2 = p3_np - p2_np

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            # 避免除以零
            if norm_v1 == 0 or norm_v2 == 0:
                return None

            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
            
            # 夹逼余弦值到[-1, 1]，防止浮点误差导致 acos 失败
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
            return angle
        except (ValueError, ZeroDivisionError, KeyError):
             # 增加KeyError以防字典缺少'x', 'y', 'confidence'
            return None