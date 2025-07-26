#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作建议智能体
通过对比staged JSON文件和标准模板，生成具体的动作纠正建议
"""

import json
import os
import math
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import base64
from io import BytesIO
import configparser

class ActionAdvisor:
    """
    动作建议智能体：对比用户动作数据与标准模板，生成具体纠正建议
    """
    
    def __init__(self, templates_dir="d:\\羽毛球项目\\templates", 
                 staged_dir="d:\\羽毛球项目\\staged_templates",
                 status_callback=None, streaming_callback=None):
        """
        初始化动作建议智能体
        
        Args:
            templates_dir: 标准模板文件目录
            staged_dir: 用户staged JSON文件目录
            status_callback: 状态回调函数，用于向UI发送连接状态信息
            streaming_callback: 流式内容回调函数（用于实时显示token生成）
        """
        self.templates_dir = templates_dir
        self.staged_dir = staged_dir
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.status_callback = status_callback
        self.streaming_callback = streaming_callback
        
        # 从config.ini文件读取API密钥
        self.api_key = self._load_api_key_from_config()
        
        # 关键角度阈值设置（更严格的标准）
        self.angle_thresholds = {
            "minor": 3,     # 轻微偏差（降低到3度）
            "moderate": 8,  # 中等偏差（降低到8度）
            "major": 15     # 严重偏差（降低到15度）
        }
        
        # 时间偏差阈值（毫秒）（更严格的标准）
        self.time_thresholds = {
            "minor": 100,   # 轻微时间偏差（降低到100ms）
            "moderate": 300, # 中等时间偏差（降低到300ms）
            "major": 600    # 严重时间偏差（降低到600ms）
        }
        
        # 5个击球动作评分维度
        self.score_dimensions = {
            "posture_stability": "姿态稳定性",
            "timing_precision": "击球时机", 
            "movement_fluency": "动作流畅性",
            "power_transfer": "力量传递",
            "technical_standard": "技术规范性"
        }
    
    def _load_api_key_from_config(self) -> str:
        """
        从config.ini文件读取API密钥
        
        Returns:
            API密钥字符串
        """
        try:
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.ini')
            config.read(config_path, encoding='utf-8')
            
            if 'API' in config and 'key' in config['API']:
                api_key = config['API']['key'].strip()
                if api_key:
                    print(f"成功从config.ini加载API密钥: {api_key[:8]}...")
                    return api_key
                else:
                    print("警告: config.ini中的API密钥为空")
                    return ''
            else:
                print("警告: config.ini中未找到API配置")
                return ''
        except Exception as e:
            print(f"读取config.ini失败: {e}")
            return ''
    
    def get_latest_staged_file(self) -> str:
        """
        获取最新的staged JSON文件
        
        Returns:
            最新staged文件的完整路径
        """
        # 查找多种可能的staged文件格式
        staged_files = []
        staged_files.extend(list(Path(self.staged_dir).glob("staged_*.json")))
        staged_files.extend(list(Path(self.staged_dir).glob("staged_*.analysis_data.json")))
        
        if not staged_files:
            raise FileNotFoundError("staged_templates目录中没有找到staged JSON文件")
        
        # 按修改时间排序，返回最新的
        latest_file = max(staged_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def get_template_file(self, template_name="击球动作模板.json") -> str:
        """
        获取标准模板文件
        
        Args:
            template_name: 模板文件名
            
        Returns:
            模板文件的完整路径
        """
        template_path = os.path.join(self.staged_dir, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"模板文件不存在: {template_path}")
        return template_path
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载JSON数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            JSON数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"加载JSON文件失败 {file_path}: {e}")
    
    def compare_stages(self, user_data: List[Dict], template_data: List[Dict]) -> Dict[str, Any]:
        """
        对比用户数据和模板数据的各个阶段
        
        Args:
            user_data: 用户的staged数据
            template_data: 标准模板数据
            
        Returns:
            对比分析结果
        """
        comparison_result = {
            "overall_score": 0,
            "stage_comparisons": [],
            "critical_issues": [],
            "improvement_suggestions": []
        }
        
        # 确保两个数据都有5个阶段
        if len(user_data) != 5 or len(template_data) != 5:
            comparison_result["critical_issues"].append(
                f"阶段数量不匹配：用户数据{len(user_data)}个阶段，模板{len(template_data)}个阶段"
            )
            return comparison_result
        
        total_score = 0
        
        for i, (user_stage, template_stage) in enumerate(zip(user_data, template_data)):
            stage_comparison = self._compare_single_stage(user_stage, template_stage)
            comparison_result["stage_comparisons"].append(stage_comparison)
            total_score += stage_comparison["score"]
            
            # 收集严重问题
            if stage_comparison["score"] < 60:
                comparison_result["critical_issues"].extend(stage_comparison["issues"])
        
        comparison_result["overall_score"] = total_score / 5
        return comparison_result
    
    def _compare_single_stage(self, user_stage: Dict, template_stage: Dict) -> Dict[str, Any]:
        """
        对比单个阶段的数据
        
        Args:
            user_stage: 用户阶段数据
            template_stage: 模板阶段数据
            
        Returns:
            单阶段对比结果
        """
        stage_result = {
            "stage_name": user_stage.get("stage", "未知阶段"),
            "score": 100,
            "timing_analysis": {},
            "angle_analysis": {},
            "issues": [],
            "suggestions": []
        }
        
        # 1. 时间分析
        timing_analysis = self._analyze_timing(
            user_stage.get("start_ms", 0),
            user_stage.get("end_ms", 0),
            template_stage.get("start_ms", 0),
            template_stage.get("end_ms", 0)
        )
        stage_result["timing_analysis"] = timing_analysis
        stage_result["score"] -= timing_analysis["penalty"]
        
        # 2. 角度分析
        angle_analysis = self._analyze_angles(
            user_stage.get("expected_values", {}),
            template_stage.get("expected_values", {})
        )
        stage_result["angle_analysis"] = angle_analysis
        stage_result["score"] -= angle_analysis["penalty"]
        
        # 3. 生成具体建议
        stage_result["suggestions"] = self._generate_stage_suggestions(
            stage_result["stage_name"],
            timing_analysis,
            angle_analysis
        )
        
        # 确保分数不低于0
        stage_result["score"] = max(0, stage_result["score"])
        
        return stage_result
    
    def _analyze_timing(self, user_start: int, user_end: int, 
                       template_start: int, template_end: int) -> Dict[str, Any]:
        """
        分析时间差异
        
        Args:
            user_start, user_end: 用户阶段的开始和结束时间
            template_start, template_end: 模板阶段的开始和结束时间
            
        Returns:
            时间分析结果
        """
        user_duration = user_end - user_start
        template_duration = template_end - template_start
        
        duration_diff = abs(user_duration - template_duration)
        start_diff = abs(user_start - template_start)
        
        penalty = 0
        issues = []
        
        # 持续时间分析（更严格的扣分）
        if duration_diff > self.time_thresholds["major"]:
            penalty += 30  # 增加扣分
            issues.append(f"阶段持续时间偏差过大：{duration_diff}ms")
        elif duration_diff > self.time_thresholds["moderate"]:
            penalty += 20  # 增加扣分
            issues.append(f"阶段持续时间偏差较大：{duration_diff}ms")
        elif duration_diff > self.time_thresholds["minor"]:
            penalty += 10  # 增加扣分
            issues.append(f"阶段持续时间略有偏差：{duration_diff}ms")
        
        # 开始时间分析（更严格的扣分）
        if start_diff > self.time_thresholds["moderate"]:
            penalty += 15  # 增加扣分
            issues.append(f"阶段开始时间偏差：{start_diff}ms")
        elif start_diff > self.time_thresholds["minor"]:
            penalty += 8   # 新增轻微偏差扣分
            issues.append(f"阶段开始时间略有偏差：{start_diff}ms")
        
        return {
            "user_duration": user_duration,
            "template_duration": template_duration,
            "duration_diff": duration_diff,
            "start_diff": start_diff,
            "penalty": penalty,
            "issues": issues
        }
    
    def _analyze_angles(self, user_angles: Dict, template_angles: Dict) -> Dict[str, Any]:
        """
        分析角度差异
        
        Args:
            user_angles: 用户角度数据
            template_angles: 模板角度数据
            
        Returns:
            角度分析结果
        """
        penalty = 0
        issues = []
        angle_details = {}
        
        # 角度名称映射（英文到中文）
        angle_name_mapping = {
            "elbow_angle": "肘部角度",
            "shoulder_angle": "肩部角度", 
            "hip_angle": "髋部角度",
            "wrist_angle": "腕部角度",
            "knee_angle": "膝部角度"
        }
        
        # 分析每个关键角度
        for angle_name in template_angles.keys():
            if angle_name not in user_angles:
                # 对于缺失的角度数据，给予较轻的惩罚
                penalty += 5  # 降低惩罚
                issues.append(f"缺少角度数据：{angle_name}")
                continue
            
            user_angle = user_angles[angle_name]
            template_angle = template_angles[angle_name]
            
            # 获取理想值进行比较
            user_ideal = user_angle.get("ideal", 0)
            template_ideal = template_angle.get("ideal", 0)
            
            angle_diff = abs(user_ideal - template_ideal)
            
            # 获取中文名称用于显示
            display_name = angle_name_mapping.get(angle_name, angle_name)
            
            angle_detail = {
                "user_ideal": user_ideal,
                "template_ideal": template_ideal,
                "difference": angle_diff,
                "severity": "normal"
            }
            
            # 根据角度差异评估严重程度（更严格的扣分）
            if angle_diff > self.angle_thresholds["major"]:
                penalty += 25  # 增加扣分
                angle_detail["severity"] = "major"
                issues.append(f"{display_name}偏差严重：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["moderate"]:
                penalty += 15  # 增加扣分
                angle_detail["severity"] = "moderate"
                issues.append(f"{display_name}偏差较大：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["minor"]:
                penalty += 8   # 增加扣分
                angle_detail["severity"] = "minor"
                issues.append(f"{display_name}略有偏差：{angle_diff:.1f}°")
            
            angle_details[angle_name] = angle_detail
        
        return {
            "penalty": penalty,
            "issues": issues,
            "angle_details": angle_details
        }
    
    def _generate_stage_suggestions(self, stage_name: str, timing_analysis: Dict, 
                                  angle_analysis: Dict) -> List[str]:
        """
        根据分析结果生成具体的改进建议
        
        Args:
            stage_name: 阶段名称
            timing_analysis: 时间分析结果
            angle_analysis: 角度分析结果
            
        Returns:
            改进建议列表
        """
        suggestions = []
        
        # 时间相关建议
        if timing_analysis["duration_diff"] > self.time_thresholds["moderate"]:
            if timing_analysis["user_duration"] > timing_analysis["template_duration"]:
                suggestions.append(f"【{stage_name}】动作过慢，建议加快节奏，缩短{timing_analysis['duration_diff']}ms")
            else:
                suggestions.append(f"【{stage_name}】动作过快，建议放慢节奏，延长{timing_analysis['duration_diff']}ms")
        
        # 角度相关建议
        for angle_name, angle_detail in angle_analysis.get("angle_details", {}).items():
            if angle_detail["severity"] in ["major", "moderate"]:
                user_val = angle_detail["user_ideal"]
                template_val = angle_detail["template_ideal"]
                
                # 获取角度的中文显示名称
                angle_name_mapping = {
                    "elbow_angle": "肘部角度",
                    "shoulder_angle": "肩部角度", 
                    "hip_angle": "髋部角度",
                    "wrist_angle": "腕部角度",
                    "knee_angle": "膝部角度"
                }
                display_name = angle_name_mapping.get(angle_name, angle_name)
                
                if "elbow" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{display_name}过大({user_val:.1f}°)，建议收紧手臂至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{display_name}过小({user_val:.1f}°)，建议展开手臂至{template_val:.1f}°")
                elif "shoulder" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{display_name}过大({user_val:.1f}°)，建议降低肩膀至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{display_name}过小({user_val:.1f}°)，建议抬高肩膀至{template_val:.1f}°")
                elif "hip" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{display_name}过大({user_val:.1f}°)，建议收紧腰部至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{display_name}过小({user_val:.1f}°)，建议放松腰部至{template_val:.1f}°")
        
        return suggestions
    
    def calculate_five_dimension_scores(self, comparison_result: Dict[str, Any]) -> Dict[str, float]:
        """
        计算5个维度的评分
        
        Args:
            comparison_result: 对比分析结果
            
        Returns:
            5个维度的评分字典
        """
        scores = {}
        stage_comparisons = comparison_result.get("stage_comparisons", [])
        
        if not stage_comparisons:
            return {dim: 0 for dim in self.score_dimensions.keys()}
        
        # 1. 姿态稳定性 - 基于关键关节角度的一致性
        posture_scores = []
        for stage in stage_comparisons:
            angle_analysis = stage.get("angle_analysis", {})
            angle_details = angle_analysis.get("angle_details", {})
            
            stage_posture_score = 100
            for angle_name, detail in angle_details.items():
                if "shoulder" in angle_name or "hip" in angle_name:  # 核心姿态角度
                    diff = detail.get("difference", 0)
                    if diff > 15:
                        stage_posture_score -= 25
                    elif diff > 8:
                        stage_posture_score -= 15
                    elif diff > 3:
                        stage_posture_score -= 8
            
            posture_scores.append(max(0, stage_posture_score))
        
        scores["posture_stability"] = sum(posture_scores) / len(posture_scores)
        
        # 2. 击球时机 - 基于时间偏差
        timing_scores = []
        for stage in stage_comparisons:
            timing_analysis = stage.get("timing_analysis", {})
            duration_diff = timing_analysis.get("duration_diff", 0)
            start_diff = timing_analysis.get("start_diff", 0)
            
            stage_timing_score = 100
            if duration_diff > 600:
                stage_timing_score -= 30
            elif duration_diff > 300:
                stage_timing_score -= 20
            elif duration_diff > 100:
                stage_timing_score -= 10
            
            if start_diff > 300:
                stage_timing_score -= 15
            elif start_diff > 100:
                stage_timing_score -= 8
            
            timing_scores.append(max(0, stage_timing_score))
        
        scores["timing_precision"] = sum(timing_scores) / len(timing_scores)
        
        # 3. 动作流畅性 - 基于阶段间的连贯性
        fluency_score = 100
        for i in range(len(stage_comparisons) - 1):
            current_stage = stage_comparisons[i]
            next_stage = stage_comparisons[i + 1]
            
            # 检查阶段转换的流畅性
            current_end = current_stage.get("timing_analysis", {}).get("user_duration", 0)
            next_start = next_stage.get("timing_analysis", {}).get("start_diff", 0)
            
            if abs(next_start) > 200:  # 阶段间隔过大
                fluency_score -= 15
            elif abs(next_start) > 100:
                fluency_score -= 8
        
        scores["movement_fluency"] = max(0, fluency_score)
        
        # 4. 力量传递 - 基于肘部和手腕角度的协调性
        power_scores = []
        for stage in stage_comparisons:
            angle_analysis = stage.get("angle_analysis", {})
            angle_details = angle_analysis.get("angle_details", {})
            
            stage_power_score = 100
            elbow_angles = [detail for name, detail in angle_details.items() if "elbow" in name]
            wrist_angles = [detail for name, detail in angle_details.items() if "wrist" in name]
            
            for detail in elbow_angles + wrist_angles:
                diff = detail.get("difference", 0)
                if diff > 15:
                    stage_power_score -= 20
                elif diff > 8:
                    stage_power_score -= 12
                elif diff > 3:
                    stage_power_score -= 6
            
            power_scores.append(max(0, stage_power_score))
        
        scores["power_transfer"] = sum(power_scores) / len(power_scores) if power_scores else 0
        
        # 5. 技术规范性 - 基于整体评分
        overall_score = comparison_result.get("overall_score", 0)
        scores["technical_standard"] = overall_score
        
        return scores
    
    def generate_radar_chart(self, scores: Dict[str, float]) -> str:
        """
        生成5维度雷达图
        
        Args:
            scores: 5个维度的评分字典
            
        Returns:
            雷达图的base64编码字符串
        """
        try:
            # 设置matplotlib为非交互式后端
            import matplotlib
            matplotlib.use('Agg')  # 使用非GUI后端
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"警告: 设置matplotlib后端失败: {e}")
            # 返回空字符串，让程序继续运行
            return ""
        
        try:
            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # 维度标签和数值
            dimensions = list(self.score_dimensions.values())
            values = [scores.get(key, 0) for key in self.score_dimensions.keys()]
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
            values += values[:1]  # 闭合图形
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4', label='当前表现')
            ax.fill(angles, values, alpha=0.25, color='#1f77b4')
            
            # 添加参考线
            reference_values = [100] * (len(dimensions) + 1)
            ax.plot(angles, reference_values, '--', linewidth=1, color='gray', alpha=0.7, label='满分参考')
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dimensions, fontsize=12)
            
            # 设置径向刻度
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
            ax.grid(True)
            
            # 添加标题和图例
            plt.title('羽毛球击球动作五维度评分', fontsize=16, fontweight='bold', pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            # 在每个维度旁边显示具体分数
            for angle, value, dimension in zip(angles[:-1], values[:-1], dimensions):
                ax.text(angle, value + 5, f'{value:.1f}', 
                       horizontalalignment='center', fontsize=10, fontweight='bold')
            
            # 保存为base64字符串
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"警告: 生成雷达图失败: {e}")
            # 返回空字符串，让程序继续运行
            return ""
    
    def generate_comprehensive_advice(self, user_file_path: str = None, 
                                    template_file_path: str = None) -> Dict[str, Any]:
        """
        生成综合的动作建议报告
        
        Args:
            user_file_path: 用户staged文件路径（可选，默认使用最新文件）
            template_file_path: 模板文件路径（可选，默认使用标准模板）
            
        Returns:
            综合建议报告
        """
        try:
            # 获取文件路径
            if not user_file_path:
                user_file_path = self.get_latest_staged_file()
            if not template_file_path:
                template_file_path = self.get_template_file()
            
            # 加载数据
            user_data = self.load_json_data(user_file_path)
            template_data = self.load_json_data(template_file_path)
            
            # 进行对比分析
            comparison_result = self.compare_stages(user_data, template_data)
            
            # 计算5维度评分
            dimension_scores = self.calculate_five_dimension_scores(comparison_result)
            
            # 生成雷达图
            radar_chart_base64 = self.generate_radar_chart(dimension_scores)
            
            # 生成LLM增强建议（使用流式版本）
            llm_advice = self._generate_llm_advice_streaming(comparison_result, user_data, template_data)
            
            # 构建最终报告
            final_report = {
                "analysis_timestamp": self._get_timestamp(),
                "user_file": os.path.basename(user_file_path),
                "template_file": os.path.basename(template_file_path),
                "overall_score": comparison_result["overall_score"],
                "performance_level": self._get_performance_level(comparison_result["overall_score"]),
                "dimension_scores": dimension_scores,
                "radar_chart": radar_chart_base64,
                "stage_analysis": comparison_result["stage_comparisons"],
                "critical_issues": comparison_result["critical_issues"],
                "detailed_suggestions": self._collect_all_suggestions(comparison_result),
                "llm_enhanced_advice": llm_advice,
                "practice_plan": self._generate_practice_plan(comparison_result)
            }
            
            return final_report
            
        except Exception as e:
            return {
                "error": f"生成建议失败: {str(e)}",
                "analysis_timestamp": self._get_timestamp()
            }
    
    def generate_advice(self, user_file_path: str = None, template_file_path: str = None, 
                       return_format: str = "dict") -> Any:
        """
        生成动作建议的便捷方法
        
        Args:
            user_file_path: 用户staged文件路径（可选）
            template_file_path: 模板文件路径（可选）
            return_format: 返回格式 ("dict", "readable", "both")
            
        Returns:
            根据return_format返回不同格式的建议
        """
        # 生成完整报告
        report = self.generate_comprehensive_advice(user_file_path, template_file_path)
        
        if "error" in report:
            return report
        
        if return_format == "readable":
            # 只返回可读性文本
            return self.generate_ui_friendly_report(report)
        elif return_format == "both":
            # 返回两种格式
            return {
                "raw_data": report,
                "readable_text": self.generate_ui_friendly_report(report)
            }
        else:
            # 默认返回原始数据
            return report
    
    def _generate_llm_advice(self, comparison_result: Dict, user_data: List[Dict], 
                           template_data: List[Dict]) -> str:
        """
        使用LLM生成增强的建议
        
        Args:
            comparison_result: 对比分析结果
            user_data: 用户数据
            template_data: 模板数据
            
        Returns:
            LLM生成的建议文本
        """
        if not self.api_key:
            return "未配置API密钥，无法生成LLM增强建议"
    
    def _generate_llm_advice_streaming(self, comparison_result: Dict, user_data: List[Dict], 
                                     template_data: List[Dict]) -> str:
        """
        使用LLM生成增强的建议（流式版本）
        
        Args:
            comparison_result: 对比分析结果
            user_data: 用户数据
            template_data: 模板数据
            
        Returns:
            LLM生成的建议文本
        """
        if not self.api_key:
            return "未配置API密钥，无法生成LLM增强建议"
        
        # 构建提示词
        prompt = f"""
你是一位专业的羽毛球教练，请基于以下动作分析数据，为学员提供具体、可操作的训练建议。

【整体评分】: {comparison_result['overall_score']:.1f}/100

【关键问题】:
{chr(10).join(comparison_result['critical_issues']) if comparison_result['critical_issues'] else '无严重问题'}

【各阶段表现】:
{self._format_stage_summary(comparison_result['stage_comparisons'])}

请提供：
1. 针对性的技术纠正建议（具体到身体部位和角度）
2. 训练重点和练习方法
3. 循序渐进的改进计划
4. 注意事项和常见错误避免

要求：建议要具体、实用，避免泛泛而谈。
"""
        
        # 调试信息：检查API密钥
        if not self.api_key:
            error_msg = "❌ API密钥未配置或为空\n🔄 切换到本地分析模式"
            self._send_status(error_msg)
            print("错误: API密钥未配置")
            return self._generate_fallback_advice(comparison_result)
        
        print(f"调试信息: API密钥已加载 (长度: {len(self.api_key)})")
        print(f"调试信息: API URL: {self.api_url}")
        
        self._send_status("🔗 开始连接AI服务器...")
        self._send_streaming_status("🤖 AI教练正在思考中...\n\n")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "doubao-seed-1-6-thinking-250715",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "stream": True  # 启用流式响应
            }
            
            self._send_status("📡 发送流式请求数据...")
            print(f"调试信息: 发送流式POST请求到 {self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=300,  # 5分钟超时
                stream=True,  # 启用流式接收
                proxies={'http': None, 'https': None}
            )
            
            response.raise_for_status()
            
            self._send_status("📥 开始接收流式响应...")
            
            # 处理流式响应
            full_content = ""
            import json
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # 移除 'data: ' 前缀
                        
                        if data_str.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content_chunk = delta['content']
                                    full_content += content_chunk
                                    # 实时发送每个token到UI
                                    self._send_streaming_status(content_chunk)
                        except json.JSONDecodeError:
                            continue
            
            self._send_status("✅ AI服务流式调用成功！")
            print("LLM流式调用成功！")
            return full_content
            
        except requests.exceptions.Timeout:
            error_msg = "⏰ 流式连接超时\n🔄 切换到本地分析模式"
            self._send_status(error_msg)
            print("LLM流式调用超时，使用本地分析结果")
            return self._generate_fallback_advice(comparison_result)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"❌ 网络连接失败: {str(e)}\n🔄 切换到本地分析模式"
            self._send_status(error_msg)
            print(f"网络连接错误: {str(e)}")
            return self._generate_fallback_advice(comparison_result)
        except Exception as e:
            error_msg = f"❌ AI服务流式调用失败: {str(e)}\n🔄 切换到本地分析模式"
            self._send_status(error_msg)
            print(f"LLM流式调用失败: {str(e)}")
            return self._generate_fallback_advice(comparison_result)
        
        # 构建提示词
        prompt = f"""
你是一位专业的羽毛球教练，请基于以下动作分析数据，为学员提供具体、可操作的训练建议。

【整体评分】: {comparison_result['overall_score']:.1f}/100

【关键问题】:
{chr(10).join(comparison_result['critical_issues']) if comparison_result['critical_issues'] else '无严重问题'}

【各阶段表现】:
{self._format_stage_summary(comparison_result['stage_comparisons'])}

请提供：
1. 针对性的技术纠正建议（具体到身体部位和角度）
2. 训练重点和练习方法
3. 循序渐进的改进计划
4. 注意事项和常见错误避免

要求：建议要具体、实用，避免泛泛而谈。
"""
        
        # 尝试多次调用LLM，增加容错性
        max_retries = 3
        timeout_values = [60, 180, 300]  # 递增的超时时间（1分钟、3分钟、5分钟）
        
        # 调试信息：检查API密钥
        if not self.api_key:
            error_msg = "❌ API密钥未配置或为空\n🔄 切换到本地分析模式"
            self._send_status(error_msg)
            print("错误: API密钥未配置")
            return self._generate_fallback_advice(comparison_result)
        
        print(f"调试信息: API密钥已加载 (长度: {len(self.api_key)})")
        print(f"调试信息: API URL: {self.api_url}")
        
        self._send_status("🔗 开始连接AI服务器...")
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": "doubao-seed-1-6-thinking-250715",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1024 if attempt > 0 else 2048,  # 重试时减少token数量
                    "top_p": 0.9
                }
                
                # 使用递增的超时时间
                current_timeout = timeout_values[attempt]
                status_msg = f"🌐 正在调用AI服务 (尝试 {attempt + 1}/{max_retries})\n   服务器: ark.cn-beijing.volces.com\n   超时设置: {current_timeout}秒\n   模型: doubao-seed-1-6-thinking-250715"
                self._send_status(status_msg)
                print(f"正在调用LLM (尝试 {attempt + 1}/{max_retries}, 超时: {current_timeout}秒)...")
                
                self._send_status("📡 发送请求数据...")
                print(f"调试信息: 发送POST请求到 {self.api_url}")
                print(f"调试信息: 请求头包含Authorization: Bearer {self.api_key[:10]}...")
                print(f"调试信息: 请求数据模型: {data['model']}")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=current_timeout,
                    proxies={'http': None, 'https': None}
                )
                
                self._send_status("📥 接收服务器响应...")
                print(f"调试信息: 响应状态码: {response.status_code}")
                print(f"调试信息: 响应头: {dict(response.headers)}")
                response.raise_for_status()
                
                result = response.json()
                self._send_status("✅ AI服务调用成功！正在处理响应数据...")
                print("LLM调用成功！")
                return result['choices'][0]['message']['content']
                
            except requests.exceptions.Timeout:
                timeout_msg = f"⏰ 连接超时 ({current_timeout}秒)"
                if attempt < max_retries - 1:
                    retry_msg = f"{timeout_msg}，正在重试 ({attempt + 1}/{max_retries})..."
                    self._send_status(retry_msg)
                    print(f"LLM调用超时 ({current_timeout}秒)，正在重试 ({attempt + 1}/{max_retries})...")
                    continue
                else:
                    final_msg = f"{timeout_msg}，所有重试已用完\n🔄 切换到本地分析模式"
                    self._send_status(final_msg)
                    print("LLM调用最终超时，使用本地分析结果")
                    return self._generate_fallback_advice(comparison_result)
            except requests.exceptions.ConnectionError as e:
                error_msg = f"❌ 网络连接失败: {str(e)}\n🔄 切换到本地分析模式"
                self._send_status(error_msg)
                print(f"网络连接错误: {str(e)}")
                return self._generate_fallback_advice(comparison_result)
            except Exception as e:
                if attempt < max_retries - 1:
                    retry_msg = f"⚠️ 调用失败，正在重试 ({attempt + 1}/{max_retries})\n   错误: {str(e)}"
                    self._send_status(retry_msg)
                    print(f"LLM调用失败，正在重试 ({attempt + 1}/{max_retries}): {str(e)}")
                    continue
                else:
                    final_msg = f"❌ AI服务最终失败: {str(e)}\n🔄 切换到本地分析模式"
                    self._send_status(final_msg)
                    print(f"LLM调用最终失败: {str(e)}")
                    return self._generate_fallback_advice(comparison_result)
    
    def _send_status(self, message: str) -> None:
        """
        发送状态信息到UI
        
        Args:
            message: 状态消息
        """
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception as e:
                print(f"状态回调失败: {e}")
    
    def _send_streaming_status(self, content: str) -> None:
        """
        发送流式内容到UI（用于实时显示token生成）
        
        Args:
            content: 流式内容片段
        """
        if hasattr(self, 'streaming_callback') and self.streaming_callback:
            try:
                self.streaming_callback(content)
            except Exception as e:
                print(f"流式回调失败: {e}")
        else:
            # 如果没有专门的流式回调，使用普通状态回调
            if self.status_callback:
                try:
                    self.status_callback(content)
                except Exception as e:
                    print(f"流式状态回调失败: {e}")
    
    def _generate_fallback_advice(self, comparison_result: Dict) -> str:
        """
        生成本地降级建议（当LLM不可用时）
        
        Args:
            comparison_result: 对比分析结果
            
        Returns:
            本地生成的建议文本
        """
        advice_lines = []
        advice_lines.append("📋 基于本地分析的专业建议")
        advice_lines.append("="*30)
        
        overall_score = comparison_result.get('overall_score', 0)
        
        # 根据整体评分给出总体建议
        if overall_score >= 85:
            advice_lines.append("🎯 整体表现: 您的技术水平很不错！")
            advice_lines.append("💡 建议重点: 继续保持现有水平，注重动作的一致性和稳定性。")
        elif overall_score >= 65:
            advice_lines.append("🎯 整体表现: 基础扎实，但仍有改进空间。")
            advice_lines.append("💡 建议重点: 重点改进标记的问题区域，循序渐进提升技术。")
        else:
            advice_lines.append("🎯 整体表现: 需要系统性的技术改进。")
            advice_lines.append("💡 建议重点: 从基础动作开始，建立正确的动作模式。")
        
        advice_lines.append("")
        
        # 分析各阶段问题并给出具体建议
        stage_comparisons = comparison_result.get('stage_comparisons', [])
        problem_stages = [stage for stage in stage_comparisons if stage.get('score', 0) < 75]
        
        if problem_stages:
            advice_lines.append("🔧 重点改进阶段:")
            for stage in problem_stages:
                stage_name = stage.get('stage_name', '未知阶段')
                score = stage.get('score', 0)
                advice_lines.append(f"\n📍 {stage_name} (得分: {score:.0f}/100)")
                
                # 基于角度分析给出建议
                angle_analysis = stage.get('angle_analysis', {})
                if angle_analysis.get('issues'):
                    advice_lines.append("   角度问题:")
                    for issue in angle_analysis['issues'][:3]:  # 最多显示3个问题
                        advice_lines.append(f"   • {issue}")
                
                # 基于时间分析给出建议
                timing_analysis = stage.get('timing_analysis', {})
                if timing_analysis.get('issues'):
                    advice_lines.append("   时机问题:")
                    for issue in timing_analysis['issues'][:2]:  # 最多显示2个问题
                        advice_lines.append(f"   • {issue}")
                
                # 给出具体的改进建议
                suggestions = stage.get('suggestions', [])
                if suggestions:
                    advice_lines.append("   改进建议:")
                    for suggestion in suggestions[:3]:  # 最多显示3个建议
                        advice_lines.append(f"   ✓ {suggestion}")
        
        advice_lines.append("")
        advice_lines.append("📚 训练建议:")
        
        # 根据评分给出训练建议
        if overall_score < 50:
            advice_lines.append("1. 从基础挥拍动作开始，重点练习正确的握拍和站位")
            advice_lines.append("2. 分解练习各个击球阶段，确保每个动作都标准")
            advice_lines.append("3. 建议寻求专业教练指导，建立正确的动作模式")
        elif overall_score < 75:
            advice_lines.append("1. 重点改进得分较低的动作阶段")
            advice_lines.append("2. 加强动作连贯性练习，提高整体流畅度")
            advice_lines.append("3. 注意击球时机的把握，多做节奏练习")
        else:
            advice_lines.append("1. 保持现有技术水平，注重动作的稳定性")
            advice_lines.append("2. 可以尝试更高难度的技术动作")
            advice_lines.append("3. 重点提升动作的精准度和一致性")
        
        advice_lines.append("")
        advice_lines.append("⚠️ 注意: 由于网络问题，本次分析使用本地算法生成。")
        advice_lines.append("建议网络恢复后重新分析以获得更详细的AI建议。")
        
        return "\n".join(advice_lines)
    
    def _format_stage_summary(self, stage_comparisons: List[Dict]) -> str:
        """
        格式化阶段摘要信息
        
        Args:
            stage_comparisons: 阶段对比结果列表
            
        Returns:
            格式化的摘要文本
        """
        summary_lines = []
        for stage in stage_comparisons:
            stage_name = stage['stage_name']
            score = stage['score']
            issues = stage.get('issues', [])
            
            summary_lines.append(f"- {stage_name}: {score:.1f}分")
            if issues:
                for issue in issues[:2]:  # 只显示前2个问题
                    summary_lines.append(f"  问题: {issue}")
        
        return chr(10).join(summary_lines)
    
    def _collect_all_suggestions(self, comparison_result: Dict) -> List[str]:
        """
        收集所有详细建议
        
        Args:
            comparison_result: 对比分析结果
            
        Returns:
            所有建议的列表
        """
        all_suggestions = []
        for stage in comparison_result['stage_comparisons']:
            all_suggestions.extend(stage.get('suggestions', []))
        return all_suggestions
    
    def _generate_practice_plan(self, comparison_result: Dict) -> Dict[str, List[str]]:
        """
        生成练习计划
        
        Args:
            comparison_result: 对比分析结果
            
        Returns:
            分阶段的练习计划
        """
        practice_plan = {
            "immediate_focus": [],  # 立即重点练习
            "short_term": [],      # 短期目标（1-2周）
            "long_term": []        # 长期目标（1个月以上）
        }
        
        # 根据评分确定练习重点（更严格的标准）
        for stage in comparison_result['stage_comparisons']:
            stage_name = stage['stage_name']
            score = stage['score']
            
            if score < 65:  # 提高标准
                practice_plan["immediate_focus"].append(f"重点练习{stage_name}阶段的基本动作")
            elif score < 85:  # 提高标准
                practice_plan["short_term"].append(f"改进{stage_name}阶段的技术细节")
            else:
                practice_plan["long_term"].append(f"优化{stage_name}阶段的动作流畅性")
        
        return practice_plan
    
    def _get_performance_level(self, score: float) -> str:
        """
        根据评分获取表现等级（更严格的标准）
        
        Args:
            score: 综合评分
            
        Returns:
            表现等级描述
        """
        if score >= 95:
            return "优秀 - 动作标准，继续保持"
        elif score >= 85:
            return "良好 - 基本标准，需要细节优化"
        elif score >= 75:
            return "中等 - 有明显问题，需要重点改进"
        elif score >= 65:
            return "及格 - 问题较多，需要系统训练"
        else:
            return "不及格 - 基础薄弱，建议从基本动作开始"
    
    def _get_timestamp(self) -> str:
        """
        获取当前时间戳
        
        Returns:
            格式化的时间戳
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_ui_friendly_report(self, report: Dict[str, Any]) -> str:
        """
        生成适合UI展示的可读性报告
        
        Args:
            report: 建议报告数据
            
        Returns:
            格式化的可读性报告文本
        """
        lines = []
        
        # 标题和基本信息
        lines.append("🏸 羽毛球动作分析报告")
        lines.append("=" * 30)
        lines.append(f"📅 分析时间: {report.get('analysis_timestamp', '未知')}")
        lines.append(f"📁 分析文件: {report.get('user_file', '未知')}")
        lines.append("")
        
        # 整体评分
        score = report.get('overall_score', 0)
        level = report.get('performance_level', '未知')
        score_emoji = self._get_score_emoji(score)
        lines.append(f"📊 整体评分: {score:.1f}/100 {score_emoji}")
        lines.append(f"🎯 技术水平: {level}")
        lines.append("")
        
        # 各阶段详细分析
        lines.append("📋 各阶段分析详情")
        lines.append("-" * 25)
        
        stage_analysis = report.get('stage_analysis', [])
        for i, stage in enumerate(stage_analysis, 1):
            stage_name = stage.get('stage_name', '未知阶段')
            stage_score = stage.get('score', 0)
            stage_emoji = self._get_score_emoji(stage_score)
            
            lines.append(f"{i}. 【{stage_name}】 {stage_score:.0f}/100 {stage_emoji}")
            
            # 时间分析
            timing = stage.get('timing_analysis', {})
            if timing.get('issues'):
                lines.append("   ⏱️ 时间节奏问题:")
                for issue in timing['issues']:
                    lines.append(f"     • {issue}")
            
            # 角度分析
            angle_analysis = stage.get('angle_analysis', {})
            if angle_analysis.get('issues'):
                lines.append("   📐 角度偏差问题:")
                for issue in angle_analysis['issues']:
                    lines.append(f"     • {issue}")
            
            # 改进建议
            suggestions = stage.get('suggestions', [])
            if suggestions:
                lines.append("   💡 改进建议:")
                for suggestion in suggestions:
                    lines.append(f"     ✓ {suggestion}")
            
            if stage_score >= 85:
                lines.append("   ✅ 该阶段表现良好，继续保持！")
            
            lines.append("")
        
        # 关键问题汇总
        critical_issues = report.get('critical_issues', [])
        if critical_issues:
            lines.append("⚠️ 关键问题汇总")
            lines.append("-" * 20)
            for issue in critical_issues:
                lines.append(f"❌ {issue}")
            lines.append("")
        
        # 详细建议
        detailed_suggestions = report.get('detailed_suggestions', [])
        if detailed_suggestions:
            lines.append("🎯 具体改进建议")
            lines.append("-" * 20)
            for i, suggestion in enumerate(detailed_suggestions, 1):
                lines.append(f"{i}. {suggestion}")
            lines.append("")
        
        # 训练计划
        practice_plan = report.get('practice_plan', {})
        if any(practice_plan.values()):
            lines.append("📚 个性化训练计划")
            lines.append("-" * 22)
            
            immediate = practice_plan.get('immediate_focus', [])
            if immediate:
                lines.append("🔥 立即重点练习:")
                for item in immediate:
                    lines.append(f"   • {item}")
                lines.append("")
            
            short_term = practice_plan.get('short_term', [])
            if short_term:
                lines.append("📈 短期目标 (1-2周):")
                for item in short_term:
                    lines.append(f"   • {item}")
                lines.append("")
            
            long_term = practice_plan.get('long_term', [])
            if long_term:
                lines.append("🎯 长期目标 (1个月+):")
                for item in long_term:
                    lines.append(f"   • {item}")
                lines.append("")
        
        # LLM增强建议
        llm_advice = report.get('llm_enhanced_advice', '')
        if llm_advice and "未配置API密钥" not in llm_advice and "失败" not in llm_advice:
            lines.append("🤖 AI教练专业建议")
            lines.append("-" * 22)
            lines.append(llm_advice)
            lines.append("")
        
        # 结尾鼓励
        if score >= 85:
            lines.append("🌟 总结: 您的技术水平很不错，继续保持并精益求精！")
        elif score >= 65:
            lines.append("💪 总结: 基础扎实，重点改进标记的问题，很快就能提升！")
        else:
            lines.append("🚀 总结: 还有很大提升空间，按照建议坚持练习，进步会很明显！")
        
        return "\n".join(lines)
    
    def _get_score_emoji(self, score: float) -> str:
        """
        根据分数获取对应的表情符号
        
        Args:
            score: 分数
            
        Returns:
            表情符号
        """
        if score >= 95:
            return "🏆"
        elif score >= 85:
            return "😊"
        elif score >= 75:
            return "😐"
        elif score >= 65:
            return "😕"
        else:
            return "😰"
    
    def save_advice_report(self, report: Dict[str, Any], output_path: str = None, 
                          format_type: str = "json") -> str:
        """
        保存建议报告到文件
        
        Args:
            report: 建议报告数据
            output_path: 输出文件路径（可选）
            format_type: 输出格式 ("json" 或 "txt")
            
        Returns:
            保存的文件路径
        """
        if not output_path:
            timestamp = report.get('analysis_timestamp', 'unknown').replace(':', '-').replace(' ', '_')
            if format_type == "txt":
                output_path = os.path.join(self.staged_dir, f"advice_report_{timestamp}.txt")
            else:
                output_path = os.path.join(self.staged_dir, f"advice_report_{timestamp}.json")
        
        try:
            if format_type == "txt":
                # 保存可读性报告
                readable_report = self.generate_ui_friendly_report(report)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(readable_report)
            else:
                # 保存JSON格式
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            return output_path
        except Exception as e:
            raise Exception(f"保存报告失败: {e}")