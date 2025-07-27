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
        
        # 移除评分维度相关配置
    
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
        
        for i, (user_stage, template_stage) in enumerate(zip(user_data, template_data)):
            stage_comparison = self._compare_single_stage(user_stage, template_stage)
            comparison_result["stage_comparisons"].append(stage_comparison)
            
            # 收集关键问题
            timing_issues = stage_comparison.get("timing_analysis", {}).get("issues", [])
            angle_issues = stage_comparison.get("angle_analysis", {}).get("issues", [])
            
            # 将严重的时间和角度问题添加到关键问题列表
            for issue in timing_issues + angle_issues:
                if "偏差过大" in issue or "偏差严重" in issue:
                    comparison_result["critical_issues"].append(issue)
        
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
        stage_name = user_stage.get("stage", "未知阶段")
        stage_result = {
            "stage_name": stage_name,
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
        
        # 2. 角度分析
        angle_analysis = self._analyze_angles(
            user_stage.get("expected_values", {}),
            template_stage.get("expected_values", {})
        )
        stage_result["angle_analysis"] = angle_analysis
        
        # 3. 生成具体建议
        stage_result["suggestions"] = self._generate_stage_suggestions(
            stage_result["stage_name"],
            timing_analysis,
            angle_analysis
        )
        
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
        
        issues = []
        
        # 持续时间分析
        if duration_diff > self.time_thresholds["major"]:
            issues.append(f"阶段持续时间偏差过大：{self._format_time_diff(duration_diff)}")
        elif duration_diff > self.time_thresholds["moderate"]:
            issues.append(f"阶段持续时间偏差较大：{self._format_time_diff(duration_diff)}")
        elif duration_diff > self.time_thresholds["minor"]:
            issues.append(f"阶段持续时间略有偏差：{self._format_time_diff(duration_diff)}")
        
        # 开始时间分析
        if start_diff > self.time_thresholds["moderate"]:
            issues.append(f"阶段开始时间偏差：{self._format_time_diff(start_diff)}")
        elif start_diff > self.time_thresholds["minor"]:
            issues.append(f"阶段开始时间略有偏差：{self._format_time_diff(start_diff)}")
        
        return {
            "user_duration": user_duration,
            "template_duration": template_duration,
            "duration_diff": duration_diff,
            "start_diff": start_diff,
            "issues": issues
        }
    
    def _format_time_diff(self, time_ms: int) -> str:
        """
        将毫秒时间差转换为更友好的显示格式
        
        Args:
            time_ms: 时间差（毫秒）
            
        Returns:
            格式化的时间字符串
        """
        if time_ms >= 1000:
            seconds = time_ms / 1000
            return f"{seconds:.1f}秒"
        else:
            return f"{time_ms}毫秒"
    
    def _analyze_angles(self, user_angles: Dict, template_angles: Dict) -> Dict[str, Any]:
        """
        分析角度差异
        
        Args:
            user_angles: 用户角度数据
            template_angles: 模板角度数据
            
        Returns:
            角度分析结果
        """
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
            
            # 根据角度差异评估严重程度
            if angle_diff > self.angle_thresholds["major"]:
                angle_detail["severity"] = "major"
                issues.append(f"{display_name}偏差严重：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["moderate"]:
                angle_detail["severity"] = "moderate"
                issues.append(f"{display_name}偏差较大：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["minor"]:
                angle_detail["severity"] = "minor"
                issues.append(f"{display_name}略有偏差：{angle_diff:.1f}°")
            
            angle_details[angle_name] = angle_detail
        
        return {
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
            time_diff_str = self._format_time_diff(timing_analysis['duration_diff'])
            if timing_analysis["user_duration"] > timing_analysis["template_duration"]:
                suggestions.append(f"【{stage_name}】动作过慢，建议加快节奏，缩短{time_diff_str}")
            else:
                suggestions.append(f"【{stage_name}】动作过快，建议放慢节奏，延长{time_diff_str}")
        
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
            
            # 生成LLM增强建议（使用流式版本）
            llm_response = self._generate_llm_advice_streaming(comparison_result, user_data, template_data)
            
            # 提取LLM返回的建议
            if isinstance(llm_response, dict):
                llm_advice = llm_response.get("advice", "AI建议生成失败")
            else:
                llm_advice = llm_response if isinstance(llm_response, str) else "AI建议生成失败"
            
            # 构建最终报告（移除评分相关内容）
            final_report = {
                "analysis_timestamp": self._get_timestamp(),
                "user_file": os.path.basename(user_file_path),
                "template_file": os.path.basename(template_file_path),
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
        使用LLM生成增强的建议（非流式版本，作为备用）
        
        Args:
            comparison_result: 对比分析结果
            user_data: 用户数据
            template_data: 模板数据
            
        Returns:
            LLM生成的建议文本
        """
        if not self.api_key:
            return "未配置API密钥，无法生成LLM增强建议"
        
        # 构建优化的提示词
        prompt = f"""
你是一位专业的羽毛球教练，请基于以下动作分析数据，为学员提供具体、可操作的训练建议。

【关键问题】:
{chr(10).join(comparison_result['critical_issues']) if comparison_result['critical_issues'] else '无严重问题'}

【各阶段表现】:
{self._format_stage_summary(comparison_result['stage_comparisons'])}

训练建议要求：
1. 针对性的技术纠正建议（具体到身体部位和角度）
2. 训练重点和练习方法
3. 循序渐进的改进计划
4. 注意事项和常见错误避免

要求：建议要具体、实用，避免泛泛而谈。请直接提供训练建议文本。
"""
        
        max_retries = 2  # 减少重试次数
        timeout_values = [60, 90]  # 更短的超时时间
        
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
                    "max_tokens": 3072 if attempt == 0 else 2048,  # 首次尝试更多token
                    "top_p": 0.9
                }
                
                current_timeout = timeout_values[attempt]
                status_msg = f"🌐 正在调用AI服务 (尝试 {attempt + 1}/{max_retries})\n   超时设置: {current_timeout}秒"
                self._send_status(status_msg)
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=current_timeout,
                    proxies={'http': None, 'https': None}
                )
                
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 直接返回文本内容
                return content.strip()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    self._send_status(f"⏰ 连接超时，正在重试...")
                    continue
                else:
                    self._send_status("⏰ 连接超时，切换到本地分析模式")
                    return self._generate_fallback_advice(comparison_result)
            except Exception as e:
                if attempt < max_retries - 1:
                    self._send_status(f"⚠️ 调用失败，正在重试...")
                    continue
                else:
                    self._send_status(f"❌ AI服务失败，切换到本地分析模式")
                    return self._generate_fallback_advice(comparison_result)
        
        return self._generate_fallback_advice(comparison_result)
    
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

【关键问题】:
{chr(10).join(comparison_result['critical_issues']) if comparison_result['critical_issues'] else '无严重问题'}

【各阶段表现】:
{self._format_stage_summary(comparison_result['stage_comparisons'])}

训练建议要求：
1. 针对性的技术纠正建议（具体到身体部位和角度）
2. 训练重点和练习方法
3. 循序渐进的改进计划
4. 注意事项和常见错误避免

要求：建议要具体、实用，避免泛泛而谈。请直接提供训练建议文本，不需要JSON格式。
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
                "max_tokens": 4096,  # 增加token限制以获得完整回复
                "top_p": 0.9,
                "stream": True  # 启用流式响应
            }
            
            self._send_status("📡 发送流式请求数据...")
            print(f"调试信息: 发送流式POST请求到 {self.api_url}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=120,  # 减少超时时间到2分钟，提高响应速度
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
            
            # 返回完整的建议文本
            return full_content.strip() if full_content.strip() else self._generate_fallback_advice(comparison_result)
            
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
        
        # 分析关键问题给出总体建议
        critical_issues = comparison_result.get('critical_issues', [])
        if not critical_issues:
            advice_lines.append("🎯 整体表现: 您的技术水平不错！")
            advice_lines.append("💡 建议重点: 继续保持现有水平，注重动作的一致性和稳定性。")
        elif len(critical_issues) <= 2:
            advice_lines.append("🎯 整体表现: 基础扎实，但仍有改进空间。")
            advice_lines.append("💡 建议重点: 重点改进标记的问题区域，循序渐进提升技术。")
        else:
            advice_lines.append("🎯 整体表现: 需要系统性的技术改进。")
            advice_lines.append("💡 建议重点: 从基础动作开始，建立正确的动作模式。")
        
        advice_lines.append("")
        
        # 分析各阶段问题并给出具体建议
        stage_comparisons = comparison_result.get('stage_comparisons', [])
        # 找出有问题的阶段（基于问题数量）
        problem_stages = []
        for stage in stage_comparisons:
            timing_issues = stage.get('timing_analysis', {}).get('issues', [])
            angle_issues = stage.get('angle_analysis', {}).get('issues', [])
            if len(timing_issues) + len(angle_issues) > 0:
                problem_stages.append(stage)
        
        if problem_stages:
            advice_lines.append("🔧 重点改进阶段:")
            for stage in problem_stages:
                stage_name = stage.get('stage_name', '未知阶段')
                timing_issues = stage.get('timing_analysis', {}).get('issues', [])
                angle_issues = stage.get('angle_analysis', {}).get('issues', [])
                total_issues = len(timing_issues) + len(angle_issues)
                advice_lines.append(f"\n📍 {stage_name} (发现 {total_issues} 个问题)")
                
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
        
        # 根据问题数量给出训练建议
        if len(critical_issues) > 3:
            advice_lines.append("1. 从基础挥拍动作开始，重点练习正确的握拍和站位")
            advice_lines.append("2. 分解练习各个击球阶段，确保每个动作都标准")
            advice_lines.append("3. 建议寻求专业教练指导，建立正确的动作模式")
        elif len(critical_issues) > 0:
            advice_lines.append("1. 重点改进标记的问题动作阶段")
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
            # 移除对score的引用
            issues = stage.get('issues', [])
            
            # 显示阶段名称和问题
            if issues:
                summary_lines.append(f"- {stage_name}: 发现 {len(issues)} 个问题")
                for issue in issues[:2]:  # 只显示前2个问题
                    summary_lines.append(f"  问题: {issue}")
            else:
                summary_lines.append(f"- {stage_name}: 动作良好")
        
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
        
        # 根据问题数量确定练习重点
        for stage in comparison_result['stage_comparisons']:
            stage_name = stage['stage_name']
            timing_issues = stage.get('timing_analysis', {}).get('issues', [])
            angle_issues = stage.get('angle_analysis', {}).get('issues', [])
            total_issues = len(timing_issues) + len(angle_issues)
            
            if total_issues >= 3:  # 问题较多
                practice_plan["immediate_focus"].append(f"重点练习{stage_name}阶段的基本动作")
            elif total_issues >= 1:  # 有少量问题
                practice_plan["short_term"].append(f"改进{stage_name}阶段的技术细节")
            else:  # 无明显问题
                practice_plan["long_term"].append(f"优化{stage_name}阶段的动作流畅性")
        
        return practice_plan
    

    
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
        
        # 移除整体评分显示
        
        # 各阶段详细分析
        lines.append("📋 各阶段分析详情")
        lines.append("-" * 25)
        
        stage_analysis = report.get('stage_analysis', [])
        for i, stage in enumerate(stage_analysis, 1):
            stage_name = stage.get('stage_name', '未知阶段')
            
            lines.append(f"{i}. 【{stage_name}】")
            
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
        critical_count = len(report.get('critical_issues', []))
        if critical_count == 0:
            lines.append("🌟 总结: 您的技术水平很不错，继续保持并精益求精！")
        elif critical_count <= 2:
            lines.append("💪 总结: 基础扎实，重点改进标记的问题，很快就能提升！")
        else:
            lines.append("🚀 总结: 还有很大提升空间，按照建议坚持练习，进步会很明显！")
        
        return "\n".join(lines)
    

    
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