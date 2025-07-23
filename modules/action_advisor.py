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

class ActionAdvisor:
    """
    动作建议智能体：对比用户动作数据与标准模板，生成具体纠正建议
    """
    
    def __init__(self, templates_dir="d:\\羽毛球项目\\templates", 
                 staged_dir="d:\\羽毛球项目\\staged_templates"):
        """
        初始化动作建议智能体
        
        Args:
            templates_dir: 标准模板文件目录
            staged_dir: 用户staged JSON文件目录
        """
        self.templates_dir = templates_dir
        self.staged_dir = staged_dir
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.api_key = os.environ.get('VOLCENGINE_API_KEY', '')
        
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
    
    def get_latest_staged_file(self) -> str:
        """
        获取最新的staged JSON文件
        
        Returns:
            最新staged文件的完整路径
        """
        staged_files = list(Path(self.staged_dir).glob("staged_*.json"))
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
        
        # 分析每个关键角度
        for angle_name in template_angles.keys():
            if angle_name not in user_angles:
                penalty += 15
                issues.append(f"缺少关键角度数据：{angle_name}")
                continue
            
            user_angle = user_angles[angle_name]
            template_angle = template_angles[angle_name]
            
            # 获取理想值进行比较
            user_ideal = user_angle.get("ideal", 0)
            template_ideal = template_angle.get("ideal", 0)
            
            angle_diff = abs(user_ideal - template_ideal)
            
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
                issues.append(f"{angle_name}角度偏差严重：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["moderate"]:
                penalty += 15  # 增加扣分
                angle_detail["severity"] = "moderate"
                issues.append(f"{angle_name}角度偏差较大：{angle_diff:.1f}°")
            elif angle_diff > self.angle_thresholds["minor"]:
                penalty += 8   # 增加扣分
                angle_detail["severity"] = "minor"
                issues.append(f"{angle_name}角度略有偏差：{angle_diff:.1f}°")
            
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
                
                if "肘" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{angle_name}过大({user_val:.1f}°)，建议收紧手臂至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{angle_name}过小({user_val:.1f}°)，建议展开手臂至{template_val:.1f}°")
                elif "肩" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{angle_name}过大({user_val:.1f}°)，建议降低肩膀至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{angle_name}过小({user_val:.1f}°)，建议抬高肩膀至{template_val:.1f}°")
                elif "髋" in angle_name:
                    if user_val > template_val:
                        suggestions.append(f"【{stage_name}】{angle_name}过大({user_val:.1f}°)，建议收紧腰部至{template_val:.1f}°")
                    else:
                        suggestions.append(f"【{stage_name}】{angle_name}过小({user_val:.1f}°)，建议放松腰部至{template_val:.1f}°")
        
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
            
            # 生成LLM增强建议
            llm_advice = self._generate_llm_advice(comparison_result, user_data, template_data)
            
            # 构建最终报告
            final_report = {
                "analysis_timestamp": self._get_timestamp(),
                "user_file": os.path.basename(user_file_path),
                "template_file": os.path.basename(template_file_path),
                "overall_score": comparison_result["overall_score"],
                "performance_level": self._get_performance_level(comparison_result["overall_score"]),
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
                "top_p": 0.9
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30,
                proxies={'http': None, 'https': None}
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"LLM建议生成失败: {str(e)}"
    
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