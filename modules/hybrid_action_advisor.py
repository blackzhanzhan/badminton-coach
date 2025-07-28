#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合动作顾问
整合传统规则分析和机器学习预测的增强版动作顾问
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# 导入原始模块
from .action_advisor import ActionAdvisor
from .enhanced_pose_analyzer import EnhancedPoseAnalyzer
from .action_quality_predictor import ActionQualityPredictor
from .online_learning_manager import OnlineLearningManager

class HybridActionAdvisor(ActionAdvisor):
    """
    混合动作顾问
    继承原始ActionAdvisor，增加机器学习能力
    """
    
    def __init__(self, templates_dir: str, staged_dir: str, 
                 model_dir: str = "models",
                 learning_data_dir: str = "learning_data",
                 enable_ml: bool = True,
                 ml_weight: float = 0.4,
                 rule_weight: float = 0.6,
                 callback_progress=None, callback_complete=None):
        """
        初始化混合动作顾问
        
        Args:
            templates_dir: 模板目录
            staged_dir: 分阶段数据目录
            model_dir: 机器学习模型目录
            learning_data_dir: 学习数据目录
            enable_ml: 是否启用机器学习
            ml_weight: 机器学习预测权重
            rule_weight: 规则分析权重
            callback_progress: 进度回调
            callback_complete: 完成回调
        """
        # 初始化父类
        super().__init__(templates_dir, staged_dir, callback_progress, callback_complete)
        
        # 机器学习配置
        self.enable_ml = enable_ml
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        # 确保权重和为1
        total_weight = ml_weight + rule_weight
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
        
        # 初始化增强组件
        self.enhanced_pose_analyzer = EnhancedPoseAnalyzer()
        
        if self.enable_ml:
            self.quality_predictor = ActionQualityPredictor(model_dir)
            self.learning_manager = OnlineLearningManager(
                learning_data_dir, 
                feedback_callback=self._handle_feedback
            )
        else:
            self.quality_predictor = None
            self.learning_manager = None
        
        # 分析配置
        self.analysis_config = {
            'use_enhanced_features': True,
            'enable_stage_detection': True,
            'enable_quality_prediction': self.enable_ml,
            'enable_error_prediction': self.enable_ml,
            'fusion_strategy': 'weighted_average',  # 'weighted_average', 'confidence_based', 'voting'
            'confidence_threshold': 0.7,
            'ml_fallback_to_rules': True
        }
        
        # 会话管理
        self.current_session = {
            'session_id': None,
            'start_time': None,
            'user_data': None,
            'template_data': None,
            'rule_analysis': None,
            'ml_prediction': None,
            'final_result': None
        }
        
        print(f"混合动作顾问初始化完成 (ML: {'启用' if enable_ml else '禁用'})")
    
    def analyze_action(self, user_file: str, template_name: str = None) -> Dict[str, Any]:
        """
        分析用户动作（增强版）
        
        Args:
            user_file: 用户数据文件路径
            template_name: 模板名称
            
        Returns:
            分析结果字典
        """
        try:
            # 开始新会话
            self._start_session(user_file, template_name)
            
            # 执行传统规则分析
            rule_result = self._perform_rule_analysis(user_file, template_name)
            
            # 执行机器学习分析
            ml_result = self._perform_ml_analysis(user_file) if self.enable_ml else None
            
            # 融合分析结果
            final_result = self._fuse_analysis_results(rule_result, ml_result)
            
            # 增强建议生成
            final_result = self._enhance_recommendations(final_result)
            
            # 保存会话结果
            self._save_session_result(final_result)
            
            return final_result
            
        except Exception as e:
            print(f"动作分析失败: {e}")
            return self._get_error_result(str(e))
    
    def collect_user_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        收集用户反馈
        
        Args:
            feedback_data: 反馈数据
            
        Returns:
            是否成功收集
        """
        if not self.enable_ml or not self.learning_manager:
            return False
        
        try:
            # 获取当前会话数据
            session_data = {
                'session_id': self.current_session['session_id'],
                'landmarks': self.current_session.get('user_data', []),
                'predicted_quality': self.current_session.get('ml_prediction', {}).get('quality_score', 50),
                'predicted_error': self.current_session.get('ml_prediction', {}).get('error_type', 'normal')
            }
            
            # 收集反馈
            success = self.learning_manager.collect_feedback(session_data, feedback_data)
            
            if success:
                print("用户反馈收集成功")
            
            return success
            
        except Exception as e:
            print(f"反馈收集失败: {e}")
            return False
    
    def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """
        重新训练机器学习模型
        
        Args:
            force: 是否强制重训练
            
        Returns:
            重训练结果
        """
        if not self.enable_ml or not self.learning_manager or not self.quality_predictor:
            return {'success': False, 'message': '机器学习未启用'}
        
        try:
            # 触发重训练
            retrain_result = self.learning_manager.trigger_retrain(force)
            
            if not retrain_result['success']:
                return retrain_result
            
            # 获取训练数据
            training_data = retrain_result['processed_data']
            
            # 训练模型
            training_result = self.quality_predictor.train_models(training_data)
            
            # 合并结果
            final_result = {
                'success': training_result['success'],
                'retrain_info': retrain_result,
                'training_info': training_result
            }
            
            if training_result['success']:
                print("模型重训练成功")
            else:
                print(f"模型重训练失败: {training_result.get('message', '未知错误')}")
            
            return final_result
            
        except Exception as e:
            print(f"模型重训练失败: {e}")
            return {'success': False, 'message': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        status = {
            'ml_enabled': self.enable_ml,
            'weights': {
                'ml_weight': self.ml_weight,
                'rule_weight': self.rule_weight
            },
            'analysis_config': self.analysis_config.copy(),
            'current_session': {
                'active': self.current_session['session_id'] is not None,
                'session_id': self.current_session['session_id']
            }
        }
        
        # 机器学习组件状态
        if self.enable_ml:
            if self.quality_predictor:
                status['ml_models'] = self.quality_predictor.get_model_info()
            
            if self.learning_manager:
                status['learning_stats'] = self.learning_manager.get_feedback_stats()
        
        return status
    
    def update_analysis_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新分析配置
        
        Args:
            new_config: 新配置
            
        Returns:
            是否更新成功
        """
        try:
            # 验证配置
            valid_keys = set(self.analysis_config.keys())
            for key in new_config:
                if key not in valid_keys:
                    print(f"无效的配置项: {key}")
                    return False
            
            # 更新配置
            self.analysis_config.update(new_config)
            
            # 更新权重
            if 'ml_weight' in new_config or 'rule_weight' in new_config:
                ml_weight = new_config.get('ml_weight', self.ml_weight)
                rule_weight = new_config.get('rule_weight', self.rule_weight)
                
                total_weight = ml_weight + rule_weight
                if total_weight > 0:
                    self.ml_weight = ml_weight / total_weight
                    self.rule_weight = rule_weight / total_weight
            
            print("分析配置更新成功")
            return True
            
        except Exception as e:
            print(f"配置更新失败: {e}")
            return False
    
    def _start_session(self, user_file: str, template_name: str):
        """
        开始新的分析会话
        
        Args:
            user_file: 用户文件
            template_name: 模板名称
        """
        self.current_session = {
            'session_id': f"session_{int(time.time())}_{os.path.basename(user_file)}",
            'start_time': datetime.now().isoformat(),
            'user_file': user_file,
            'template_name': template_name,
            'user_data': None,
            'template_data': None,
            'rule_analysis': None,
            'ml_prediction': None,
            'final_result': None
        }
    
    def _perform_rule_analysis(self, user_file: str, template_name: str) -> Dict[str, Any]:
        """
        执行传统规则分析
        
        Args:
            user_file: 用户文件
            template_name: 模板名称
            
        Returns:
            规则分析结果
        """
        try:
            # 调用父类的分析方法
            rule_result = super().analyze_action(user_file, template_name)
            
            # 保存到会话
            self.current_session['rule_analysis'] = rule_result
            
            # 加载用户数据用于后续ML分析
            if os.path.exists(user_file):
                with open(user_file, 'r', encoding='utf-8') as f:
                    self.current_session['user_data'] = json.load(f)
            
            return rule_result
            
        except Exception as e:
            print(f"规则分析失败: {e}")
            return self._get_error_result(f"规则分析失败: {str(e)}")
    
    def _perform_ml_analysis(self, user_file: str) -> Optional[Dict[str, Any]]:
        """
        执行机器学习分析
        
        Args:
            user_file: 用户文件
            
        Returns:
            ML分析结果
        """
        if not self.quality_predictor:
            return None
        
        try:
            # 获取用户数据
            user_data = self.current_session.get('user_data')
            if not user_data:
                return None
            
            # 提取姿态时间线
            landmarks_timeline = []
            
            # 处理不同的数据格式
            if isinstance(user_data, list):
                landmarks_timeline = user_data
            elif isinstance(user_data, dict):
                # 如果是分阶段数据
                for stage_name, stage_data in user_data.items():
                    if isinstance(stage_data, dict) and 'key_landmarks' in stage_data:
                        landmarks_timeline.extend(stage_data['key_landmarks'])
                    elif isinstance(stage_data, list):
                        landmarks_timeline.extend(stage_data)
            
            if not landmarks_timeline:
                return None
            
            # 执行质量预测
            ml_result = self.quality_predictor.predict_quality(landmarks_timeline)
            
            # 保存到会话
            self.current_session['ml_prediction'] = ml_result
            
            return ml_result
            
        except Exception as e:
            print(f"ML分析失败: {e}")
            return None
    
    def _fuse_analysis_results(self, rule_result: Dict[str, Any], 
                              ml_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        融合规则分析和ML预测结果
        
        Args:
            rule_result: 规则分析结果
            ml_result: ML预测结果
            
        Returns:
            融合后的结果
        """
        # 如果没有ML结果，直接返回规则结果
        if not ml_result or not self.enable_ml:
            fused_result = rule_result.copy()
            fused_result['analysis_method'] = 'rule_based_only'
            return fused_result
        
        # 开始融合
        fused_result = rule_result.copy()
        fused_result['analysis_method'] = 'hybrid'
        fused_result['ml_prediction'] = ml_result
        
        try:
            # 融合质量分数
            rule_score = self._extract_rule_quality_score(rule_result)
            ml_score = ml_result.get('quality_score', 50)
            
            if self.analysis_config['fusion_strategy'] == 'weighted_average':
                fused_score = (rule_score * self.rule_weight + 
                             ml_score * self.ml_weight)
            elif self.analysis_config['fusion_strategy'] == 'confidence_based':
                ml_confidence = ml_result.get('confidence', 0.5)
                if ml_confidence >= self.analysis_config['confidence_threshold']:
                    fused_score = ml_score
                else:
                    fused_score = rule_score
            else:  # voting or other strategies
                fused_score = (rule_score + ml_score) / 2
            
            fused_result['fused_quality_score'] = round(fused_score, 1)
            fused_result['quality_level'] = self._score_to_level(fused_score)
            
            # 融合错误类型和建议
            fused_result = self._fuse_recommendations(fused_result, rule_result, ml_result)
            
            # 添加置信度信息
            fused_result['confidence_info'] = {
                'rule_weight': self.rule_weight,
                'ml_weight': self.ml_weight,
                'ml_confidence': ml_result.get('confidence', 0.5),
                'fusion_strategy': self.analysis_config['fusion_strategy']
            }
            
        except Exception as e:
            print(f"结果融合失败: {e}")
            # 融合失败时回退到规则结果
            fused_result = rule_result.copy()
            fused_result['analysis_method'] = 'rule_based_fallback'
            fused_result['fusion_error'] = str(e)
        
        return fused_result
    
    def _extract_rule_quality_score(self, rule_result: Dict[str, Any]) -> float:
        """
        从规则分析结果中提取质量分数
        
        Args:
            rule_result: 规则分析结果
            
        Returns:
            质量分数
        """
        # 基于关键问题数量计算分数
        critical_issues = rule_result.get('critical_issues_summary', {}).get('total_critical', 0)
        total_stages = len(rule_result.get('stage_analysis', {}))
        
        if total_stages == 0:
            return 50  # 默认分数
        
        # 简单的评分逻辑
        issue_ratio = critical_issues / total_stages
        
        if issue_ratio == 0:
            base_score = 85
        elif issue_ratio <= 0.2:
            base_score = 75
        elif issue_ratio <= 0.4:
            base_score = 65
        elif issue_ratio <= 0.6:
            base_score = 55
        else:
            base_score = 45
        
        # 根据具体问题类型调整
        timing_issues = sum(1 for stage in rule_result.get('stage_analysis', {}).values() 
                          if stage.get('timing_analysis', {}).get('has_issues', False))
        angle_issues = sum(1 for stage in rule_result.get('stage_analysis', {}).values() 
                         if stage.get('angle_analysis', {}).get('has_issues', False))
        
        # 时机问题影响较大
        score_adjustment = -(timing_issues * 5 + angle_issues * 3)
        
        final_score = max(base_score + score_adjustment, 0)
        return min(final_score, 100)
    
    def _fuse_recommendations(self, fused_result: Dict[str, Any], 
                            rule_result: Dict[str, Any], 
                            ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        融合建议信息
        
        Args:
            fused_result: 融合结果
            rule_result: 规则结果
            ml_result: ML结果
            
        Returns:
            更新后的融合结果
        """
        # 合并建议
        rule_suggestions = rule_result.get('detailed_suggestions', [])
        ml_recommendations = ml_result.get('recommendations', [])
        
        # 去重并合并建议
        all_suggestions = list(rule_suggestions)
        for ml_rec in ml_recommendations:
            if ml_rec not in all_suggestions:
                all_suggestions.append(f"[ML建议] {ml_rec}")
        
        fused_result['fused_suggestions'] = all_suggestions
        
        # 错误类型融合
        ml_error_type = ml_result.get('error_type', 'normal')
        if ml_error_type != 'normal':
            fused_result['ml_detected_error'] = ml_error_type
            fused_result['ml_error_probability'] = ml_result.get('error_probability', 0.5)
        
        # 详细错误概率
        if 'detailed_scores' in ml_result:
            fused_result['ml_error_details'] = ml_result['detailed_scores']
        
        return fused_result
    
    def _enhance_recommendations(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强建议生成
        
        Args:
            result: 分析结果
            
        Returns:
            增强后的结果
        """
        try:
            # 生成个性化练习计划
            practice_plan = self._generate_practice_plan(result)
            result['personalized_practice_plan'] = practice_plan
            
            # 生成进步追踪建议
            progress_tracking = self._generate_progress_tracking(result)
            result['progress_tracking'] = progress_tracking
            
            # 生成技术要点总结
            technical_summary = self._generate_technical_summary(result)
            result['technical_summary'] = technical_summary
            
        except Exception as e:
            print(f"建议增强失败: {e}")
        
        return result
    
    def _generate_practice_plan(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成个性化练习计划
        
        Args:
            result: 分析结果
            
        Returns:
            练习计划
        """
        quality_score = result.get('fused_quality_score', 
                                 result.get('quality_score', 50))
        
        plan = {
            'difficulty_level': 'beginner',
            'focus_areas': [],
            'recommended_exercises': [],
            'practice_duration': '15-20分钟',
            'frequency': '每日练习'
        }
        
        # 根据质量分数确定难度
        if quality_score >= 80:
            plan['difficulty_level'] = 'advanced'
            plan['practice_duration'] = '10-15分钟'
            plan['frequency'] = '隔日练习'
        elif quality_score >= 60:
            plan['difficulty_level'] = 'intermediate'
            plan['practice_duration'] = '15-20分钟'
            plan['frequency'] = '每日练习'
        else:
            plan['difficulty_level'] = 'beginner'
            plan['practice_duration'] = '20-30分钟'
            plan['frequency'] = '每日练习'
        
        # 根据问题类型确定重点领域
        if 'ml_detected_error' in result:
            error_type = result['ml_detected_error']
            if 'timing' in error_type:
                plan['focus_areas'].append('时机控制')
                plan['recommended_exercises'].append('节拍器配合练习')
            if 'angle' in error_type:
                plan['focus_areas'].append('动作角度')
                plan['recommended_exercises'].append('镜前慢动作练习')
            if 'speed' in error_type:
                plan['focus_areas'].append('动作速度')
                plan['recommended_exercises'].append('渐进式速度训练')
        
        # 基于规则分析添加重点
        critical_issues = result.get('critical_issues_summary', {})
        if critical_issues.get('timing_issues', 0) > 0:
            if '时机控制' not in plan['focus_areas']:
                plan['focus_areas'].append('时机控制')
        if critical_issues.get('angle_issues', 0) > 0:
            if '动作角度' not in plan['focus_areas']:
                plan['focus_areas'].append('动作角度')
        
        # 默认练习
        if not plan['recommended_exercises']:
            plan['recommended_exercises'] = [
                '基础动作分解练习',
                '慢动作标准化训练',
                '视频对比分析'
            ]
        
        return plan
    
    def _generate_progress_tracking(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成进步追踪建议
        
        Args:
            result: 分析结果
            
        Returns:
            进步追踪信息
        """
        tracking = {
            'key_metrics': [],
            'improvement_targets': [],
            'tracking_frequency': '每周',
            'success_criteria': []
        }
        
        quality_score = result.get('fused_quality_score', 
                                 result.get('quality_score', 50))
        
        # 关键指标
        tracking['key_metrics'] = [
            '整体动作质量分数',
            '关键角度准确性',
            '时机控制精度'
        ]
        
        # 改进目标
        if quality_score < 60:
            tracking['improvement_targets'] = [
                f'质量分数提升至65分以上',
                '减少关键错误至2个以下',
                '提高动作一致性'
            ]
        elif quality_score < 80:
            tracking['improvement_targets'] = [
                f'质量分数提升至85分以上',
                '优化动作细节',
                '提高动作流畅度'
            ]
        else:
            tracking['improvement_targets'] = [
                '保持当前水平',
                '探索高级技巧',
                '提升动作美感'
            ]
        
        # 成功标准
        tracking['success_criteria'] = [
            '连续3次分析质量分数达标',
            '关键错误数量持续减少',
            '用户主观感受改善'
        ]
        
        return tracking
    
    def _generate_technical_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成技术要点总结
        
        Args:
            result: 分析结果
            
        Returns:
            技术要点总结
        """
        summary = {
            'strengths': [],
            'weaknesses': [],
            'key_improvements': [],
            'technical_notes': []
        }
        
        quality_score = result.get('fused_quality_score', 
                                 result.get('quality_score', 50))
        
        # 优势分析
        if quality_score >= 70:
            summary['strengths'].append('整体动作质量良好')
        
        critical_issues = result.get('critical_issues_summary', {})
        if critical_issues.get('total_critical', 0) <= 1:
            summary['strengths'].append('关键错误较少')
        
        # 弱点分析
        if 'ml_detected_error' in result:
            error_type = result['ml_detected_error']
            error_mapping = {
                'timing_early': '击球时机偏早',
                'timing_late': '击球时机偏晚',
                'angle_elbow': '肘部角度需调整',
                'angle_shoulder': '肩部姿态需改进',
                'speed_fast': '动作速度过快',
                'speed_slow': '动作速度偏慢'
            }
            if error_type in error_mapping:
                summary['weaknesses'].append(error_mapping[error_type])
        
        # 关键改进点
        if result.get('fused_suggestions'):
            summary['key_improvements'] = result['fused_suggestions'][:3]
        
        # 技术注释
        if result.get('analysis_method') == 'hybrid':
            summary['technical_notes'].append('本次分析结合了规则分析和机器学习预测')
        
        ml_confidence = result.get('confidence_info', {}).get('ml_confidence', 0)
        if ml_confidence >= 0.8:
            summary['technical_notes'].append('机器学习预测置信度较高')
        elif ml_confidence <= 0.5:
            summary['technical_notes'].append('建议增加训练数据以提高预测准确性')
        
        return summary
    
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
    
    def _save_session_result(self, result: Dict[str, Any]):
        """
        保存会话结果
        
        Args:
            result: 分析结果
        """
        self.current_session['final_result'] = result
        self.current_session['end_time'] = datetime.now().isoformat()
    
    def _handle_feedback(self, feedback_record: Dict[str, Any]):
        """
        处理反馈回调
        
        Args:
            feedback_record: 反馈记录
        """
        print(f"收到用户反馈: 满意度 {feedback_record.get('user_satisfaction', 'N/A')}")
        
        # 这里可以添加更多的反馈处理逻辑
        # 例如：实时调整模型参数、更新推荐策略等
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        获取错误结果
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误结果字典
        """
        return {
            'success': False,
            'error': error_message,
            'analysis_time': datetime.now().isoformat(),
            'quality_score': 0,
            'quality_level': '分析失败',
            'suggestions': ['请检查输入数据格式', '确保文件路径正确', '联系技术支持'],
            'analysis_method': 'error'
        }

# 导出主要类
__all__ = ['HybridActionAdvisor']