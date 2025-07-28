#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线学习管理器
管理用户反馈收集和模型的持续改进
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import sqlite3
import numpy as np
from collections import defaultdict, deque

class OnlineLearningManager:
    """
    在线学习管理器
    负责收集用户反馈、管理训练数据、触发模型重训练
    """
    
    def __init__(self, data_dir: str = "learning_data", 
                 feedback_callback: Optional[Callable] = None):
        """
        初始化在线学习管理器
        
        Args:
            data_dir: 学习数据存储目录
            feedback_callback: 反馈处理回调函数
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据库文件路径
        self.db_path = self.data_dir / "learning_data.db"
        
        # 反馈处理回调
        self.feedback_callback = feedback_callback
        
        # 学习配置
        self.config = {
            'min_feedback_for_retrain': 50,      # 触发重训练的最小反馈数量
            'retrain_interval_hours': 24,        # 重训练间隔（小时）
            'max_feedback_buffer': 1000,         # 最大反馈缓存数量
            'feedback_weight_decay': 0.95,       # 反馈权重衰减因子
            'quality_threshold': 0.7,            # 数据质量阈值
            'auto_retrain': True,                # 是否自动重训练
            'backup_models': True                # 是否备份旧模型
        }
        
        # 反馈缓存
        self.feedback_buffer = deque(maxlen=self.config['max_feedback_buffer'])
        
        # 统计信息
        self.stats = {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'last_retrain_time': None,
            'retrain_count': 0,
            'data_quality_score': 0.0
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
        
        # 加载统计信息
        self._load_stats()
        
        print("在线学习管理器初始化完成")
    
    def collect_feedback(self, session_data: Dict[str, Any], 
                        user_feedback: Dict[str, Any]) -> bool:
        """
        收集用户反馈
        
        Args:
            session_data: 会话数据（包含姿态数据、预测结果等）
            user_feedback: 用户反馈（评分、纠正信息等）
            
        Returns:
            是否成功收集反馈
        """
        try:
            with self._lock:
                # 验证数据完整性
                if not self._validate_feedback_data(session_data, user_feedback):
                    return False
                
                # 创建反馈记录
                feedback_record = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_data.get('session_id', f"session_{int(time.time())}"),
                    'landmarks_data': session_data.get('landmarks', []),
                    'predicted_quality': session_data.get('predicted_quality', 50),
                    'predicted_error': session_data.get('predicted_error', 'normal'),
                    'user_quality_rating': user_feedback.get('quality_rating'),
                    'user_error_correction': user_feedback.get('error_correction'),
                    'user_satisfaction': user_feedback.get('satisfaction', 3),
                    'comments': user_feedback.get('comments', ''),
                    'data_quality': self._assess_data_quality(session_data),
                    'feedback_weight': 1.0
                }
                
                # 添加到缓存
                self.feedback_buffer.append(feedback_record)
                
                # 保存到数据库
                self._save_feedback_to_db(feedback_record)
                
                # 更新统计信息
                self._update_stats(feedback_record)
                
                # 检查是否需要触发重训练
                if self.config['auto_retrain']:
                    self._check_retrain_trigger()
                
                # 调用反馈回调
                if self.feedback_callback:
                    try:
                        self.feedback_callback(feedback_record)
                    except Exception as e:
                        print(f"反馈回调执行失败: {e}")
                
                print(f"反馈收集成功，当前缓存大小: {len(self.feedback_buffer)}")
                return True
                
        except Exception as e:
            print(f"反馈收集失败: {e}")
            return False
    
    def get_training_data(self, min_quality: float = 0.5, 
                         max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取用于训练的数据
        
        Args:
            min_quality: 最小数据质量要求
            max_samples: 最大样本数量
            
        Returns:
            训练数据列表
        """
        try:
            with self._lock:
                # 从数据库获取反馈数据
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = """
                SELECT landmarks_data, user_quality_rating, user_error_correction, 
                       data_quality, feedback_weight, timestamp
                FROM feedback 
                WHERE data_quality >= ? AND user_quality_rating IS NOT NULL
                ORDER BY timestamp DESC
                """
                
                if max_samples:
                    query += f" LIMIT {max_samples}"
                
                cursor.execute(query, (min_quality,))
                rows = cursor.fetchall()
                conn.close()
                
                training_data = []
                for row in rows:
                    landmarks_str, quality_rating, error_correction, data_quality, weight, timestamp = row
                    
                    try:
                        landmarks = json.loads(landmarks_str) if landmarks_str else []
                        
                        # 计算时间权重（越新的数据权重越高）
                        time_weight = self._calculate_time_weight(timestamp)
                        final_weight = weight * time_weight
                        
                        training_item = {
                            'landmarks': landmarks,
                            'quality_score': quality_rating,
                            'error_type': error_correction or 'normal',
                            'data_quality': data_quality,
                            'weight': final_weight,
                            'timestamp': timestamp
                        }
                        
                        training_data.append(training_item)
                        
                    except json.JSONDecodeError:
                        continue
                
                print(f"获取训练数据: {len(training_data)} 条")
                return training_data
                
        except Exception as e:
            print(f"获取训练数据失败: {e}")
            return []
    
    def trigger_retrain(self, force: bool = False) -> Dict[str, Any]:
        """
        触发模型重训练
        
        Args:
            force: 是否强制重训练
            
        Returns:
            重训练结果
        """
        try:
            with self._lock:
                # 检查重训练条件
                if not force and not self._should_retrain():
                    return {
                        'success': False,
                        'message': '不满足重训练条件',
                        'conditions': self._get_retrain_conditions()
                    }
                
                print("开始模型重训练...")
                
                # 获取训练数据
                training_data = self.get_training_data(
                    min_quality=self.config['quality_threshold']
                )
                
                if len(training_data) < self.config['min_feedback_for_retrain']:
                    return {
                        'success': False,
                        'message': f'训练数据不足，需要至少 {self.config["min_feedback_for_retrain"]} 条',
                        'current_data_count': len(training_data)
                    }
                
                # 数据预处理
                processed_data = self._preprocess_training_data(training_data)
                
                # 更新重训练统计
                self.stats['last_retrain_time'] = datetime.now().isoformat()
                self.stats['retrain_count'] += 1
                
                retrain_result = {
                    'success': True,
                    'training_data_count': len(processed_data),
                    'retrain_time': self.stats['last_retrain_time'],
                    'retrain_count': self.stats['retrain_count'],
                    'data_quality_avg': np.mean([d['data_quality'] for d in processed_data]),
                    'processed_data': processed_data
                }
                
                # 保存统计信息
                self._save_stats()
                
                print(f"重训练准备完成，数据量: {len(processed_data)}")
                return retrain_result
                
        except Exception as e:
            print(f"重训练触发失败: {e}")
            return {
                'success': False,
                'message': f'重训练失败: {str(e)}'
            }
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        获取反馈统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            # 从数据库获取最新统计
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总反馈数
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_count = cursor.fetchone()[0]
            
            # 最近24小时反馈数
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE timestamp > ?", (yesterday,))
            recent_count = cursor.fetchone()[0]
            
            # 平均数据质量
            cursor.execute("SELECT AVG(data_quality) FROM feedback WHERE data_quality IS NOT NULL")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # 用户满意度分布
            cursor.execute("""
                SELECT user_satisfaction, COUNT(*) 
                FROM feedback 
                WHERE user_satisfaction IS NOT NULL 
                GROUP BY user_satisfaction
            """)
            satisfaction_dist = dict(cursor.fetchall())
            
            conn.close()
            
            stats = {
                'total_feedback': total_count,
                'recent_24h_feedback': recent_count,
                'buffer_size': len(self.feedback_buffer),
                'avg_data_quality': round(avg_quality, 3),
                'satisfaction_distribution': satisfaction_dist,
                'last_retrain_time': self.stats['last_retrain_time'],
                'retrain_count': self.stats['retrain_count'],
                'retrain_conditions': self._get_retrain_conditions(),
                'config': self.config.copy()
            }
            
            return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            new_config: 新配置字典
            
        Returns:
            是否更新成功
        """
        try:
            with self._lock:
                # 验证配置
                valid_keys = set(self.config.keys())
                for key in new_config:
                    if key not in valid_keys:
                        print(f"无效的配置项: {key}")
                        return False
                
                # 更新配置
                self.config.update(new_config)
                
                # 调整缓存大小
                if 'max_feedback_buffer' in new_config:
                    new_maxlen = new_config['max_feedback_buffer']
                    if new_maxlen != self.feedback_buffer.maxlen:
                        # 重新创建缓存
                        old_data = list(self.feedback_buffer)
                        self.feedback_buffer = deque(old_data[-new_maxlen:], maxlen=new_maxlen)
                
                print("配置更新成功")
                return True
                
        except Exception as e:
            print(f"配置更新失败: {e}")
            return False
    
    def export_feedback_data(self, output_file: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> bool:
        """
        导出反馈数据
        
        Args:
            output_file: 输出文件路径
            start_date: 开始日期 (ISO格式)
            end_date: 结束日期 (ISO格式)
            
        Returns:
            是否导出成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM feedback"
            params = []
            
            if start_date or end_date:
                conditions = []
                if start_date:
                    conditions.append("timestamp >= ?")
                    params.append(start_date)
                if end_date:
                    conditions.append("timestamp <= ?")
                    params.append(end_date)
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # 获取列名
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # 转换为字典列表
            export_data = []
            for row in rows:
                record = dict(zip(column_names, row))
                # 解析JSON字段
                if record['landmarks_data']:
                    try:
                        record['landmarks_data'] = json.loads(record['landmarks_data'])
                    except json.JSONDecodeError:
                        record['landmarks_data'] = []
                export_data.append(record)
            
            # 保存到文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"反馈数据导出成功: {len(export_data)} 条记录 -> {output_file}")
            return True
            
        except Exception as e:
            print(f"数据导出失败: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
            
        Returns:
            删除的记录数
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 删除旧记录
            cursor.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"清理完成，删除了 {deleted_count} 条旧记录")
            return deleted_count
            
        except Exception as e:
            print(f"数据清理失败: {e}")
            return 0
    
    def _init_database(self):
        """
        初始化数据库
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建反馈表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                landmarks_data TEXT,
                predicted_quality REAL,
                predicted_error TEXT,
                user_quality_rating REAL,
                user_error_correction TEXT,
                user_satisfaction INTEGER,
                comments TEXT,
                data_quality REAL,
                feedback_weight REAL DEFAULT 1.0
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_quality ON feedback(data_quality)")
        
        conn.commit()
        conn.close()
    
    def _validate_feedback_data(self, session_data: Dict[str, Any], 
                               user_feedback: Dict[str, Any]) -> bool:
        """
        验证反馈数据的完整性
        
        Args:
            session_data: 会话数据
            user_feedback: 用户反馈
            
        Returns:
            是否有效
        """
        # 检查必要字段
        if not session_data.get('landmarks'):
            return False
        
        # 检查用户反馈的有效性
        quality_rating = user_feedback.get('quality_rating')
        if quality_rating is not None:
            if not (0 <= quality_rating <= 100):
                return False
        
        satisfaction = user_feedback.get('satisfaction', 3)
        if not (1 <= satisfaction <= 5):
            return False
        
        return True
    
    def _assess_data_quality(self, session_data: Dict[str, Any]) -> float:
        """
        评估数据质量
        
        Args:
            session_data: 会话数据
            
        Returns:
            数据质量分数 (0-1)
        """
        landmarks = session_data.get('landmarks', [])
        
        if not landmarks:
            return 0.0
        
        quality_score = 0.0
        factors = 0
        
        # 数据长度评估
        if len(landmarks) >= 10:  # 至少10帧
            quality_score += 0.3
        elif len(landmarks) >= 5:
            quality_score += 0.15
        factors += 1
        
        # 数据完整性评估
        valid_frames = 0
        for frame in landmarks:
            if isinstance(frame, dict) and 'landmarks' in frame:
                if len(frame['landmarks']) >= 20:  # 至少20个关键点
                    valid_frames += 1
        
        if landmarks:
            completeness = valid_frames / len(landmarks)
            quality_score += completeness * 0.4
        factors += 1
        
        # 时间戳连续性评估
        timestamps = []
        for frame in landmarks:
            if isinstance(frame, dict) and 'timestamp' in frame:
                timestamps.append(frame['timestamp'])
        
        if len(timestamps) > 1:
            time_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            if time_intervals:
                avg_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)
                if avg_interval > 0:
                    consistency = 1.0 - min(std_interval / avg_interval, 1.0)
                    quality_score += consistency * 0.3
        factors += 1
        
        return quality_score / factors if factors > 0 else 0.0
    
    def _calculate_time_weight(self, timestamp_str: str) -> float:
        """
        计算时间权重
        
        Args:
            timestamp_str: 时间戳字符串
            
        Returns:
            时间权重
        """
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            days_ago = (now - timestamp).days
            
            # 使用指数衰减
            weight = self.config['feedback_weight_decay'] ** days_ago
            return max(weight, 0.1)  # 最小权重0.1
            
        except:
            return 0.5  # 默认权重
    
    def _should_retrain(self) -> bool:
        """
        检查是否应该重训练
        
        Returns:
            是否应该重训练
        """
        # 检查反馈数量
        if len(self.feedback_buffer) < self.config['min_feedback_for_retrain']:
            return False
        
        # 检查时间间隔
        if self.stats['last_retrain_time']:
            try:
                last_retrain = datetime.fromisoformat(self.stats['last_retrain_time'])
                hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
                if hours_since < self.config['retrain_interval_hours']:
                    return False
            except:
                pass
        
        return True
    
    def _check_retrain_trigger(self):
        """
        检查并触发重训练
        """
        if self._should_retrain():
            print("满足重训练条件，准备触发重训练...")
            # 这里可以异步触发重训练
            # 实际应用中可能需要通过消息队列或其他机制
    
    def _get_retrain_conditions(self) -> Dict[str, Any]:
        """
        获取重训练条件状态
        
        Returns:
            条件状态字典
        """
        conditions = {
            'min_feedback_required': self.config['min_feedback_for_retrain'],
            'current_feedback_count': len(self.feedback_buffer),
            'feedback_sufficient': len(self.feedback_buffer) >= self.config['min_feedback_for_retrain'],
            'retrain_interval_hours': self.config['retrain_interval_hours'],
            'time_sufficient': True
        }
        
        if self.stats['last_retrain_time']:
            try:
                last_retrain = datetime.fromisoformat(self.stats['last_retrain_time'])
                hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
                conditions['hours_since_last_retrain'] = round(hours_since, 1)
                conditions['time_sufficient'] = hours_since >= self.config['retrain_interval_hours']
            except:
                conditions['hours_since_last_retrain'] = 'unknown'
        else:
            conditions['hours_since_last_retrain'] = 'never'
        
        conditions['should_retrain'] = (conditions['feedback_sufficient'] and 
                                      conditions['time_sufficient'])
        
        return conditions
    
    def _preprocess_training_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        预处理训练数据
        
        Args:
            training_data: 原始训练数据
            
        Returns:
            预处理后的训练数据
        """
        processed_data = []
        
        for item in training_data:
            # 数据清洗和验证
            if (item.get('landmarks') and 
                item.get('quality_score') is not None and
                item.get('data_quality', 0) >= self.config['quality_threshold']):
                
                processed_item = {
                    'landmarks': item['landmarks'],
                    'quality_score': item['quality_score'],
                    'error_type': item.get('error_type', 'normal'),
                    'weight': item.get('weight', 1.0)
                }
                
                processed_data.append(processed_item)
        
        return processed_data
    
    def _save_feedback_to_db(self, feedback_record: Dict[str, Any]):
        """
        保存反馈到数据库
        
        Args:
            feedback_record: 反馈记录
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (
                timestamp, session_id, landmarks_data, predicted_quality, predicted_error,
                user_quality_rating, user_error_correction, user_satisfaction, 
                comments, data_quality, feedback_weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_record['timestamp'],
            feedback_record['session_id'],
            json.dumps(feedback_record['landmarks_data']),
            feedback_record['predicted_quality'],
            feedback_record['predicted_error'],
            feedback_record['user_quality_rating'],
            feedback_record['user_error_correction'],
            feedback_record['user_satisfaction'],
            feedback_record['comments'],
            feedback_record['data_quality'],
            feedback_record['feedback_weight']
        ))
        
        conn.commit()
        conn.close()
    
    def _update_stats(self, feedback_record: Dict[str, Any]):
        """
        更新统计信息
        
        Args:
            feedback_record: 反馈记录
        """
        self.stats['total_feedback'] += 1
        
        satisfaction = feedback_record.get('user_satisfaction', 3)
        if satisfaction >= 4:
            self.stats['positive_feedback'] += 1
        elif satisfaction <= 2:
            self.stats['negative_feedback'] += 1
        
        # 更新数据质量分数
        data_quality = feedback_record.get('data_quality', 0)
        current_quality = self.stats['data_quality_score']
        # 使用移动平均
        alpha = 0.1
        self.stats['data_quality_score'] = (1 - alpha) * current_quality + alpha * data_quality
    
    def _load_stats(self):
        """
        加载统计信息
        """
        stats_file = self.data_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
                print("统计信息加载成功")
            except Exception as e:
                print(f"统计信息加载失败: {e}")
    
    def _save_stats(self):
        """
        保存统计信息
        """
        stats_file = self.data_dir / "stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"统计信息保存失败: {e}")

# 导出主要类
__all__ = ['OnlineLearningManager']