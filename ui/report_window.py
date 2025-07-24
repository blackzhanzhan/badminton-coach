#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QTextEdit, QScrollArea, QWidget, QPushButton,
                            QTabWidget, QTableWidget, QTableWidgetItem,
                            QProgressBar, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QColor, QPen
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from modules.pose_analyzer import PoseAnalyzer

class AnalysisWorker(QThread):
    """后台分析线程"""
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, standard_json_path, learner_json_path):
        super().__init__()
        self.standard_json_path = standard_json_path
        self.learner_json_path = learner_json_path
        self.analyzer = PoseAnalyzer()
    
    def run(self):
        try:
            self.progress_updated.emit(20)
            
            # 执行JSON差异分析
            suggestions = self.analyzer.analyze_json_difference(
                self.standard_json_path, self.learner_json_path
            )
            
            self.progress_updated.emit(60)
            
            # 生成分析报告数据
            report_data = {
                'suggestions': suggestions,
                'standard_path': self.standard_json_path,
                'learner_path': self.learner_json_path,
                'analysis_type': 'JSON差异分析',
                'timestamp': self._get_timestamp()
            }
            
            self.progress_updated.emit(100)
            self.analysis_completed.emit(report_data)
            
        except Exception as e:
            self.error_occurred.emit(f"分析过程中发生错误: {str(e)}")
    
    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ReportWindow(QDialog):
    """分析报告窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("羽毛球动作分析报告")
        self.setGeometry(200, 200, 1000, 700)
        self.setModal(False)  # 允许同时操作主窗口
        
        self.report_data = None
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("羽毛球动作分析报告")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        
        # 分析结果选项卡
        self.analysis_tab = self.create_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "分析结果")
        
        # 详细数据选项卡
        self.data_tab = self.create_data_tab()
        self.tab_widget.addTab(self.data_tab, "详细数据")
        
        # 可视化选项卡
        self.chart_tab = self.create_chart_tab()
        self.tab_widget.addTab(self.chart_tab, "数据可视化")
        
        layout.addWidget(self.tab_widget)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("导出报告")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addStretch()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def create_analysis_tab(self):
        """创建分析结果选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 分析状态
        self.status_frame = QFrame()
        self.status_frame.setFrameStyle(QFrame.Box)
        status_layout = QVBoxLayout(self.status_frame)
        
        self.status_label = QLabel("等待分析...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.status_frame)
        
        # 分析建议
        suggestions_label = QLabel("分析建议:")
        suggestions_font = QFont()
        suggestions_font.setBold(True)
        suggestions_label.setFont(suggestions_font)
        
        self.suggestions_text = QTextEdit()
        self.suggestions_text.setReadOnly(True)
        self.suggestions_text.setPlainText("暂无分析结果")
        
        layout.addWidget(suggestions_label)
        layout.addWidget(self.suggestions_text)
        
        return widget
    
    def create_data_tab(self):
        """创建详细数据选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 数据表格
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(3)
        self.data_table.setHorizontalHeaderLabels(["项目", "标准值", "用户值"])
        
        layout.addWidget(QLabel("详细对比数据:"))
        layout.addWidget(self.data_table)
        
        return widget
    
    def create_chart_tab(self):
        """创建数据可视化选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 创建matplotlib图表
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout.addWidget(QLabel("动作轨迹对比:"))
        layout.addWidget(self.canvas)
        
        return widget
    
    def start_analysis(self, standard_json_path, learner_json_path):
        """开始分析"""
        self.status_label.setText("正在分析中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动分析线程
        self.analysis_worker = AnalysisWorker(standard_json_path, learner_json_path)
        self.analysis_worker.progress_updated.connect(self.update_progress)
        self.analysis_worker.analysis_completed.connect(self.display_results)
        self.analysis_worker.error_occurred.connect(self.display_error)
        self.analysis_worker.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def display_results(self, report_data):
        """显示分析结果"""
        self.report_data = report_data
        
        # 更新状态
        self.status_label.setText(f"分析完成 - {report_data['timestamp']}")
        self.progress_bar.setVisible(False)
        
        # 显示建议
        suggestions_text = "\n\n".join(report_data['suggestions'])
        self.suggestions_text.setPlainText(suggestions_text)
        
        # 更新数据表格
        self.update_data_table(report_data)
        
        # 更新图表
        self.update_chart(report_data)
        
        # 启用导出按钮
        self.export_button.setEnabled(True)
    
    def display_error(self, error_message):
        """显示错误信息"""
        self.status_label.setText(f"分析失败: {error_message}")
        self.progress_bar.setVisible(False)
        self.suggestions_text.setPlainText(f"错误: {error_message}")
    
    def update_data_table(self, report_data):
        """更新数据表格"""
        # 示例数据，实际应该从分析结果中提取
        sample_data = [
            ("分析类型", report_data['analysis_type'], "-"),
            ("标准模板", report_data['standard_path'].split('/')[-1], "-"),
            ("用户数据", report_data['learner_path'].split('/')[-1], "-"),
            ("建议数量", str(len(report_data['suggestions'])), "-")
        ]
        
        self.data_table.setRowCount(len(sample_data))
        
        for row, (item, standard, user) in enumerate(sample_data):
            self.data_table.setItem(row, 0, QTableWidgetItem(item))
            self.data_table.setItem(row, 1, QTableWidgetItem(standard))
            self.data_table.setItem(row, 2, QTableWidgetItem(user))
        
        self.data_table.resizeColumnsToContents()
    
    def update_chart(self, report_data):
        """更新图表"""
        self.figure.clear()
        
        # 创建示例图表
        ax = self.figure.add_subplot(111)
        
        # 示例数据 - 实际应该从分析结果中提取
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)  # 标准动作轨迹
        y2 = np.sin(x + 0.5) + 0.1 * np.random.randn(100)  # 用户动作轨迹
        
        ax.plot(x, y1, 'b-', label='标准动作', linewidth=2)
        ax.plot(x, y2, 'r--', label='用户动作', linewidth=2)
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('角度 (度)')
        ax.set_title('动作轨迹对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def export_report(self):
        """导出报告"""
        if not self.report_data:
            return
        
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析报告", "羽毛球分析报告.txt", "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("羽毛球动作分析报告\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"分析时间: {self.report_data['timestamp']}\n")
                    f.write(f"分析类型: {self.report_data['analysis_type']}\n")
                    f.write(f"标准模板: {self.report_data['standard_path']}\n")
                    f.write(f"用户数据: {self.report_data['learner_path']}\n\n")
                    
                    f.write("分析建议:\n")
                    f.write("-" * 30 + "\n")
                    for i, suggestion in enumerate(self.report_data['suggestions'], 1):
                        f.write(f"{i}. {suggestion}\n\n")
                
                QMessageBox.information(self, "导出成功", f"报告已保存到: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存文件时发生错误: {str(e)}")