import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

from modules.pose_detector import PoseDetector
from modules.badminton_analyzer import BadmintonAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("羽毛球发球动作纠正系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化组件
        self.init_ui()
        
        # 初始化姿势检测器和分析器
        self.pose_detector = PoseDetector()
        self.badminton_analyzer = BadmintonAnalyzer()
        
        # 视频相关变量
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera = True
        self.camera_id = 0
        
    def init_ui(self):
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 上部分：视频显示和分析结果
        top_layout = QHBoxLayout()
        
        # 左侧：视频显示
        self.video_label = QLabel("等待视频输入...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid gray;")
        top_layout.addWidget(self.video_label, 2)
        
        # 右侧：分析结果
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        self.result_label = QLabel("动作分析结果")
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        
        self.feedback_label = QLabel("等待分析...")
        self.feedback_label.setAlignment(Qt.AlignTop)
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("background-color: #f0f0f0; padding: 10px;")
        self.feedback_label.setMinimumHeight(400)
        
        analysis_layout.addWidget(self.result_label)
        analysis_layout.addWidget(self.feedback_label)
        
        top_layout.addWidget(analysis_widget, 1)
        
        # 下部分：控制按钮
        bottom_layout = QHBoxLayout()
        
        # 输入源选择
        self.source_combo = QComboBox()
        self.source_combo.addItems(["摄像头", "视频文件"])
        self.source_combo.currentIndexChanged.connect(self.change_source)
        
        # 控制按钮
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.toggle_detection)
        self.start_button.setMinimumWidth(100)
        
        self.file_button = QPushButton("选择文件")
        self.file_button.clicked.connect(self.open_file)
        self.file_button.setEnabled(False)
        self.file_button.setMinimumWidth(100)
        
        bottom_layout.addWidget(self.source_combo)
        bottom_layout.addWidget(self.file_button)
        bottom_layout.addWidget(self.start_button)
        
        # 添加到主布局
        main_layout.addLayout(top_layout, 4)
        main_layout.addLayout(bottom_layout, 1)
        
    def change_source(self, index):
        self.is_camera = (index == 0)
        self.file_button.setEnabled(not self.is_camera)
        if self.cap and self.cap.isOpened():
            self.stop_detection()
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path
            self.start_detection()
    
    def toggle_detection(self):
        if self.timer.isActive():
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        if self.is_camera:
            self.cap = cv2.VideoCapture(self.camera_id)
        else:
            if hasattr(self, 'video_path'):
                self.cap = cv2.VideoCapture(self.video_path)
            else:
                return
        
        if not self.cap.isOpened():
            self.feedback_label.setText("无法打开视频源")
            return
        
        self.timer.start(30)  # 约30fps
        self.start_button.setText("停止")
    
    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.start_button.setText("开始")
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            if not self.is_camera:
                self.feedback_label.setText("视频播放完毕")
            return
        
        # 姿势检测
        frame, landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks:
            # 分析发球动作
            feedback = self.badminton_analyzer.analyze_serve(landmarks)
            self.feedback_label.setText(feedback)
        
        # 显示图像
        h, w, c = frame.shape
        q_img = QImage(frame.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), 
            Qt.KeepAspectRatio))
    
    def closeEvent(self, event):
        self.stop_detection()
        event.accept() 