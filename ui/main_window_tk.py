import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import numpy as np
from modules.pose_detector import PoseDetector
from modules.pose_analyzer import PoseAnalyzer
from modules.badminton_analyzer import BadmintonAnalyzer

class MainWindow:
    def __init__(self, root):
        """初始化主窗口"""
        self.root = root
        self.root.title("羽毛球接球姿态分析与教练系统")
        self.root.geometry("1200x800")
        
        # 将检测器和分析器的初始化推迟到 start_detection
        self.pose_detector = None
        self.pose_analyzer = None
        
        # 视频相关变量
        self.cap = None
        self.is_camera = True
        self.camera_id = 0
        self.is_running = False
        
        # 调试模式
        self.debug_mode = False
        
        # 线程相关
        self.thread = None
        
        # 初始化UI
        self.init_ui()
        
        # 新增变量用于视频文件分析
        self.all_landmarks_timeline = []
        self.processed_frames = 0
        self.total_frames = 0
        self.fps = 30  # 默认 FPS
        
    def init_ui(self):
        """初始化用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上部分：视频显示和分析结果
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 左侧：视频显示
        self.video_frame = ttk.LabelFrame(top_frame, text="视频显示")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = ttk.Label(self.video_frame, text="请选择计算设备和输入源后，点击开始")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧：分析结果
        analysis_frame = ttk.LabelFrame(top_frame, text="动作分析与建议")
        analysis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5)
        analysis_frame.config(width=400)

        self.feedback_text = tk.Text(analysis_frame, wrap=tk.WORD, height=20, width=45)
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.feedback_text.insert(tk.END, "等待开始...")
        
        # 下部分：控制按钮
        self.bottom_frame = ttk.Frame(main_frame)
        self.bottom_frame.pack(fill=tk.X, pady=5)
        
        # 计算设备选择
        device_label = ttk.Label(self.bottom_frame, text="计算设备:")
        device_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.device_var = tk.StringVar(value="CPU")
        self.device_combo = ttk.Combobox(self.bottom_frame, textvariable=self.device_var, 
                                   values=["CPU", "GPU"], width=8, state="readonly")
        self.device_combo.pack(side=tk.LEFT, padx=5)
        
        # 分隔符
        ttk.Separator(self.bottom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill='y', padx=10)

        # 分析模式选择
        analysis_mode_label = ttk.Label(self.bottom_frame, text="分析模式:")
        analysis_mode_label.pack(side=tk.LEFT, padx=5)

        self.analysis_mode_var = tk.StringVar(value=PoseAnalyzer.AnalysisMode.STATIC_READY_STANCE.value)
        analysis_modes = ["静态-准备姿势", "动态-接球分析"]
        self.analysis_mode_combo = ttk.Combobox(self.bottom_frame, textvariable=self.analysis_mode_var, 
                                           values=analysis_modes, width=15, state="readonly")
        self.analysis_mode_combo.pack(side=tk.LEFT, padx=5)

        # 分隔符
        ttk.Separator(self.bottom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill='y', padx=10)

        # 输入源选择
        source_label = ttk.Label(self.bottom_frame, text="输入源:")
        source_label.pack(side=tk.LEFT, padx=5)
        
        self.source_var = tk.StringVar(value="摄像头")
        self.source_combo = ttk.Combobox(self.bottom_frame, textvariable=self.source_var, 
                                   values=["摄像头", "视频文件"], width=10, state="readonly")
        self.source_combo.pack(side=tk.LEFT, padx=5)
        self.source_combo.bind("<<ComboboxSelected>>", self.change_source)
        
        # 文件选择按钮
        self.file_button = ttk.Button(self.bottom_frame, text="选择文件", command=self.open_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        self.file_button["state"] = "disabled"
        
        # 开始/停止按钮
        self.start_button = ttk.Button(self.bottom_frame, text="开始", command=self.toggle_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 调试模式切换按钮
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(self.bottom_frame, text="调试模式", 
                                     variable=self.debug_var, 
                                     command=self.toggle_debug)
        self.debug_check.pack(side=tk.LEFT, padx=5)
        
    def disable_controls(self):
        """禁用除停止按钮外的所有控制控件"""
        for widget in self.bottom_frame.winfo_children():
            if isinstance(widget, (ttk.Combobox, ttk.Button, ttk.Checkbutton)):
                if widget != self.start_button:
                    widget.config(state="disabled")

    def enable_controls(self):
        """启用所有控制控件"""
        for widget in self.bottom_frame.winfo_children():
            if isinstance(widget, (ttk.Combobox, ttk.Button, ttk.Checkbutton)):
                widget.config(state="normal")
        # 根据输入源重新设置文件按钮的状态
        self.file_button["state"] = "disabled" if self.is_camera else "normal"
        
    def toggle_debug(self):
        """切换调试模式"""
        self.debug_mode = self.debug_var.get()
        
    def change_source(self, event):
        """切换输入源"""
        self.is_camera = (self.source_var.get() == "摄像头")
        self.file_button["state"] = "disabled" if self.is_camera else "normal"
        
        if self.is_running:
            self.stop_detection()
    
    def open_file(self):
        """打开视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.video_path = file_path
            self.start_detection()
    
    def toggle_detection(self):
        """切换检测状态"""
        if self.is_running:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """开始检测"""
        # --- 步骤1: 根据用户选择，在每次开始时重新初始化检测器 ---
        selected_device = self.device_var.get().lower()
        self.update_feedback_box(f"正在使用 {selected_device.upper()} 初始化模型，请稍候...")
        self.root.update_idletasks()

        try:
            self.pose_detector = PoseDetector(device=selected_device)
            # 根据UI选择的模式来初始化分析器
            selected_mode_str = self.analysis_mode_var.get()
            selected_mode = next((mode for mode in PoseAnalyzer.AnalysisMode if mode.value == selected_mode_str), 
                                 PoseAnalyzer.AnalysisMode.STATIC_READY_STANCE)
            
            # 使用选择的模式和检测器提供的关键点信息来初始化分析器
            self.pose_analyzer = PoseAnalyzer(self.pose_detector.get_landmarks_info(), analysis_mode=selected_mode)

        except Exception as e:
            self.update_feedback_box(f"创建检测器或分析器时发生未知错误: {e}")
            return

        # --- 步骤2: 检查初始化是否成功 (特别是GPU模式) ---
        if self.pose_detector.initialization_error:
            self.update_feedback_box(self.pose_detector.initialization_error)
            return

        # --- 步骤3: 初始化成功后，打开视频源 ---
        if self.is_camera:
            self.cap = cv2.VideoCapture(self.camera_id)
        else:
            if hasattr(self, 'video_path'):
                self.cap = cv2.VideoCapture(self.video_path)
            else:
                messagebox.showwarning("提示", "请先选择一个视频文件")
                return
        
        if not self.cap.isOpened():
            self.update_feedback_box("错误：无法打开指定的视频源！")
            return
        
        self.all_landmarks_timeline = []
        self.processed_frames = 0

        if not self.is_camera:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.update_feedback_box(f"开始分析视频... 总帧数: {self.total_frames}")

        self.is_running = True
        self.start_button["text"] = "停止"
        self.disable_controls()
        self.update_frame()  # 统一启动更新循环
    
    def _reset_ui_state(self):
        """仅重置UI控件的状态，不修改文本内容。"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None  # 释放摄像头资源
        
        self.start_button["text"] = "开始"
        self.enable_controls()

    def stop_detection(self):
        """停止检测或分析（通常由用户点击触发）"""
        if not self.is_running:
            return
        
        self.update_feedback_box("--- UI线程: stop_detection() 被调用 ---")
        self.save_analysis_data()  # 保存数据
        self._reset_ui_state()
        self.update_feedback_box("分析已停止。")

    def save_analysis_data(self):
        if not hasattr(self, 'video_path') or not self.all_landmarks_timeline:
            return

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.basename(self.video_path)
        report_filename = f"{video_filename}.analysis_data.json"
        report_filepath = os.path.join(output_dir, report_filename)

        try:
            with open(report_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.all_landmarks_timeline, f, ensure_ascii=False, indent=4)
            self.update_feedback_box(f"姿态数据已保存至: {report_filepath}")
        except Exception as e:
            self.update_feedback_box(f"保存分析数据时出错: {e}")

    def update_feedback_box(self, message):
        """安全地从任何线程更新反馈文本框的内容 (追加模式用于调试)"""
        self.feedback_text.insert(tk.END, message + "\n")

    def update_frame(self):
        """更新视频帧 (仅用于实时摄像头模式)"""
        if not self.is_running:
            return

        if self.cap is None:
            self.stop_detection()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            if not self.is_camera:
                self.update_feedback_box("视频分析完毕")
            return

        self.processed_frames += 1
        timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

        if self.pose_detector is None or self.pose_analyzer is None:
            self.update_feedback_box("检测器或分析器未初始化")
            return

        # 姿势检测
        processed_frame, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)

        if landmarks:
            self.all_landmarks_timeline.append({'time_ms': timestamp_ms, 'landmarks': landmarks})

        # 分析
        feedback_list = self.pose_analyzer.analyze_pose(landmarks)
        feedback_str = "\n".join(feedback_list)
        self.update_feedback_box(feedback_str)

        # 显示
        processed_frame = cv2.resize(processed_frame, (640, 480))
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk  # type: ignore[attr-defined]

        # 更新进度（对于视频）
        if not self.is_camera and self.total_frames > 0:
            progress_percent = (self.processed_frames / self.total_frames) * 100
            self.update_feedback_box(f"进度: {self.processed_frames}/{self.total_frames} ({progress_percent:.1f}%)")

        # 定时更新，基于 FPS
        delay = int(1000 / self.fps)
        self.root.after(delay, self.update_frame) 