import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import numpy as np
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pose_detector import PoseDetector
from modules.pose_analyzer import PoseAnalyzer
from modules.json_converter import JsonConverter

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
        
        # 新增变量用于保存最近分析的JSON文件路径
        self.last_json_path = None
        
    def init_ui(self):
        """初始化用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # API密钥配置区域（顶部突出显示）
        api_frame = ttk.LabelFrame(main_frame, text="🔑 API配置 - 必须先配置才能使用")
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        api_inner_frame = ttk.Frame(api_frame)
        api_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(api_inner_frame, text="火山引擎API密钥:").pack(side=tk.LEFT, padx=(0, 5))
        self.api_key_entry = ttk.Entry(api_inner_frame, width=50, show="*")
        self.api_key_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 保存API密钥按钮
        self.save_api_button = ttk.Button(api_inner_frame, text="保存密钥", command=self.save_api_key)
        self.save_api_button.pack(side=tk.LEFT, padx=5)
        
        # 视频处理区域
        video_frame = ttk.LabelFrame(main_frame, text="📹 视频处理")
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        video_inner_frame = ttk.Frame(video_frame)
        video_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 计算设备选择
        ttk.Label(video_inner_frame, text="计算设备:").pack(side=tk.LEFT, padx=(0, 5))
        self.device_var = tk.StringVar(value="CPU")
        self.device_combo = ttk.Combobox(video_inner_frame, textvariable=self.device_var, 
                                   values=["CPU", "GPU"], width=8, state="readonly")
        self.device_combo.pack(side=tk.LEFT, padx=5)
        
        # 分隔符
        ttk.Separator(video_inner_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill='y', padx=15)
        
        # 视频文件选择
        ttk.Label(video_inner_frame, text="选择视频:").pack(side=tk.LEFT, padx=(0, 5))
        self.file_button = ttk.Button(video_inner_frame, text="📁 选择视频文件", command=self.select_and_process_video)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        # 当前文件显示
        self.current_file_label = ttk.Label(video_inner_frame, text="未选择文件", foreground="gray")
        self.current_file_label.pack(side=tk.LEFT, padx=10)
        
        # 上部分：视频显示和分析结果
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 左侧：视频显示
        self.video_display_frame = ttk.LabelFrame(content_frame, text="视频预览")
        self.video_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(self.video_display_frame, text="请先配置API密钥，然后选择视频文件")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 右侧：处理状态和结果
        status_frame = ttk.LabelFrame(content_frame, text="处理状态")
        status_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        status_frame.config(width=400)

        self.feedback_text = tk.Text(status_frame, wrap=tk.WORD, height=20, width=45)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=scrollbar.set)
        
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.feedback_text.insert(tk.END, "🚀 欢迎使用羽毛球姿态分析系统！\n\n")
        self.feedback_text.insert(tk.END, "📋 使用步骤：\n")
        self.feedback_text.insert(tk.END, "1. 输入火山引擎API密钥并保存\n")
        self.feedback_text.insert(tk.END, "2. 选择要分析的视频文件\n")
        self.feedback_text.insert(tk.END, "3. 系统将自动完成分析和转换\n\n")
        self.feedback_text.insert(tk.END, "等待您的操作...\n")
        
        # 控制按钮区域
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # 控制按钮
        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(button_frame, text="⏸️ 停止处理", command=self.stop_detection, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 调试模式
        self.debug_var = tk.BooleanVar(value=False)
        self.debug_check = ttk.Checkbutton(button_frame, text="🔧 调试模式", 
                                     variable=self.debug_var, 
                                     command=self.toggle_debug)
        self.debug_check.pack(side=tk.RIGHT, padx=5)
        
        # 初始化变量
        self.api_key_saved = False
        self.auto_process_enabled = True
        
        # 初始状态：禁用文件选择按钮
        self.file_button.config(state="disabled")
        
    def save_api_key(self):
        """保存API密钥"""
        api_key = self.api_key_entry.get().strip()
        if not api_key:
            messagebox.showwarning("警告", "请输入有效的API密钥")
            return
        
        # 简单验证API密钥格式（火山引擎API密钥通常较长）
        if len(api_key) < 20:
            messagebox.showwarning("警告", "API密钥格式可能不正确，请检查")
            return
        
        self.api_key_saved = True
        self.update_feedback_box("✅ API密钥已保存，现在可以选择视频文件进行分析")
        messagebox.showinfo("成功", "API密钥已保存！现在可以选择视频文件进行自动分析。")
        
        # 启用视频选择按钮
        self.file_button.config(state="normal")
    
    def select_and_process_video(self):
        """选择视频文件并自动开始处理"""
        if not self.api_key_saved:
            messagebox.showwarning("提示", "请先保存API密钥")
            return
        
        if self.is_running:
            messagebox.showwarning("提示", "当前正在处理视频，请等待完成或停止当前处理")
            return
        
        file_path = filedialog.askopenfilename(
            title="选择要分析的视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.current_file_label.config(text=f"已选择: {filename}", foreground="blue")
            self.update_feedback_box(f"\n📁 已选择视频文件: {filename}")
            
            # 自动开始处理
            self.auto_process_video()
    
    def auto_process_video(self):
        """自动处理视频：分析 + 转换为staged格式"""
        if not hasattr(self, 'video_path'):
            return
        
        self.update_feedback_box("\n🚀 开始自动处理视频...")
        self.update_feedback_box("第1步: 初始化姿态检测模型")
        
        # 在新线程中执行处理
        thread = threading.Thread(target=self._auto_process_thread)
        thread.daemon = True
        thread.start()
    
    def _auto_process_thread(self):
        """自动处理线程"""
        try:
            # 第1步：初始化模型
            self.root.after(0, lambda: self.update_progress(10, "初始化模型中..."))
            self._initialize_models()
            
            # 第2步：分析视频
            self.root.after(0, lambda: self.update_progress(20, "开始分析视频..."))
            self._analyze_video()
            
            # 第3步：转换为staged格式
            self.root.after(0, lambda: self.update_progress(80, "转换为阶段化格式..."))
            self._convert_to_staged()
            
            # 完成
            self.root.after(0, lambda: self.update_progress(100, "处理完成！"))
            self.root.after(0, self._on_process_complete)
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.root.after(0, lambda: self.update_feedback_box(f"❌ {error_msg}"))
            self.root.after(0, lambda: messagebox.showerror("错误", error_msg))
            self.root.after(0, self._reset_ui_state)
    
    def _initialize_models(self):
        """初始化检测和分析模型"""
        selected_device = self.device_var.get().lower()
        self.root.after(0, lambda: self.update_feedback_box(f"正在使用 {selected_device.upper()} 初始化模型..."))
        
        self.pose_detector = PoseDetector(device=selected_device)
        self.pose_analyzer = PoseAnalyzer(self.pose_detector.get_landmarks_info())
        
        if self.pose_detector.initialization_error:
            raise Exception(self.pose_detector.initialization_error)
        
        self.root.after(0, lambda: self.update_feedback_box("✅ 模型初始化完成"))
    
    def _analyze_video(self):
        """分析视频文件"""
        self.root.after(0, lambda: self.start_button.config(state="normal"))  # 启用停止按钮
        self.is_running = True
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.all_landmarks_timeline = []
        self.processed_frames = 0
        
        self.root.after(0, lambda: self.update_feedback_box(f"📊 视频信息: {self.total_frames}帧, {self.fps:.1f}FPS"))
        
        frame_count = 0
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp_ms = int(frame_count * (1000 / self.fps))
            
            # 姿态检测
            processed_frame, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)
            
            if landmarks:
                self.all_landmarks_timeline.append({'time_ms': timestamp_ms, 'landmarks': landmarks})
            
            # 更新进度
            progress = 20 + (frame_count / self.total_frames) * 60  # 20-80%的进度用于视频分析
            self.root.after(0, lambda p=progress: self.update_progress(p, f"分析进度: {frame_count}/{self.total_frames}"))
            
            # 每100帧更新一次显示
            if frame_count % 100 == 0:
                processed_frame = cv2.resize(processed_frame, (640, 480))
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.root.after(0, lambda img=img_tk: self._update_video_display(img))
        
        cap.release()
        self.processed_frames = frame_count
        
        # 保存分析数据
        self._save_analysis_data()
        self.root.after(0, lambda: self.update_feedback_box(f"✅ 视频分析完成，共处理 {frame_count} 帧"))
    
    def _save_analysis_data(self):
        """保存分析数据"""
        if not self.all_landmarks_timeline:
            return
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.basename(self.video_path)
        report_filename = f"{video_filename}.analysis_data.json"
        report_filepath = os.path.join(output_dir, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_landmarks_timeline, f, ensure_ascii=False, indent=4)
        
        self.last_json_path = report_filepath
        self.root.after(0, lambda: self.update_feedback_box(f"💾 分析数据已保存: {report_filename}"))
    
    def _convert_to_staged(self):
        """转换为阶段化格式"""
        if not self.last_json_path:
            raise Exception("没有找到分析数据文件")
        
        api_key = self.api_key_entry.get().strip()
        os.environ['VOLCENGINE_API_KEY'] = api_key
        
        try:
            converter = JsonConverter()
            output_path = converter.convert_to_staged_format(self.last_json_path)
            
            self.root.after(0, lambda: self.update_feedback_box(f"✅ 阶段化转换完成: {os.path.basename(output_path)}"))
            return output_path
            
        finally:
            if 'VOLCENGINE_API_KEY' in os.environ:
                del os.environ['VOLCENGINE_API_KEY']
    
    def _update_video_display(self, img_tk):
        """更新视频显示"""
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk
    
    def update_progress(self, value, status_text):
        """更新进度条和状态"""
        self.progress_var.set(value)
        if hasattr(self, 'feedback_text'):
            self.update_feedback_box(f"📊 {status_text} ({value:.1f}%)")
    
    def _on_process_complete(self):
        """处理完成后的操作"""
        self.update_feedback_box("\n🎉 自动处理完成！")
        self.update_feedback_box("📁 生成的文件:")
        self.update_feedback_box(f"  - 原始分析数据: {os.path.basename(self.last_json_path)}")
        
        # 查找staged文件
        staged_files = [f for f in os.listdir("staged_templates") if f.endswith(".json") and "staged_" in f]
        if staged_files:
            latest_staged = max(staged_files, key=lambda x: os.path.getmtime(os.path.join("staged_templates", x)))
            self.update_feedback_box(f"  - 阶段化数据: {latest_staged}")
        
        self.update_feedback_box("\n✨ 您可以选择新的视频文件继续分析")
        
        messagebox.showinfo("完成", "视频分析和转换已完成！\n\n生成的文件已保存到相应目录。")
        self._reset_ui_state()
    
    def disable_controls(self):
        """禁用控制控件"""
        self.file_button.config(state="disabled")
        self.device_combo.config(state="disabled")
        self.save_api_button.config(state="disabled")

    def enable_controls(self):
        """启用控制控件"""
        if self.api_key_saved:
            self.file_button.config(state="normal")
        self.device_combo.config(state="normal")
        self.save_api_button.config(state="normal")
        
    def toggle_debug(self):
        """切换调试模式"""
        self.debug_mode = self.debug_var.get()
    
    # 旧的start_detection方法已被自动处理流程替代
    
    def _reset_ui_state(self):
        """重置UI控件的状态"""
        self.is_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(text="⏸️ 停止处理", state="disabled")
        self.progress_var.set(0)
        self.enable_controls()

    def stop_detection(self):
        """停止当前处理"""
        if not self.is_running:
            return
        
        self.update_feedback_box("\n⏸️ 用户请求停止处理...")
        self.is_running = False
        
        # 如果有分析数据，保存它
        if hasattr(self, 'all_landmarks_timeline') and self.all_landmarks_timeline:
            self._save_analysis_data()
        
        self._reset_ui_state()
        self.update_feedback_box("✅ 处理已停止")
        messagebox.showinfo("提示", "处理已停止。如有分析数据已自动保存。")

    def update_feedback_box(self, message):
        """安全地从任何线程更新反馈文本框的内容"""
        self.feedback_text.insert(tk.END, message + "\n")
        self.feedback_text.see(tk.END)  # 自动滚动到底部 

    # 旧的手动转换方法已被自动处理流程替代