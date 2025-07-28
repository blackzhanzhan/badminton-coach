import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
import json
import numpy as np
import sys
from datetime import datetime
import configparser
from html.parser import HTMLParser

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hybrid_action_advisor import HybridActionAdvisor
from modules.action_quality_predictor import ActionQualityPredictor
from modules.online_learning_manager import OnlineLearningManager

class MarkdownHTMLParser(HTMLParser):
    """HTML解析器，用于将HTML渲染到tkinter Text组件"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.tag_stack = []
        self.in_list = False
        self.list_level = 0
        
    def handle_starttag(self, tag, attrs):
        if tag in ['h1', 'h2', 'h3']:
            self.tag_stack.append(tag)
        elif tag == 'strong' or tag == 'b':
            self.tag_stack.append('bold')
        elif tag == 'em' or tag == 'i':
            self.tag_stack.append('italic')
        elif tag == 'code':
            self.tag_stack.append('code')
        elif tag in ['ul', 'ol']:
            self.in_list = True
            self.list_level += 1
            if self.text_widget.get("end-2c", "end-1c") != "\n":
                self.text_widget.insert(tk.END, "\n")
        elif tag == 'li':
            if self.in_list:
                # 添加适当的缩进
                indent = "  " * (self.list_level - 1)
                self.text_widget.insert(tk.END, f"{indent}• ", "list_bullet")
            self.tag_stack.append('list_item')
        elif tag == 'br':
            self.text_widget.insert(tk.END, "\n")
        elif tag in ['p', 'div']:
            if self.text_widget.get("end-2c", "end-1c") != "\n":
                self.text_widget.insert(tk.END, "\n")
    
    def handle_endtag(self, tag):
        if tag in ['h1', 'h2', 'h3', 'strong', 'b', 'em', 'i', 'code', 'li']:
            if self.tag_stack:
                self.tag_stack.pop()
        elif tag in ['ul', 'ol']:
            self.in_list = False if self.list_level == 1 else True
            self.list_level = max(0, self.list_level - 1)
            self.text_widget.insert(tk.END, "\n")
        if tag in ['p', 'div', 'li', 'h1', 'h2', 'h3']:
            self.text_widget.insert(tk.END, "\n")
    
    def handle_data(self, data):
        if self.tag_stack:
            current_tag = self.tag_stack[-1]
            self.text_widget.insert(tk.END, data, current_tag)
        else:
            self.text_widget.insert(tk.END, data, "content")

class MainWindow:
    def __init__(self, root):
        """初始化主窗口"""
        self.root = root
        self.root.title("羽毛球接球姿态分析与教练系统")
        self.root.geometry("1200x800")
        
        # 机器学习模块初始化
        # 检测器和分析器功能已集成到HybridActionAdvisor中
        
        # 视频相关变量
        self.cap = None
        self.is_camera = True
        self.camera_id = 0
        self.is_running = False
        
        # 调试模式
        
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
        
        # 混合动作顾问相关
        self.hybrid_advisor = None
        self.ml_enabled = True
        self.current_analysis_result = None
        
        # 用户反馈相关
        self.feedback_window = None
        self.user_rating = None
        
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
        
        # 显示/隐藏API密钥按钮
        self.show_key_var = tk.BooleanVar()
        self.show_key_check = ttk.Checkbutton(api_inner_frame, text="显示", 
                                            variable=self.show_key_var,
                                            command=self.toggle_api_key_visibility)
        self.show_key_check.pack(side=tk.LEFT, padx=2)
        
        # 保存API密钥按钮
        self.save_api_button = ttk.Button(api_inner_frame, text="保存密钥", command=self.save_api_key)
        self.save_api_button.pack(side=tk.LEFT, padx=5)
        
        # 机器学习配置区域
        ml_frame = ttk.LabelFrame(main_frame, text="🤖 机器学习配置")
        ml_frame.pack(fill=tk.X, pady=(0, 10))
        
        ml_inner_frame = ttk.Frame(ml_frame)
        ml_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # ML启用开关
        self.ml_enabled_var = tk.BooleanVar(value=True)
        self.ml_check = ttk.Checkbutton(ml_inner_frame, text="启用机器学习增强分析", 
                                       variable=self.ml_enabled_var,
                                       command=self.toggle_ml_mode)
        self.ml_check.pack(side=tk.LEFT, padx=(0, 20))
        
        # ML权重配置
        ttk.Label(ml_inner_frame, text="ML权重:").pack(side=tk.LEFT, padx=(0, 5))
        self.ml_weight_var = tk.DoubleVar(value=0.4)
        self.ml_weight_scale = ttk.Scale(ml_inner_frame, from_=0.0, to=1.0, 
                                        variable=self.ml_weight_var, 
                                        orient=tk.HORIZONTAL, length=100)
        self.ml_weight_scale.pack(side=tk.LEFT, padx=5)
        
        self.ml_weight_label = ttk.Label(ml_inner_frame, text="0.4")
        self.ml_weight_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # 绑定权重变化事件
        self.ml_weight_var.trace('w', self.update_ml_weight_label)
        
        # 模型状态显示
        self.ml_status_label = ttk.Label(ml_inner_frame, text="模型状态: 未初始化", foreground="gray")
        self.ml_status_label.pack(side=tk.LEFT, padx=10)
        
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
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        right_panel.config(width=400)
        
        # 处理状态区域
        status_frame = ttk.LabelFrame(right_panel, text="处理状态")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.feedback_text = tk.Text(status_frame, wrap=tk.WORD, height=12, width=45)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=scrollbar.set)
        
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # AI流式生成区域
        streaming_frame = ttk.LabelFrame(right_panel, text="🤖 AI教练实时生成")
        streaming_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.streaming_text = tk.Text(streaming_frame, wrap=tk.WORD, height=12, width=45, 
                                     bg='#f8f9fa', fg='#2c3e50', font=('Consolas', 9))
        streaming_scrollbar = ttk.Scrollbar(streaming_frame, orient="vertical", command=self.streaming_text.yview)
        self.streaming_text.configure(yscrollcommand=streaming_scrollbar.set)
        
        self.streaming_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        streaming_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化流式显示区域
        self.streaming_text.insert(tk.END, "等待AI教练开始分析...\n")
        self.streaming_text.config(state=tk.DISABLED)
        
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
        
        # 用户反馈按钮
        self.feedback_button = ttk.Button(button_frame, text="📝 提供反馈", command=self.show_feedback_dialog, state="disabled")
        self.feedback_button.pack(side=tk.LEFT, padx=5)
        
        # 模型重训练按钮
        self.retrain_button = ttk.Button(button_frame, text="🔄 重训练模型", command=self.retrain_models, state="disabled")
        self.retrain_button.pack(side=tk.LEFT, padx=5)
        
        # 系统状态按钮
        self.status_button = ttk.Button(button_frame, text="📊 系统状态", command=self.show_system_status, state="disabled")
        self.status_button.pack(side=tk.LEFT, padx=5)
        
        # 初始化变量
        self.api_key_saved = False
        self.auto_process_enabled = True
        self.config_file = "config.ini"
        
        # 加载保存的配置
        self.load_config()
        
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
        
        # 保存到配置文件
        self.save_config(api_key)
        
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
        self.pose_analyzer = PoseAnalyzer()
        
        if self.pose_detector.initialization_error:
            raise Exception(self.pose_detector.initialization_error)
        
        # 初始化传统动作顾问
        self.action_advisor = ActionAdvisor(
            status_callback=self.update_llm_status,
            streaming_callback=self.update_streaming_content
        )
        
        # 初始化混合动作顾问
        if self.ml_enabled_var.get():
            try:
                self.hybrid_advisor = HybridActionAdvisor(
                    templates_dir="data/staged_templates",
                    staged_dir="data/staged_templates",
                    model_dir="models",
                    learning_data_dir="data/feedback",
                    enable_ml=True,
                    ml_weight=self.ml_weight_var.get(),
                    rule_weight=1.0 - self.ml_weight_var.get(),
                    callback_progress=self.update_progress,
                    callback_complete=self.update_feedback_box
                )
                self.ml_status_label.config(text="模型状态: 已初始化", foreground="green")
                # 启用相关按钮
                self.feedback_button.config(state="normal")
                self.retrain_button.config(state="normal")
                self.status_button.config(state="normal")
                self.root.after(0, lambda: self.update_feedback_box("✅ 混合ML模型初始化完成"))
            except Exception as e:
                self.root.after(0, lambda: self.update_feedback_box(f"⚠️ ML模型初始化失败，使用传统模式: {str(e)}"))
                self.ml_enabled = False
                self.ml_enabled_var.set(False)
                self.ml_status_label.config(text="模型状态: 初始化失败", foreground="red")
        
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
        self.root.after(0, lambda: self.update_feedback_box("🎬 开始视频预览..."))
        
        frame_count = 0
        first_frame_displayed = False
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp_ms = int(frame_count * (1000 / self.fps))
            
            # 姿态检测
            processed_frame, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)
            
            # 显示第一帧作为初始预览
            if not first_frame_displayed:
                preview_frame = processed_frame.copy()
                preview_frame = cv2.resize(preview_frame, (640, 480))
                frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.root.after(0, lambda img=img_tk: self._update_video_display(img))
                first_frame_displayed = True
            
            if landmarks:
                self.all_landmarks_timeline.append({'time_ms': timestamp_ms, 'landmarks': landmarks})
            
            # 更新进度
            progress = 20 + (frame_count / self.total_frames) * 60  # 20-80%的进度用于视频分析
            self.root.after(0, lambda p=progress: self.update_progress(p, f"分析进度: {frame_count}/{self.total_frames}"))
            
            # 每3帧更新一次显示，提供更高帧率的预览
            if frame_count % 3 == 0:
                # 创建预览帧的副本
                preview_frame = processed_frame.copy()
                preview_frame = cv2.resize(preview_frame, (640, 480))
                frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.root.after(0, lambda img=img_tk: self._update_video_display(img))
        
        cap.release()
        self.processed_frames = frame_count
        
        # 保存分析数据
        self._save_analysis_data()
        self.root.after(0, lambda: self.update_feedback_box(f"✅ 视频分析完成，共处理 {frame_count} 帧"))
        self.root.after(0, lambda: self.update_feedback_box("🎬 视频预览已结束"))
    
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
    
    def _auto_analyze_with_ai(self):
        """自动进行AI智能分析"""
        try:
            # 设置API密钥
            api_key = self.api_key_entry.get().strip()
            if not api_key:
                self.update_feedback_box("❌ 请先设置API密钥")
                return
            
            os.environ['VOLCENGINE_API_KEY'] = api_key
            
            # 查找标准模板文件
            template_path = "data/staged_templates/击球动作模板.json"
            if not os.path.exists(template_path):
                self.update_feedback_box("❌ 未找到标准模板文件")
                return
            
            # 查找staged用户数据文件
            staged_user_file = None
            staged_dir = "data/staged_templates"
            if os.path.exists(staged_dir):
                # 查找最新的staged用户数据文件
                staged_files = [f for f in os.listdir(staged_dir) 
                              if f.startswith("staged_") and f.endswith(".json") 
                              and "模板" not in f and "template" not in f.lower()]
                if staged_files:
                    # 按修改时间排序，获取最新的
                    staged_files.sort(key=lambda x: os.path.getmtime(os.path.join(staged_dir, x)), reverse=True)
                    staged_user_file = os.path.join(staged_dir, staged_files[0])
            
            if not staged_user_file or not os.path.exists(staged_user_file):
                self.update_feedback_box("❌ 未找到阶段化用户数据文件，请先完成视频分析和转换")
                return
            
            self.update_feedback_box(f"📊 正在对比分析: {os.path.basename(staged_user_file)}")
            
            # 清空流式显示区域
            self.streaming_text.config(state=tk.NORMAL)
            self.streaming_text.delete(1.0, tk.END)
            self.streaming_text.insert(tk.END, "🤖 AI教练开始分析...\n\n")
            self.streaming_text.config(state=tk.DISABLED)
            
            # 使用混合动作顾问进行智能分析
            if hasattr(self, 'hybrid_advisor') and self.hybrid_advisor and self.ml_enabled_var.get():
                # 使用混合ML分析
                self.update_feedback_box("🤖 使用混合ML模式进行分析...")
                comprehensive_report = self.hybrid_advisor.analyze_action(
                    staged_user_file, 
                    template_path
                )
                # 保存当前分析结果用于反馈收集
                self.current_analysis_result = comprehensive_report
            else:
                # 使用传统分析
                self.update_feedback_box("📊 使用传统模式进行分析...")
                # 设置API密钥
                self.action_advisor.api_key = api_key
                comprehensive_report = self.action_advisor.generate_comprehensive_advice(
                    staged_user_file, template_path
                )
                self.current_analysis_result = comprehensive_report
            
            # 显示分析结果
            self.update_feedback_box("\n🎯 智能分析结果:")
            self.update_feedback_box("=" * 50)
            
            # 保存分析报告
            self._save_analysis_report(comprehensive_report)
            
            # 显示分析报告窗口
            self._show_analysis_report_window(comprehensive_report)
            
            self.update_feedback_box("\n✅ 智能分析完成！")
            
            # 调用分析完成后的操作
            self._on_analysis_complete()
            
        except Exception as e:
            error_msg = f"智能分析失败: {str(e)}"
            self.update_feedback_box(f"❌ {error_msg}")
            # 即使出错也要重置UI状态
            self._reset_ui_state()
        
        finally:
            if 'VOLCENGINE_API_KEY' in os.environ:
                del os.environ['VOLCENGINE_API_KEY']
    
    def _save_analysis_report(self, comprehensive_report):
        """保存分析报告"""
        try:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            video_filename = os.path.basename(self.video_path)
            report_filename = f"{video_filename}.analysis_report.txt"
            report_filepath = os.path.join(output_dir, report_filename)
            
            with open(report_filepath, 'w', encoding='utf-8') as f:
                f.write("羽毛球动作智能分析报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"视频文件: {video_filename}\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                

                
                # 详细建议
                detailed_suggestions = comprehensive_report.get('detailed_suggestions', [])
                if detailed_suggestions:
                    f.write("具体改进建议:\n")
                    f.write("-" * 30 + "\n")
                    for i, suggestion in enumerate(detailed_suggestions, 1):
                        f.write(f"{i}. {suggestion}\n\n")
                
                # LLM增强建议
                llm_advice = comprehensive_report.get('llm_enhanced_advice', '')
                if llm_advice:
                    f.write("AI智能建议:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"{llm_advice}\n\n")
            
            self.update_feedback_box(f"💾 分析报告已保存: {report_filename}")
            
        except Exception as e:
            self.update_feedback_box(f"❌ 保存报告失败: {str(e)}")
    
    def _show_analysis_report_window(self, comprehensive_report):
        """显示分析报告窗口"""
        # 创建新窗口
        report_window = tk.Toplevel(self.root)
        report_window.title("🏸 羽毛球动作智能分析报告")
        report_window.geometry("900x700")
        report_window.resizable(True, True)
        report_window.configure(bg='#f0f0f0')
        
        # 设置窗口图标（如果有的话）
        try:
            report_window.iconbitmap(default="icon.ico")
        except:
            pass
        
        # 创建样式
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Microsoft YaHei', 18, 'bold'), foreground='#2c3e50')
        style.configure('Info.TLabel', font=('Microsoft YaHei', 10), foreground='#34495e')
        style.configure('Header.TLabelframe.Label', font=('Microsoft YaHei', 12, 'bold'), foreground='#2980b9')
        style.configure('Custom.TButton', font=('Microsoft YaHei', 10))
        
        # 主框架 - 添加渐变背景效果
        main_frame = tk.Frame(report_window, bg='#ffffff', relief=tk.RAISED, bd=1)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 顶部装饰条
        top_bar = tk.Frame(main_frame, bg='#3498db', height=5)
        top_bar.pack(fill=tk.X, pady=(0, 20))
        
        # 标题区域
        title_frame = tk.Frame(main_frame, bg='#ffffff')
        title_frame.pack(fill=tk.X, pady=(0, 25))
        
        title_label = tk.Label(title_frame, text="🏸 羽毛球动作智能分析报告", 
                              font=('Microsoft YaHei', 20, 'bold'), 
                              fg='#2c3e50', bg='#ffffff')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="AI Powered Badminton Motion Analysis", 
                                 font=('Arial', 10, 'italic'), 
                                 fg='#7f8c8d', bg='#ffffff')
        subtitle_label.pack(pady=(5, 0))
        
        # 信息卡片
        info_card = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RAISED, bd=1)
        info_card.pack(fill=tk.X, pady=(0, 20), padx=20)
        
        info_inner = tk.Frame(info_card, bg='#ecf0f1')
        info_inner.pack(fill=tk.X, padx=20, pady=15)
        
        video_filename = os.path.basename(self.video_path)
        analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 视频文件信息
        file_frame = tk.Frame(info_inner, bg='#ecf0f1')
        file_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(file_frame, text="📹 视频文件:", font=('Microsoft YaHei', 11, 'bold'), 
                fg='#2980b9', bg='#ecf0f1').pack(side=tk.LEFT)
        tk.Label(file_frame, text=video_filename, font=('Microsoft YaHei', 11), 
                fg='#2c3e50', bg='#ecf0f1').pack(side=tk.LEFT, padx=(10, 0))
        
        # 分析时间信息
        time_frame = tk.Frame(info_inner, bg='#ecf0f1')
        time_frame.pack(fill=tk.X)
        
        tk.Label(time_frame, text="⏰ 分析时间:", font=('Microsoft YaHei', 11, 'bold'), 
                fg='#2980b9', bg='#ecf0f1').pack(side=tk.LEFT)
        tk.Label(time_frame, text=analysis_time, font=('Microsoft YaHei', 11), 
                fg='#2c3e50', bg='#ecf0f1').pack(side=tk.LEFT, padx=(10, 0))
        
        # 分析建议区域
        suggestions_frame = tk.LabelFrame(main_frame, text="💡 智能分析建议", 
                                         font=('Microsoft YaHei', 14, 'bold'),
                                         fg='#2980b9', bg='#ffffff', 
                                         relief=tk.RAISED, bd=2)
        suggestions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20), padx=20)
        
        # 创建文本框容器
        text_container = tk.Frame(suggestions_frame, bg='#ffffff')
        text_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # 文本框
        suggestions_text = tk.Text(text_container, wrap=tk.WORD, 
                                  font=('Microsoft YaHei', 12), 
                                  state=tk.NORMAL, 
                                  bg='#fafafa', 
                                  fg='#2c3e50',
                                  relief=tk.FLAT,
                                  selectbackground='#3498db',
                                  selectforeground='white',
                                  insertbackground='#2c3e50',
                                  padx=15, pady=15,
                                  spacing1=5, spacing2=3, spacing3=5)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL, command=suggestions_text.yview)
        suggestions_text.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        suggestions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 插入分析建议内容 - 支持Markdown渲染
        suggestions_text.delete(1.0, tk.END)
        
        # 配置文本标签样式
        suggestions_text.tag_configure("h1", font=('Microsoft YaHei', 16, 'bold'), foreground='#2c3e50')
        suggestions_text.tag_configure("h2", font=('Microsoft YaHei', 14, 'bold'), foreground='#2980b9')
        suggestions_text.tag_configure("h3", font=('Microsoft YaHei', 13, 'bold'), foreground='#34495e')
        suggestions_text.tag_configure("bold", font=('Microsoft YaHei', 11, 'bold'), foreground='#2c3e50')
        suggestions_text.tag_configure("italic", font=('Microsoft YaHei', 11, 'italic'), foreground='#2c3e50')
        suggestions_text.tag_configure("number", font=('Microsoft YaHei', 12, 'bold'), foreground='#e74c3c')
        suggestions_text.tag_configure("content", font=('Microsoft YaHei', 11), foreground='#2c3e50')
        suggestions_text.tag_configure("separator", font=('Microsoft YaHei', 8), foreground='#bdc3c7')
        suggestions_text.tag_configure("code", font=('Consolas', 10), background='#f8f9fa', foreground='#e74c3c')
        suggestions_text.tag_configure("list_item", font=('Microsoft YaHei', 11), foreground='#2c3e50', lmargin1=20, lmargin2=20)
        

        
        # 显示详细建议
        detailed_suggestions = comprehensive_report.get('detailed_suggestions', [])
        if detailed_suggestions:
            suggestions_text.insert(tk.END, "💡 具体改进建议\n", "h2")
            suggestions_text.insert(tk.END, "─" * 30 + "\n", "separator")
            for i, suggestion in enumerate(detailed_suggestions, 1):
                suggestions_text.insert(tk.END, f"【建议 {i}】", "number")
                suggestions_text.insert(tk.END, "\n")
                self._append_markdown_content(suggestions_text, suggestion)
                suggestions_text.insert(tk.END, "\n\n", "separator")
                if i < len(detailed_suggestions):
                    suggestions_text.insert(tk.END, "─" * 50 + "\n\n", "separator")
        
        # 显示LLM增强建议
        llm_advice = comprehensive_report.get('llm_enhanced_advice', '')
        if llm_advice:
            suggestions_text.insert(tk.END, "🤖 AI智能建议\n", "h2")
            suggestions_text.insert(tk.END, "─" * 20 + "\n", "separator")
            self._append_markdown_content(suggestions_text, llm_advice)
            suggestions_text.insert(tk.END, "\n\n", "separator")
        
        suggestions_text.config(state=tk.DISABLED)
        
        # 按钮区域 - 美化按钮
        button_frame = tk.Frame(main_frame, bg='#ffffff')
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # 左侧按钮组
        left_buttons = tk.Frame(button_frame, bg='#ffffff')
        left_buttons.pack(side=tk.LEFT)
        
        # 导出按钮
        export_btn = tk.Button(left_buttons, text="📄 导出报告", 
                              font=('Microsoft YaHei', 10, 'bold'),
                              bg='#27ae60', fg='white',
                              relief=tk.FLAT, padx=20, pady=8,
                              cursor='hand2',
                              command=lambda: self._export_current_report(comprehensive_report))
        export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 复制按钮
        copy_btn = tk.Button(left_buttons, text="📋 复制内容", 
                            font=('Microsoft YaHei', 10),
                            bg='#3498db', fg='white',
                            relief=tk.FLAT, padx=20, pady=8,
                            cursor='hand2',
                            command=lambda: self._copy_report_content(comprehensive_report))
        copy_btn.pack(side=tk.LEFT)
        
        # 右侧按钮组
        right_buttons = tk.Frame(button_frame, bg='#ffffff')
        right_buttons.pack(side=tk.RIGHT)
        
        # 关闭按钮
        close_btn = tk.Button(right_buttons, text="✖ 关闭", 
                             font=('Microsoft YaHei', 10),
                             bg='#e74c3c', fg='white',
                             relief=tk.FLAT, padx=20, pady=8,
                             cursor='hand2',
                             command=report_window.destroy)
        close_btn.pack(side=tk.RIGHT)
        
        # 按钮悬停效果
        def on_enter(event, btn, color):
            btn.configure(bg=color)
        
        def on_leave(event, btn, color):
            btn.configure(bg=color)
        
        export_btn.bind("<Enter>", lambda e: on_enter(e, export_btn, '#229954'))
        export_btn.bind("<Leave>", lambda e: on_leave(e, export_btn, '#27ae60'))
        
        copy_btn.bind("<Enter>", lambda e: on_enter(e, copy_btn, '#2980b9'))
        copy_btn.bind("<Leave>", lambda e: on_leave(e, copy_btn, '#3498db'))
        
        close_btn.bind("<Enter>", lambda e: on_enter(e, close_btn, '#c0392b'))
        close_btn.bind("<Leave>", lambda e: on_leave(e, close_btn, '#e74c3c'))
        
        # 窗口设置
        report_window.transient(self.root)
        report_window.grab_set()
        
        # 计算居中位置
        report_window.update_idletasks()
        x = (report_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (report_window.winfo_screenheight() // 2) - (700 // 2)
        report_window.geometry(f"900x700+{x}+{y}")
        
        # 添加淡入效果
        report_window.attributes('-alpha', 0.0)
        report_window.after(10, lambda: self._fade_in_window(report_window))
    
    def _fade_in_window(self, window, alpha=0.0):
        """窗口淡入效果"""
        alpha += 0.1
        if alpha <= 1.0:
            window.attributes('-alpha', alpha)
            window.after(30, lambda: self._fade_in_window(window, alpha))
        else:
            window.attributes('-alpha', 1.0)
    
    def _render_markdown_content(self, text_widget, content):
        """渲染Markdown内容到Text组件（改进版）"""
        import re
        
        # 清空文本组件
        text_widget.delete(1.0, tk.END)
        
        # 配置文本样式
        self._configure_text_styles(text_widget)
        
        # 使用markdown库将markdown转换为HTML
        try:
            import markdown
            html_content = markdown.markdown(content, extensions=['extra', 'codehilite'])
        except:
            # 如果markdown库不可用，使用简化版本
            html_content = self._simple_markdown_to_html(content)
        
        # 解析HTML并渲染到Text组件
        parser = MarkdownHTMLParser(text_widget)
        parser.feed(html_content)
    
    def _append_markdown_content(self, text_widget, content):
        """追加Markdown内容到Text组件（不清空现有内容）"""
        import re
        
        # 配置文本样式（如果还没有配置）
        self._configure_text_styles(text_widget)
        
        # 使用markdown库将markdown转换为HTML
        try:
            import markdown
            html_content = markdown.markdown(content, extensions=['extra', 'codehilite'])
        except:
            # 如果markdown库不可用，使用简化版本
            html_content = self._simple_markdown_to_html(content)
        
        # 解析HTML并渲染到Text组件
        parser = MarkdownHTMLParser(text_widget)
        parser.feed(html_content)
    
    def _configure_text_styles(self, text_widget):
        """配置文本组件的样式"""
        text_widget.tag_configure("h1", font=("Arial", 16, "bold"), foreground="#2c3e50")
        text_widget.tag_configure("h2", font=("Arial", 14, "bold"), foreground="#34495e")
        text_widget.tag_configure("h3", font=("Arial", 12, "bold"), foreground="#7f8c8d")
        text_widget.tag_configure("bold", font=("Arial", 10, "bold"))
        text_widget.tag_configure("italic", font=("Arial", 10, "italic"))
        text_widget.tag_configure("code", font=("Courier", 9), background="#f8f9fa", foreground="#e74c3c")
        text_widget.tag_configure("list_bullet", foreground="#3498db")
        text_widget.tag_configure("content", font=("Arial", 10))
    
    def _simple_markdown_to_html(self, content):
        """简化的markdown到HTML转换"""
        import re
        
        lines = content.split('\n')
        html_lines = []
        in_list = False
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            if not line:
                if in_list:
                    # 空行可能结束列表
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if not (next_line.startswith('- ') or next_line.startswith('* ') or re.match(r'^\d+\. ', next_line)):
                        html_lines.append('</ul>')
                        in_list = False
                html_lines.append('<br>')
                continue
            
            # 处理标题
            if line.startswith('### '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h3>{line[4:]}</h3>')
            elif line.startswith('## '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('# '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h1>{line[2:]}</h1>')
            # 处理列表项
            elif line.startswith('- ') or line.startswith('* '):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                formatted_content = self._format_inline_html(line[2:])
                html_lines.append(f'<li>{formatted_content}</li>')
            elif re.match(r'^\d+\. ', line):
                if not in_list:
                    html_lines.append('<ol>')
                    in_list = True
                match = re.match(r'^\d+\. (.+)', line)
                if match:
                    formatted_content = self._format_inline_html(match.group(1))
                    html_lines.append(f'<li>{formatted_content}</li>')
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                # 处理行内格式
                formatted_line = self._format_inline_html(line)
                html_lines.append(f'<p>{formatted_line}</p>')
        
        # 确保列表正确关闭
        if in_list:
            html_lines.append('</ul>')
        
        return '\n'.join(html_lines)
    
    def _format_inline_html(self, text):
        """格式化行内HTML"""
        import re
        
        # 处理粗体
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        # 处理斜体
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        # 处理代码
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        return text
    
    def _copy_report_content(self, comprehensive_report):
        """复制报告内容到剪贴板"""
        try:
            content = "羽毛球动作智能分析报告\n"
            content += "=" * 50 + "\n"
            content += f"视频文件: {os.path.basename(self.video_path)}\n"
            content += f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # 详细建议
            detailed_suggestions = comprehensive_report.get('detailed_suggestions', [])
            if detailed_suggestions:
                content += "具体改进建议:\n"
                content += "-" * 30 + "\n"
                for i, suggestion in enumerate(detailed_suggestions, 1):
                    content += f"{i}. {suggestion}\n\n"
            
            # LLM增强建议
            llm_advice = comprehensive_report.get('llm_enhanced_advice', '')
            if llm_advice:
                content += "AI智能建议:\n"
                content += "-" * 20 + "\n"
                content += f"{llm_advice}\n\n"
            
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("复制成功", "报告内容已复制到剪贴板！")
            
        except Exception as e:
            messagebox.showerror("复制失败", f"复制内容时发生错误: {str(e)}")
    
    def _export_current_report(self, comprehensive_report):
        """导出当前分析报告"""
        file_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialname=f"{os.path.basename(self.video_path)}_分析报告.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("羽毛球动作智能分析报告\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"视频文件: {os.path.basename(self.video_path)}\n")
                    f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # 详细建议
                    detailed_suggestions = comprehensive_report.get('detailed_suggestions', [])
                    if detailed_suggestions:
                        f.write("具体改进建议:\n")
                        f.write("-" * 30 + "\n")
                        for i, suggestion in enumerate(detailed_suggestions, 1):
                            f.write(f"{i}. {suggestion}\n\n")
                    
                    # LLM增强建议
                    llm_advice = comprehensive_report.get('llm_enhanced_advice', '')
                    if llm_advice:
                        f.write("AI智能建议:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"{llm_advice}\n\n")
                
                messagebox.showinfo("导出成功", f"报告已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("导出失败", f"保存文件时发生错误: {str(e)}")
    
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
        staged_dir = "data/staged_templates"
        if os.path.exists(staged_dir):
            staged_files = [f for f in os.listdir(staged_dir) if f.endswith(".json") and "staged_" in f]
            if staged_files:
                latest_staged = max(staged_files, key=lambda x: os.path.getmtime(os.path.join(staged_dir, x)))
                self.update_feedback_box(f"  - 阶段化数据: {latest_staged}")
        
        # 自动进行智能分析
        self.update_feedback_box("\n🤖 开始智能分析...")
        self._auto_analyze_with_ai()
        
    def update_llm_status(self, status_message):
        """更新LLM连接状态信息"""
        self.root.after(0, lambda: self.update_feedback_box(f"🔗 {status_message}"))
    
    def update_streaming_content(self, content):
        """更新流式内容显示"""
        def _update():
            self.streaming_text.config(state=tk.NORMAL)
            self.streaming_text.insert(tk.END, content)
            self.streaming_text.see(tk.END)  # 自动滚动到底部
            self.streaming_text.config(state=tk.DISABLED)
        
        # 确保在主线程中更新UI
        if threading.current_thread() == threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def _on_analysis_complete(self):
        """分析完成后的操作"""
        self.update_feedback_box("\n✨ 您可以选择新的视频文件继续分析")
        
        messagebox.showinfo("完成", "视频分析、转换和智能分析已完成！\n\n生成的文件已保存到相应目录。")
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
    
    def load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                config = configparser.ConfigParser()
                config.read(self.config_file, encoding='utf-8')
                
                if 'API' in config and 'key' in config['API']:
                    api_key = config['API']['key']
                    if api_key:
                        self.api_key_entry.insert(0, api_key)
                        self.api_key_saved = True
                        self.file_button.config(state="normal")
                        self.update_feedback_box("✅ 已加载保存的API密钥")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def save_config(self, api_key):
        """保存配置到文件"""
        try:
            config = configparser.ConfigParser()
            config['API'] = {'key': api_key}
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                config.write(f)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def toggle_api_key_visibility(self):
        """切换API密钥显示/隐藏"""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
    
    def toggle_ml_mode(self):
        """切换机器学习模式"""
        self.ml_enabled = self.ml_enabled_var.get()
        if self.ml_enabled:
            self.ml_status_label.config(text="模型状态: 启用", foreground="green")
        else:
            self.ml_status_label.config(text="模型状态: 禁用", foreground="gray")
            # 禁用相关按钮
            self.feedback_button.config(state="disabled")
            self.retrain_button.config(state="disabled")
            self.status_button.config(state="disabled")
    
    def update_ml_weight_label(self, *args):
        """更新ML权重标签"""
        weight = self.ml_weight_var.get()
        self.ml_weight_label.config(text=f"{weight:.1f}")
    
    def show_feedback_dialog(self):
        """显示用户反馈对话框"""
        if not self.current_analysis_result:
            messagebox.showwarning("提示", "请先完成一次分析后再提供反馈")
            return
        
        # 创建反馈窗口
        feedback_window = tk.Toplevel(self.root)
        feedback_window.title("📝 用户反馈")
        feedback_window.geometry("500x400")
        feedback_window.resizable(False, False)
        feedback_window.transient(self.root)
        feedback_window.grab_set()
        
        # 居中显示
        feedback_window.update_idletasks()
        x = (feedback_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (feedback_window.winfo_screenheight() // 2) - (400 // 2)
        feedback_window.geometry(f"500x400+{x}+{y}")
        
        main_frame = ttk.Frame(feedback_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 标题
        title_label = ttk.Label(main_frame, text="请为本次分析提供反馈", 
                               font=('Microsoft YaHei', 14, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 评分区域
        rating_frame = ttk.LabelFrame(main_frame, text="分析质量评分 (1-5分)")
        rating_frame.pack(fill=tk.X, pady=(0, 15))
        
        rating_var = tk.IntVar(value=3)
        rating_inner = ttk.Frame(rating_frame)
        rating_inner.pack(fill=tk.X, padx=10, pady=10)
        
        for i in range(1, 6):
            ttk.Radiobutton(rating_inner, text=f"{i}分", variable=rating_var, 
                           value=i).pack(side=tk.LEFT, padx=10)
        
        # 满意度区域
        satisfaction_frame = ttk.LabelFrame(main_frame, text="整体满意度")
        satisfaction_frame.pack(fill=tk.X, pady=(0, 15))
        
        satisfaction_var = tk.StringVar(value="满意")
        satisfaction_inner = ttk.Frame(satisfaction_frame)
        satisfaction_inner.pack(fill=tk.X, padx=10, pady=10)
        
        for option in ["非常满意", "满意", "一般", "不满意", "非常不满意"]:
            ttk.Radiobutton(satisfaction_inner, text=option, variable=satisfaction_var, 
                           value=option).pack(side=tk.LEFT, padx=5)
        
        # 评论区域
        comment_frame = ttk.LabelFrame(main_frame, text="详细评论 (可选)")
        comment_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        comment_text = tk.Text(comment_frame, wrap=tk.WORD, height=6,
                              font=('Microsoft YaHei', 10))
        comment_scrollbar = ttk.Scrollbar(comment_frame, orient=tk.VERTICAL, 
                                         command=comment_text.yview)
        comment_text.configure(yscrollcommand=comment_scrollbar.set)
        
        comment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        comment_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def submit_feedback():
            try:
                if hasattr(self, 'hybrid_advisor') and self.hybrid_advisor:
                    feedback_data = {
                        'user_rating': rating_var.get(),
                        'satisfaction': satisfaction_var.get(),
                        'comments': comment_text.get(1.0, tk.END).strip()
                    }
                    
                    self.hybrid_advisor.collect_user_feedback(
                        session_data=self.current_analysis_result,
                        **feedback_data
                    )
                    
                    messagebox.showinfo("成功", "感谢您的反馈！这将帮助我们改进分析质量。")
                    feedback_window.destroy()
                else:
                    messagebox.showwarning("提示", "混合分析模式未启用，无法收集反馈")
            except Exception as e:
                messagebox.showerror("错误", f"提交反馈失败: {str(e)}")
        
        ttk.Button(button_frame, text="提交反馈", command=submit_feedback).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="取消", command=feedback_window.destroy).pack(side=tk.RIGHT)
    
    def retrain_models(self):
        """重训练模型"""
        if not hasattr(self, 'hybrid_advisor') or not self.hybrid_advisor:
            messagebox.showwarning("提示", "混合分析模式未启用")
            return
        
        result = messagebox.askyesno("确认", "重训练模型可能需要一些时间，是否继续？")
        if result:
            try:
                self.update_feedback_box("🔄 开始重训练模型...")
                success = self.hybrid_advisor.retrain_models()
                if success:
                    self.update_feedback_box("✅ 模型重训练完成")
                    messagebox.showinfo("成功", "模型重训练完成！")
                else:
                    self.update_feedback_box("⚠️ 模型重训练失败")
                    messagebox.showwarning("警告", "模型重训练失败，请检查反馈数据是否充足")
            except Exception as e:
                error_msg = f"重训练失败: {str(e)}"
                self.update_feedback_box(f"❌ {error_msg}")
                messagebox.showerror("错误", error_msg)
    
    def show_system_status(self):
        """显示系统状态"""
        if not hasattr(self, 'hybrid_advisor') or not self.hybrid_advisor:
            messagebox.showwarning("提示", "混合分析模式未启用")
            return
        
        try:
            status = self.hybrid_advisor.get_system_status()
            
            # 创建状态窗口
            status_window = tk.Toplevel(self.root)
            status_window.title("📊 系统状态")
            status_window.geometry("600x500")
            status_window.resizable(True, True)
            status_window.transient(self.root)
            status_window.grab_set()
            
            # 居中显示
            status_window.update_idletasks()
            x = (status_window.winfo_screenwidth() // 2) - (600 // 2)
            y = (status_window.winfo_screenheight() // 2) - (500 // 2)
            status_window.geometry(f"600x500+{x}+{y}")
            
            main_frame = ttk.Frame(status_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # 标题
            title_label = ttk.Label(main_frame, text="系统运行状态", 
                                   font=('Microsoft YaHei', 16, 'bold'))
            title_label.pack(pady=(0, 20))
            
            # 状态文本
            status_text = tk.Text(main_frame, wrap=tk.WORD, 
                                 font=('Microsoft YaHei', 10),
                                 state=tk.NORMAL)
            status_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, 
                                           command=status_text.yview)
            status_text.configure(yscrollcommand=status_scrollbar.set)
            
            status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 插入状态信息
            status_content = f"""🤖 机器学习模型状态
{'='*50}
模型状态: {status.get('ml_model_status', 'Unknown')}
训练数据量: {status.get('training_data_count', 0)}
模型准确率: {status.get('model_accuracy', 'N/A')}
最后训练时间: {status.get('last_training_time', 'Never')}

📊 反馈统计
{'='*50}
总反馈数: {status.get('total_feedback', 0)}
平均评分: {status.get('average_rating', 'N/A')}
最新反馈时间: {status.get('latest_feedback_time', 'Never')}

⚙️ 系统配置
{'='*50}
ML权重: {self.ml_weight_var.get():.1f}
ML模式: {'启用' if self.ml_enabled_var.get() else '禁用'}
模型目录: {status.get('model_directory', 'N/A')}
数据库路径: {status.get('database_path', 'N/A')}

📈 性能指标
{'='*50}
分析会话数: {status.get('analysis_sessions', 0)}
平均分析时间: {status.get('average_analysis_time', 'N/A')}
系统运行时间: {status.get('system_uptime', 'N/A')}
"""
            
            status_text.insert(tk.END, status_content)
            status_text.config(state=tk.DISABLED)
            
            # 关闭按钮
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=(20, 0))
            
            ttk.Button(button_frame, text="关闭", 
                      command=status_window.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("错误", f"获取系统状态失败: {str(e)}")