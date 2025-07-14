import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
from modules.pose_detector import PoseDetector
from modules.pose_analyzer import PoseAnalyzer
from modules.badminton_analyzer import BadmintonAnalyzer

class MainWindow:
    def __init__(self, root):
        """初始化主窗口"""
        self.root = root
        self.root.title("羽毛球姿态分析与教练系统")
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
        
        # 初始化UI
        self.init_ui()
        
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
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        
        # 计算设备选择
        device_label = ttk.Label(bottom_frame, text="计算设备:")
        device_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.device_var = tk.StringVar(value="CPU")
        device_combo = ttk.Combobox(bottom_frame, textvariable=self.device_var, 
                                   values=["CPU", "GPU"], width=8, state="readonly")
        device_combo.pack(side=tk.LEFT, padx=5)
        
        # 分隔符
        ttk.Separator(bottom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill='y', padx=10)

        # 输入源选择
        source_label = ttk.Label(bottom_frame, text="输入源:")
        source_label.pack(side=tk.LEFT, padx=5)
        
        self.source_var = tk.StringVar(value="摄像头")
        source_combo = ttk.Combobox(bottom_frame, textvariable=self.source_var, 
                                   values=["摄像头", "视频文件"], width=10, state="readonly")
        source_combo.pack(side=tk.LEFT, padx=5)
        source_combo.bind("<<ComboboxSelected>>", self.change_source)
        
        # 文件选择按钮
        self.file_button = ttk.Button(bottom_frame, text="选择文件", command=self.open_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        self.file_button["state"] = "disabled"
        
        # 开始/停止按钮
        self.start_button = ttk.Button(bottom_frame, text="开始", command=self.toggle_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 调试模式切换按钮
        self.debug_var = tk.BooleanVar(value=False)
        debug_check = ttk.Checkbutton(bottom_frame, text="调试模式", 
                                     variable=self.debug_var, 
                                     command=self.toggle_debug)
        debug_check.pack(side=tk.LEFT, padx=5)
        
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
        self.feedback_text.delete(1.0, tk.END)
        self.feedback_text.insert(tk.END, f"正在使用 {selected_device.upper()} 初始化模型，请稍候...")
        self.root.update_idletasks()

        try:
            self.pose_detector = PoseDetector(device=selected_device)
            self.pose_analyzer = PoseAnalyzer()
        except Exception as e:
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, f"创建检测器时发生未知错误: {e}")
            return

        # --- 步骤2: 检查初始化是否成功 (特别是GPU模式) ---
        if self.pose_detector.initialization_error:
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, self.pose_detector.initialization_error)
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
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, "错误：无法打开指定的视频源！")
            return
        
        self.is_running = True
        self.start_button["text"] = "停止"
        self.update_frame()
    
    def stop_detection(self):
        """停止检测"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_button["text"] = "开始"
    
    def update_frame(self):
        """更新视频帧"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            if not self.is_camera:
                self.feedback_text.delete(1.0, tk.END)
                self.feedback_text.insert(tk.END, "视频播放完毕")
            return
        
        # 如果是调试模式，显示颜色检测的中间结果
        if self.debug_mode:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 红色标记 (需要两个范围)
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            
            # 绿色标记
            green_lower = np.array([35, 100, 100])
            green_upper = np.array([85, 255, 255])
            
            # 蓝色标记
            blue_lower = np.array([100, 100, 100])
            blue_upper = np.array([140, 255, 255])
            
            # 黄色标记
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([35, 255, 255])
            
            # 创建颜色掩码
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            mask_green = cv2.inRange(hsv, green_lower, green_upper)
            mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
            mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
            
            # 合并所有颜色掩码
            mask_combined = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_blue, mask_yellow)))
            
            # 创建彩色掩码，每种颜色用不同的颜色显示
            mask_color = np.zeros_like(frame)
            mask_color[mask_red > 0] = [0, 0, 255]  # 红色
            mask_color[mask_green > 0] = [0, 255, 0]  # 绿色
            mask_color[mask_blue > 0] = [255, 0, 0]  # 蓝色
            mask_color[mask_yellow > 0] = [0, 255, 255]  # 黄色
            
            # 创建一个包含原始图像和彩色掩码的组合图像
            h, w = frame.shape[:2]
            combined = np.zeros((h, w*2, 3), dtype=np.uint8)
            combined[:, :w] = frame
            combined[:, w:] = mask_color
            
            # 在组合图像上添加标签
            cv2.putText(combined, "原始图像", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "颜色标记检测", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 调整大小以适应显示
            combined = cv2.resize(combined, (800, 300))
            
            # 转换为Tkinter可显示的格式
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(combined_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # 更新显示
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
            
            # 在调试模式下，仍然执行姿势检测以获取反馈
            timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            _, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)
            
            feedback_list = self.pose_analyzer.analyze_ready_stance(landmarks)
            feedback_str = "\n".join(feedback_list)
            
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, "--- 准备姿势分析 ---\n\n")
            self.feedback_text.insert(tk.END, feedback_str)

        else:
            # 1. 核心姿态检测 (加入时间戳)
            timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            processed_frame, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)

            # 2. 调用新的姿态分析器获取反馈
            feedback_list = self.pose_analyzer.analyze_ready_stance(landmarks)
            feedback_str = "\n".join(feedback_list)
            
            self.feedback_text.delete(1.0, tk.END)
            self.feedback_text.insert(tk.END, "--- 准备姿势分析 ---\n\n")
            self.feedback_text.insert(tk.END, feedback_str)
            
            # 调整图像大小以适应显示区域
            processed_frame = cv2.resize(processed_frame, (640, 480))
            
            # 转换为Tkinter可显示的格式
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # 更新显示
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk  # 保持引用以防止被垃圾回收
        
        # 定时更新
        self.root.after(30, self.update_frame)  # 约30fps 