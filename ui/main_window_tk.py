import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import threading
import os
import json # 引入json模块用于保存调试数据
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
        
        # 线程相关
        self.thread = None
        
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
        analysis_modes = [mode.value for mode in PoseAnalyzer.AnalysisMode]
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
        
        self.is_running = True
        
        if self.is_camera:
            # --- 实时摄像头处理流程 ---
            self.start_button["text"] = "停止"
            self.disable_controls()
            self.update_frame()  # 启动实时分析循环
        else:
            # --- 视频文件后台处理流程 ---
            self.start_button["text"] = "取消分析"
            self.disable_controls()
            # 在后台线程中启动视频处理
            self.thread = threading.Thread(target=self.process_video_background, args=(self.video_path,), daemon=True)
            self.thread.start()
    
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
        self._reset_ui_state()
        self.update_feedback_box("分析已停止。")

    def process_video_background(self, video_path):
        """在后台线程中处理整个视频文件"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.root.after(0, self.update_feedback_box, f"错误: 无法在后台打开视频 '{os.path.basename(video_path)}'")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration_sec = total_frames / fps if fps > 0 else 0

            processed_frames = 0
            detected_poses_count = 0
            # 存储所有检测到的姿态数据和时间戳，为动作切分做准备
            all_landmarks_timeline = []
            
            initial_message = f"开始分析视频...\n文件: {os.path.basename(video_path)}\n总帧数: {total_frames}"
            self.root.after(0, self.update_feedback_box, initial_message)

            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频结束

                processed_frames += 1
                
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                _, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)

                if landmarks:
                    detected_poses_count += 1
                    # 将有用的数据存入时间线
                    all_landmarks_timeline.append({'time_ms': timestamp_ms, 'landmarks': landmarks})
                
                # 每处理100帧更新一次进度
                if processed_frames % 100 == 0:
                    progress_percent = (processed_frames / total_frames) * 100
                    progress_message = f"分析中... {processed_frames}/{total_frames} ({progress_percent:.1f}%)"
                    # 这里不用追加，只更新进度行
                    self.feedback_text.delete("2.0", tk.END)
                    self.feedback_text.insert("2.0", progress_message)


            cap.release()
            self.root.after(0, self.update_feedback_box, "--- 后台线程: 视频处理循环结束 ---")
            
            if self.is_running:
                self.root.after(0, self.update_feedback_box, "--- 后台线程: is_running为True, 准备生成总结报告 ---")
                
                summary_message = (
                    f"视频分析完毕！\n\n"
                    f"文件: {os.path.basename(video_path)}\n"
                    f"时长: {video_duration_sec:.2f} 秒\n"
                    f"总帧数: {total_frames}\n"
                    f"处理帧数: {processed_frames}\n"
                    f"检测到姿态的帧数: {detected_poses_count}\n"
                    f"已收集 {len(all_landmarks_timeline)} 条有效姿态数据点。"
                )

                # 分析完成后，总是将收集到的数据保存到json文件
                self.root.after(0, self.update_feedback_box, "--- 后台线程: 准备保存JSON分析文件 ---")
                try:
                    # 定义并创建专有的输出文件夹
                    output_dir = "output"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    video_filename = os.path.basename(video_path)
                    report_filename = f"{video_filename}.analysis_data.json"
                    report_filepath = os.path.join(output_dir, report_filename)

                    with open(report_filepath, 'w', encoding='utf-8') as f:
                        json.dump(all_landmarks_timeline, f, ensure_ascii=False, indent=4)
                    
                    # 使用完整路径，让用户更清楚文件位置
                    summary_message += f"\n\n姿态数据已保存至:\n{report_filepath}"
                    self.root.after(0, self.update_feedback_box, "--- 后台线程: JSON文件保存成功 ---")
                except Exception as e:
                    summary_message += f"\n\n保存分析数据时出错: {e}"
                    self.root.after(0, self.update_feedback_box, f"--- 后台线程: JSON文件保存失败: {e} ---")

                # 最终报告使用清空式更新
                self.root.after(0, lambda: (self.feedback_text.delete(1.0, tk.END), self.feedback_text.insert(1.0, summary_message)))
            else:
                self.root.after(0, self.update_feedback_box, "--- 后台线程: is_running为False, 跳过报告生成 ---")

        except Exception as e:
            error_msg = f"视频处理过程中发生错误: {e}"
            self.root.after(0, self.update_feedback_box, error_msg)
        finally:
            # 确保分析结束后（无论成功、失败还是取消），都重置UI状态
            self.root.after(0, self.update_feedback_box, "--- 后台线程: 即将调用_reset_ui_state ---")
            self.root.after(0, self._reset_ui_state)

    def update_feedback_box(self, message):
        """安全地从任何线程更新反馈文本框的内容 (追加模式用于调试)"""
        self.feedback_text.insert(tk.END, message + "\n")

    def update_frame(self):
        """更新视频帧 (仅用于实时摄像头模式)"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            if not self.is_camera:
                self.update_feedback_box("视频播放完毕")
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
            
            # 调用新的统一分析接口
            feedback_list = self.pose_analyzer.analyze_pose(landmarks)
            feedback_str = "\n".join(feedback_list)
            
            self.update_feedback_box(feedback_str)

        else:
            # 1. 核心姿态检测 (加入时间戳)
            timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            processed_frame, landmarks = self.pose_detector.detect_pose(frame, timestamp_ms=timestamp_ms)

            # 2. 调用新的统一分析接口获取反馈
            feedback_list = self.pose_analyzer.analyze_pose(landmarks)
            feedback_str = "\n".join(feedback_list)
            
            self.update_feedback_box(feedback_str)
            
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