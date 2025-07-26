#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
from datetime import datetime
from modules.pose_analyzer import PoseAnalyzer
from modules.pose_detector import PoseDetector
from modules.action_advisor import ActionAdvisor
from PIL import Image, ImageTk
import base64
from io import BytesIO
class ReportWindowTk:
    """Tkinter版本的分析报告窗口"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.window = None
        self.analyzer = PoseAnalyzer()
        self.action_advisor = ActionAdvisor()
        self.report_data = None
        
    def show(self):
        """显示报告窗口"""
        if self.window is not None:
            self.window.lift()
            self.window.focus_force()
            return
            
        self.window = tk.Toplevel(self.parent)
        self.window.title("羽毛球动作分析报告")
        self.window.geometry("900x600")
        self.window.resizable(True, True)
        
        # 窗口关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = ttk.Label(main_frame, text="羽毛球动作分析报告", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="选择分析文件")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        file_inner_frame = ttk.Frame(file_frame)
        file_inner_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 标准模板选择
        ttk.Label(file_inner_frame, text="标准模板:").grid(row=0, column=0, sticky='w', padx=(0, 5))
        self.standard_file_var = tk.StringVar(value="未选择")
        self.standard_file_label = ttk.Label(file_inner_frame, textvariable=self.standard_file_var, 
                                           foreground="gray")
        self.standard_file_label.grid(row=0, column=1, sticky='w', padx=5)
        
        self.select_standard_btn = ttk.Button(file_inner_frame, text="选择标准模板", 
                                            command=self.select_standard_file)
        self.select_standard_btn.grid(row=0, column=2, padx=5)
        
        # 用户数据选择
        ttk.Label(file_inner_frame, text="用户数据:").grid(row=1, column=0, sticky='w', padx=(0, 5), pady=(5, 0))
        self.learner_file_var = tk.StringVar(value="未选择")
        self.learner_file_label = ttk.Label(file_inner_frame, textvariable=self.learner_file_var, 
                                          foreground="gray")
        self.learner_file_label.grid(row=1, column=1, sticky='w', padx=5, pady=(5, 0))
        
        self.select_learner_btn = ttk.Button(file_inner_frame, text="选择用户数据", 
                                           command=self.select_learner_file)
        self.select_learner_btn.grid(row=1, column=2, padx=5, pady=(5, 0))
        
        # 自动分析选项
        ttk.Label(file_inner_frame, text="或选择视频:").grid(row=2, column=0, sticky='w', padx=(0, 5), pady=(5, 0))
        self.video_file_var = tk.StringVar(value="未选择")
        self.video_file_label = ttk.Label(file_inner_frame, textvariable=self.video_file_var, 
                                         foreground="gray")
        self.video_file_label.grid(row=2, column=1, sticky='w', padx=5, pady=(5, 0))
        
        self.select_video_btn = ttk.Button(file_inner_frame, text="选择视频文件", 
                                          command=self.select_video_file)
        self.select_video_btn.grid(row=2, column=2, padx=5, pady=(5, 0))
        
        # 分析按钮
        self.analyze_btn = ttk.Button(file_inner_frame, text="开始分析", 
                                    command=self.start_analysis, state="disabled")
        self.analyze_btn.grid(row=3, column=1, pady=10)
        
        # 配置列权重
        file_inner_frame.columnconfigure(1, weight=1)
        
        # 创建Notebook（选项卡）
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 分析结果选项卡
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="分析结果")
        self.create_analysis_tab()
        
        # 详细数据选项卡
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="详细数据")
        self.create_data_tab()
        
        # 底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.export_btn = ttk.Button(button_frame, text="导出报告", 
                                   command=self.export_report, state="disabled")
        self.export_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.close_btn = ttk.Button(button_frame, text="关闭", 
                                  command=self.close_window)
        self.close_btn.pack(side=tk.RIGHT)
        
    def create_analysis_tab(self):
        """创建分析结果选项卡"""
        # 状态区域
        status_frame = ttk.LabelFrame(self.analysis_frame, text="分析状态")
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="等待开始分析...")
        self.status_label.pack(pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.pack(pady=(0, 10))
        
        # 创建主要内容区域（左右分布）
        content_frame = tk.Frame(self.analysis_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：雷达图区域
        radar_frame = ttk.LabelFrame(content_frame, text="五维度评分雷达图")
        radar_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        # 雷达图显示区域
        self.radar_canvas = tk.Canvas(radar_frame, width=400, height=400, bg='white')
        self.radar_canvas.pack(padx=10, pady=10)
        
        # 维度评分文本显示
        self.dimension_text = tk.Text(radar_frame, height=8, width=50, wrap=tk.WORD, 
                                    font=('Microsoft YaHei', 9), state=tk.DISABLED)
        self.dimension_text.pack(padx=10, pady=(0, 10))
        
        # 右侧：建议区域
        suggestions_frame = ttk.LabelFrame(content_frame, text="分析建议")
        suggestions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 创建文本框和滚动条
        text_frame = ttk.Frame(suggestions_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.suggestions_text = tk.Text(text_frame, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar_y = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.suggestions_text.yview)
        self.suggestions_text.configure(yscrollcommand=scrollbar_y.set)
        
        self.suggestions_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始文本
        self.update_suggestions_text("暂无分析结果")
        self.update_dimension_text("暂无评分数据")
        
    def create_data_tab(self):
        """创建详细数据选项卡"""
        # 数据表格
        table_frame = ttk.LabelFrame(self.data_frame, text="详细对比数据")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建Treeview表格
        columns = ('项目', '标准值', '用户值', '差异')
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # 定义列标题
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=150, anchor='center')
        
        # 添加滚动条
        tree_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
    def select_standard_file(self):
        """选择标准模板文件"""
        file_path = filedialog.askopenfilename(
            title="选择标准动作模板",
            initialdir="templates/",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.standard_file_path = file_path
            filename = os.path.basename(file_path)
            self.standard_file_var.set(filename)
            self.standard_file_label.config(foreground="blue")
            self.check_files_selected()
    
    def select_learner_file(self):
        """选择用户数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择用户动作数据",
            initialdir="output/",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.learner_file_path = file_path
            filename = os.path.basename(file_path)
            self.learner_file_var.set(filename)
            self.learner_file_label.config(foreground="blue")
            # 清除视频选择
            if hasattr(self, 'video_file_path'):
                delattr(self, 'video_file_path')
                self.video_file_var.set("未选择")
                self.video_file_label.config(foreground="gray")
            self.check_files_selected()
    
    def select_video_file(self):
        """选择视频文件进行自动分析"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            initialdir="testvideo/",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.video_file_path = file_path
            filename = os.path.basename(file_path)
            self.video_file_var.set(filename)
            self.video_file_label.config(foreground="blue")
            # 清除JSON选择
            if hasattr(self, 'learner_file_path'):
                delattr(self, 'learner_file_path')
                self.learner_file_var.set("未选择")
                self.learner_file_label.config(foreground="gray")
            self.check_files_selected()
    
    def check_files_selected(self):
        """检查是否已选择所有必要文件"""
        if hasattr(self, 'standard_file_path'):
            if hasattr(self, 'learner_file_path') or hasattr(self, 'video_file_path'):
                self.analyze_btn.config(state="normal")
            else:
                self.analyze_btn.config(state="disabled")
        else:
            self.analyze_btn.config(state="disabled")
    
    def start_analysis(self):
        """开始分析"""
        if not hasattr(self, 'standard_file_path'):
            messagebox.showwarning("警告", "请先选择标准模板文件")
            return
            
        if not hasattr(self, 'learner_file_path') and not hasattr(self, 'video_file_path'):
            messagebox.showwarning("警告", "请选择用户数据文件或视频文件")
            return
        
        # 禁用分析按钮
        self.analyze_btn.config(state="disabled")
        
        # 更新状态
        if hasattr(self, 'video_file_path'):
            self.status_label.config(text="正在处理视频并分析...")
        else:
            self.status_label.config(text="正在分析中...")
        self.progress_var.set(0)
        
        # 在后台线程中执行分析
        analysis_thread = threading.Thread(target=self.perform_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def perform_analysis(self):
        """执行分析（在后台线程中）"""
        try:
            # 如果选择了视频文件，先处理视频
            if hasattr(self, 'video_file_path'):
                self.window.after(0, lambda: self.progress_var.set(10))
                
                # 初始化姿态检测器
                pose_detector = PoseDetector()
                
                self.window.after(0, lambda: self.progress_var.set(20))
                
                # 处理视频
                landmarks_data = pose_detector.analyze_video(self.video_file_path)
                
                self.window.after(0, lambda: self.progress_var.set(40))
                
                # 保存临时JSON文件
                temp_json_path = os.path.join("output", f"temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                os.makedirs("output", exist_ok=True)
                
                with open(temp_json_path, 'w', encoding='utf-8') as f:
                    json.dump(landmarks_data, f, ensure_ascii=False, indent=2)
                
                self.learner_file_path = temp_json_path
                
                self.window.after(0, lambda: self.progress_var.set(50))
            else:
                self.window.after(0, lambda: self.progress_var.set(20))
            
            # 执行分析 - 使用新的ActionAdvisor
            comprehensive_report = self.action_advisor.generate_comprehensive_advice(
                self.learner_file_path, self.standard_file_path
            )
            
            self.window.after(0, lambda: self.progress_var.set(80))
            
            # 准备报告数据
            analysis_type = '视频自动分析' if hasattr(self, 'video_file_path') else 'JSON差异分析'
            source_path = self.video_file_path if hasattr(self, 'video_file_path') else self.learner_file_path
            
            self.report_data = {
                'suggestions': comprehensive_report,
                'standard_path': self.standard_file_path,
                'learner_path': source_path,
                'analysis_type': analysis_type,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.window.after(0, lambda: self.progress_var.set(100))
            
            # 在主线程中更新UI
            self.window.after(0, self.display_results)
            
        except Exception as e:
            error_msg = f"分析过程中发生错误: {str(e)}"
            self.window.after(0, lambda: self.display_error(error_msg))
    
    def display_results(self):
        """显示分析结果"""
        if not self.report_data:
            return
        
        # 更新状态
        self.status_label.config(text=f"分析完成 - {self.report_data['timestamp']}")
        
        # 显示建议
        if isinstance(self.report_data['suggestions'], dict):
            # 新格式：包含雷达图和维度评分
            suggestions = self.report_data['suggestions']
            
            # 显示雷达图
            if 'radar_chart' in suggestions:
                self.display_radar_chart(suggestions['radar_chart'])
            
            # 显示维度评分
            if 'dimension_analysis' in suggestions:
                dimension_text = "五维度评分分析：\n\n" + "\n".join(suggestions['dimension_analysis'])
                self.update_dimension_text(dimension_text)
            
            # 显示文字建议
            if 'llm_enhanced_advice' in suggestions:
                self.update_suggestions_text(suggestions['llm_enhanced_advice'])
            elif 'stage_suggestions' in suggestions:
                suggestions_text = "\n\n".join(suggestions['stage_suggestions'])
                self.update_suggestions_text(suggestions_text)
        else:
            # 旧格式：纯文本建议
            suggestions_text = "\n\n".join(self.report_data['suggestions'])
            self.update_suggestions_text(suggestions_text)
        
        # 更新数据表格
        self.update_data_table()
        
        # 启用导出按钮和重新分析按钮
        self.export_btn.config(state="normal")
        self.analyze_btn.config(state="normal")
    
    def display_error(self, error_message):
        """显示错误信息"""
        self.status_label.config(text="分析失败")
        self.update_suggestions_text(f"错误: {error_message}")
        self.analyze_btn.config(state="normal")
        messagebox.showerror("分析错误", error_message)
    
    def update_suggestions_text(self, text):
        """更新建议文本"""
        self.suggestions_text.config(state=tk.NORMAL)
        self.suggestions_text.delete(1.0, tk.END)
        self.suggestions_text.insert(1.0, text)
        self.suggestions_text.config(state=tk.DISABLED)
    
    def update_dimension_text(self, text):
        """更新维度评分文本"""
        self.dimension_text.config(state=tk.NORMAL)
        self.dimension_text.delete(1.0, tk.END)
        self.dimension_text.insert(1.0, text)
        self.dimension_text.config(state=tk.DISABLED)
    
    def draw_radar_chart(self, scores):
        """绘制雷达图"""
        import math
        
        # 清空画布
        self.radar_canvas.delete("all")
        
        # 画布尺寸
        width = 400
        height = 400
        center_x = width // 2
        center_y = height // 2
        radius = 150
        
        # 五个维度
        dimensions = ['技术动作', '力量控制', '节奏把握', '身体协调', '稳定性']
        num_dimensions = len(dimensions)
        
        # 绘制背景网格
        for i in range(1, 6):
            r = radius * i / 5
            points = []
            for j in range(num_dimensions):
                angle = 2 * math.pi * j / num_dimensions - math.pi / 2
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.extend([x, y])
            self.radar_canvas.create_polygon(points, outline='lightgray', fill='', width=1)
        
        # 绘制维度线
        for i in range(num_dimensions):
            angle = 2 * math.pi * i / num_dimensions - math.pi / 2
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.radar_canvas.create_line(center_x, center_y, x, y, fill='lightgray', width=1)
            
            # 添加维度标签
            label_x = center_x + (radius + 20) * math.cos(angle)
            label_y = center_y + (radius + 20) * math.sin(angle)
            self.radar_canvas.create_text(label_x, label_y, text=dimensions[i], 
                                        font=('Microsoft YaHei', 10), anchor='center')
        
        # 绘制数据多边形
        if scores and len(scores) == num_dimensions:
            points = []
            for i, score in enumerate(scores):
                angle = 2 * math.pi * i / num_dimensions - math.pi / 2
                r = radius * score / 100  # 假设分数是0-100
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.extend([x, y])
            
            # 绘制填充区域
            self.radar_canvas.create_polygon(points, outline='blue', fill='lightblue', 
                                           width=2, stipple='gray25')
            
            # 绘制数据点
            for i in range(0, len(points), 2):
                x, y = points[i], points[i+1]
                self.radar_canvas.create_oval(x-4, y-4, x+4, y+4, fill='red', outline='darkred')
    
    def display_radar_chart(self, radar_chart_base64):
        """显示雷达图"""
        try:
            # 解码base64图像
            image_data = base64.b64decode(radar_chart_base64)
            image = Image.open(BytesIO(image_data))
            
            # 调整图像大小以适应画布
            image = image.resize((380, 380), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            self.radar_image = ImageTk.PhotoImage(image)
            
            # 清空画布并显示图像
            self.radar_canvas.delete("all")
            self.radar_canvas.create_image(200, 200, image=self.radar_image)
            
        except Exception as e:
            # 如果显示图像失败，显示错误信息
            self.radar_canvas.delete("all")
            self.radar_canvas.create_text(200, 200, text=f"雷达图显示失败\n{str(e)}", 
                                        font=('Microsoft YaHei', 12), anchor='center')
    
    def update_data_table(self):
        """更新数据表格"""
        # 清空现有数据
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if not self.report_data:
            return
        
        # 添加示例数据
        sample_data = [
            ("分析类型", self.report_data['analysis_type'], "-", "-"),
            ("标准模板", os.path.basename(self.report_data['standard_path']), "-", "-"),
            ("用户数据", os.path.basename(self.report_data['learner_path']), "-", "-"),
            ("建议数量", str(len(self.report_data['suggestions'])), "-", "-"),
            ("分析时间", self.report_data['timestamp'], "-", "-")
        ]
        
        for data in sample_data:
            self.data_tree.insert('', tk.END, values=data)
    
    def export_report(self):
        """导出报告"""
        if not self.report_data:
            messagebox.showwarning("警告", "没有可导出的报告数据")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存分析报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
            initialname="羽毛球分析报告.txt"
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
                
                messagebox.showinfo("导出成功", f"报告已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("导出失败", f"保存文件时发生错误: {str(e)}")
    
    def close_window(self):
        """关闭窗口"""
        if self.window:
            self.window.destroy()
            self.window = None