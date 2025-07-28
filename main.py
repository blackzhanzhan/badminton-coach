import tkinter as tk
import os
import sys
from ui.main_window_tk import MainWindow

def setup_environment():
    """设置环境和目录结构"""
    # 创建必要的目录
    directories = [
        "models/ml_models",
        "data/feedback", 
        "data/staged_templates",
        "output",
        "templates",
        "staged"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ 环境设置完成")

if __name__ == "__main__":
    try:
        # 设置环境
        setup_environment()
        
        # 启动GUI应用
        root = tk.Tk()
        root.title("羽毛球接球动作纠正系统")
        app = MainWindow(root)
        
        print("🚀 羽毛球动作分析系统启动")
        print("📊 支持传统规则分析 + 机器学习增强分析")
        
        root.mainloop()
        
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        sys.exit(1)