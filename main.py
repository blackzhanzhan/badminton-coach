import sys
import tkinter as tk
from ui.main_window_tk import MainWindow

def main():
    """主程序入口"""
    root = tk.Tk()
    root.title("羽毛球发球动作纠正系统")
    app = MainWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main() 