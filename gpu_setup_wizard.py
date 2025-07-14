import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import webbrowser
import os
import re

class GPUWizardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPU 环境安装向导")
        self.geometry("700x550")
        self.resizable(False, False)

        # 共享数据
        self.shared_data = {
            "gpu_name": tk.StringVar(),
            "driver_version": tk.StringVar(),
            "cuda_install_path": tk.StringVar(),
            "cudnn_extract_path": tk.StringVar()
        }

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (WelcomePage, DriverCheckPage, CudaPage, CudnnPage, PathConfigPage, FinalPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class WelcomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        label = ttk.Label(self, text="欢迎使用GPU环境安装向导", font=("Arial", 18, "bold"))
        label.pack(pady=40)
        
        info_text = (
            "本工具将引导您完成在Windows上配置NVIDIA GPU加速环境的全部过程。\n\n"
            "您将依次完成以下步骤：\n"
            "1. 检查NVIDIA驱动程序\n"
            "2. 下载并安装推荐版本的CUDA Toolkit\n"
            "3. 下载并解压推荐版本的cuDNN库\n"
            "4. 自动生成配置脚本以完成最终配置\n\n"
            "请点击“下一步”开始。"
        )
        info_label = ttk.Label(self, text=info_text, wraplength=500, justify="left", font=("Arial", 11))
        info_label.pack(pady=20, padx=50)

        next_button = ttk.Button(self, text="下一步", command=lambda: controller.show_frame("DriverCheckPage"))
        next_button.pack(pady=30, ipadx=10)

class DriverCheckPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = ttk.Label(self, text="步骤一：检查NVIDIA驱动", font=("Arial", 16, "bold"))
        label.pack(pady=20)
        
        check_button = ttk.Button(self, text="点击此处，开始检测", command=self.check_driver)
        check_button.pack(pady=10)
        
        self.result_text = scrolledtext.ScrolledText(self, height=15, width=80, wrap=tk.WORD, font=("Consolas", 10))
        self.result_text.pack(pady=10, padx=20)
        self.result_text.insert(tk.END, "请点击按钮开始检测...")
        self.result_text.configure(state='disabled')

        self.nav_frame = ttk.Frame(self)
        self.nav_frame.pack(pady=20)
        
        back_button = ttk.Button(self.nav_frame, text="上一步", command=lambda: controller.show_frame("WelcomePage"))
        back_button.pack(side="left", padx=20, ipadx=10)
        
        self.next_button = ttk.Button(self.nav_frame, text="下一步", command=lambda: controller.show_frame("CudaPage"), state="disabled")
        self.next_button.pack(side="right", padx=20, ipadx=10)

    def check_driver(self):
        self.result_text.configure(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "正在运行 `nvidia-smi` 命令...\n\n")
        self.result_text.update_idletasks()
        
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, encoding='utf-8', creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode == 0:
                output = result.stdout
                self.result_text.insert(tk.END, output)
                
                gpu_name_match = re.search(r'\|\s+\d+\s+(.*?)\s{2,}', output)
                driver_version_match = re.search(r'Driver Version:\s*([\d\.]+)', output)

                if gpu_name_match and driver_version_match:
                    gpu_name = gpu_name_match.group(1).strip()
                    driver_version = driver_version_match.group(1).strip()
                    self.controller.shared_data["gpu_name"].set(gpu_name)
                    self.controller.shared_data["driver_version"].set(driver_version)
                    
                    self.result_text.insert(tk.END, "\n\n--- 检测成功 ---\n")
                    self.result_text.insert(tk.END, f"显卡型号: {gpu_name}\n")
                    self.result_text.insert(tk.END, f"驱动版本: {driver_version}\n\n")
                    self.result_text.insert(tk.END, "您的驱动已正确安装，请点击“下一步”。")
                    self.next_button.configure(state="normal")
                else:
                    self.result_text.insert(tk.END, "\n\n--- 无法解析驱动信息 ---\n")
                    self.result_text.insert(tk.END, "命令已执行，但无法从中提取显卡型号或驱动版本。")
            else:
                self.result_text.insert(tk.END, f"--- 命令执行失败 ---\n错误信息:\n{result.stderr}")

        except FileNotFoundError:
            self.result_text.insert(tk.END, "--- 错误：`nvidia-smi` 命令未找到 ---\n\n")
            self.result_text.insert(tk.END, "这通常意味着您没有安装NVIDIA显卡驱动程序。\n")
            self.result_text.insert(tk.END, "请前往NVIDIA官网下载并安装与您显卡匹配的最新驱动。")
            
        self.result_text.configure(state='disabled')

class CudaPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.cuda_version = "11.8"
        self.cuda_link = "https://developer.nvidia.com/cuda-11-8-0-download-archive"

        label = ttk.Label(self, text="步骤二：下载并安装 CUDA Toolkit", font=("Arial", 16, "bold"))
        label.pack(pady=20)

        info_text = (
            f"根据本程序依赖库，我们为您推荐 CUDA Toolkit 版本: {self.cuda_version}\n\n"
            "1. 点击下方按钮打开官方下载页面。\n"
            "2. 在页面中根据您的Windows版本选择对应的安装包并下载（约2-3GB）。\n"
            "3. 下载完成后，运行安装程序。在安装选项中，推荐选择“精简(Express)”模式。\n"
            "4. 安装完成后，点击本向导的“下一步”继续。"
        )
        info_label = ttk.Label(self, text=info_text, wraplength=600, justify="left", font=("Arial", 11))
        info_label.pack(pady=20, padx=50)

        link_button = ttk.Button(self, text=f"打开 CUDA {self.cuda_version} 下载页面", command=self.open_link)
        link_button.pack(pady=10)

        nav_frame = ttk.Frame(self)
        nav_frame.pack(pady=40)
        
        back_button = ttk.Button(nav_frame, text="上一步", command=lambda: controller.show_frame("DriverCheckPage"))
        back_button.pack(side="left", padx=20, ipadx=10)
        
        next_button = ttk.Button(nav_frame, text="下一步", command=lambda: controller.show_frame("CudnnPage"))
        next_button.pack(side="right", padx=20, ipadx=10)
        
    def open_link(self):
        webbrowser.open(self.cuda_link)

class CudnnPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.cudnn_version = "8.6"
        self.cudnn_link = "https://developer.nvidia.com/rdp/cudnn-archive"

        label = ttk.Label(self, text="步骤三：下载并解压 cuDNN", font=("Arial", 16, "bold"))
        label.pack(pady=20)

        info_text = (
            f"cuDNN是用于深度学习的加速库。根据CUDA {CudaPage(parent, controller).cuda_version}，我们推荐 cuDNN 版本: {self.cudnn_version}\n\n"
            "1. 点击下方按钮打开cuDNN归档页面（需要登录NVIDIA开发者账号）。\n"
            f"2. 在页面中找到 'Download cuDNN v{self.cudnn_version} for CUDA {CudaPage(parent, controller).cuda_version}' 并点击下载。\n"
            "3. 下载的是一个ZIP压缩包，请将其解压到一个您方便找到的位置（例如，桌面）。\n"
            "4. 解压完成后，点击本向导的“下一步”继续。"
        )
        info_label = ttk.Label(self, text=info_text, wraplength=600, justify="left", font=("Arial", 11))
        info_label.pack(pady=20, padx=50)

        link_button = ttk.Button(self, text=f"打开 cuDNN v{self.cudnn_version} 下载页面", command=self.open_link)
        link_button.pack(pady=10)
        
        nav_frame = ttk.Frame(self)
        nav_frame.pack(pady=40)
        
        back_button = ttk.Button(nav_frame, text="上一步", command=lambda: controller.show_frame("CudaPage"))
        back_button.pack(side="left", padx=20, ipadx=10)
        
        next_button = ttk.Button(nav_frame, text="下一步", command=lambda: controller.show_frame("PathConfigPage"))
        next_button.pack(side="right", padx=20, ipadx=10)
        
    def open_link(self):
        webbrowser.open(self.cudnn_link)

class PathConfigPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = ttk.Label(self, text="步骤四：指定路径并生成配置脚本", font=("Arial", 16, "bold"))
        label.pack(pady=20)
        
        # CUDA Path
        cuda_frame = ttk.Frame(self)
        cuda_frame.pack(fill="x", padx=50, pady=10)
        cuda_label = ttk.Label(cuda_frame, text="1. CUDA Toolkit 安装路径:")
        cuda_label.pack(side="left")
        self.cuda_entry = ttk.Entry(cuda_frame, textvariable=self.controller.shared_data["cuda_install_path"], width=50)
        self.cuda_entry.pack(side="left", fill="x", expand=True, padx=5)
        cuda_browse_button = ttk.Button(cuda_frame, text="浏览...", command=self.browse_cuda)
        cuda_browse_button.pack(side="left")

        # CUDNN Path
        cudnn_frame = ttk.Frame(self)
        cudnn_frame.pack(fill="x", padx=50, pady=10)
        cudnn_label = ttk.Label(cudnn_frame, text="2. cuDNN 解压路径:")
        cudnn_label.pack(side="left")
        self.cudnn_entry = ttk.Entry(cudnn_frame, textvariable=self.controller.shared_data["cudnn_extract_path"], width=50)
        self.cudnn_entry.pack(side="left", fill="x", expand=True, padx=5)
        cudnn_browse_button = ttk.Button(cudnn_frame, text="浏览...", command=self.browse_cudnn)
        cudnn_browse_button.pack(side="left")
        
        # Generate Button
        generate_button = ttk.Button(self, text="生成配置脚本", command=self.generate_script)
        generate_button.pack(pady=30, ipadx=20)

        nav_frame = ttk.Frame(self)
        nav_frame.pack(pady=20)
        back_button = ttk.Button(nav_frame, text="上一步", command=lambda: controller.show_frame("CudnnPage"))
        back_button.pack(side="left", padx=20, ipadx=10)

    def browse_cuda(self):
        # Default path for CUDA
        default_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
        path = filedialog.askdirectory(initialdir=default_path, title="请选择CUDA Toolkit的安装目录")
        if path:
            self.controller.shared_data["cuda_install_path"].set(path)
            
    def browse_cudnn(self):
        path = filedialog.askdirectory(title="请选择您解压cuDNN的目录")
        if path:
            self.controller.shared_data["cudnn_extract_path"].set(path)

    def generate_script(self):
        cuda_path = self.controller.shared_data["cuda_install_path"].get()
        cudnn_path = self.controller.shared_data["cudnn_extract_path"].get()

        if not cuda_path or not cudnn_path:
            messagebox.showerror("错误", "CUDA和cuDNN的路径都不能为空！")
            return

        if not os.path.exists(os.path.join(cuda_path, "bin", "nvcc.exe")):
            messagebox.showerror("错误", f"指定的CUDA路径无效，未在其中找到 'bin/nvcc.exe'。\n\n路径: {cuda_path}")
            return
            
        required_dirs = ['bin', 'include', 'lib']
        if not all(os.path.exists(os.path.join(cudnn_path, d)) for d in required_dirs):
             messagebox.showerror("错误", f"指定的cuDNN路径无效，未在其中找到 'bin', 'include', 'lib' 子目录。\n\n路径: {cudnn_path}")
             return

        script_content = f"""@echo off
echo.
echo ====================================================================
echo ==        NVIDIA GPU 环境自动配置脚本 (管理员权限)        ==
echo ====================================================================
echo.
echo 即将执行以下操作:
echo 1. 将 cuDNN 的文件复制到 CUDA Toolkit 的对应目录中
echo.
echo    - 源 (cuDNN) : "{cudnn_path}"
echo    - 目标 (CUDA): "{cuda_path}"
echo.
pause

echo.
echo --- 正在复制 bin 文件...
xcopy /E /Y /I "{os.path.join(cudnn_path, 'bin')}\\*.*" "{os.path.join(cuda_path, 'bin')}"
echo.
echo --- 正在复制 include 文件...
xcopy /E /Y /I "{os.path.join(cudnn_path, 'include')}\\*.*" "{os.path.join(cuda_path, 'include')}"
echo.
echo --- 正在复制 lib 文件...
xcopy /E /Y /I "{os.path.join(cudnn_path, 'lib')}\\*.*" "{os.path.join(cuda_path, 'lib')}"
echo.

echo ====================================================================
echo ==                      配置成功!                               ==
echo ====================================================================
echo.
echo 您现在可以关闭此窗口，并重新启动您的姿态分析软件来使用GPU加速了。
echo.
pause
"""
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        script_path = os.path.join(desktop, "一键配置GPU环境.bat")
        
        try:
            with open(script_path, "w", encoding="gbk") as f:
                f.write(script_content)
            self.controller.show_frame("FinalPage")
        except Exception as e:
            messagebox.showerror("生成脚本失败", f"无法在桌面创建配置文件，请检查权限。\n\n错误: {e}")

class FinalPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        label = ttk.Label(self, text="大功告成！", font=("Arial", 18, "bold"))
        label.pack(pady=40)
        
        final_text = (
            "最后一步，也是最关键的一步：\n\n"
            "1. 我已在您的桌面上生成了一个名为 “一键配置GPU环境.bat” 的文件。\n\n"
            "2. 请找到这个文件，用鼠标【右键】点击它。\n\n"
            "3. 在弹出的菜单中，选择【以管理员身份运行】。\n\n"
            "4. 在弹出的黑色窗口中根据提示按任意键，等待其提示“配置成功!”即可。\n\n"
            "完成以上操作后，您就可以重启本姿态分析软件，选择GPU模式享受加速了！"
        )
        info_label = ttk.Label(self, text=final_text, wraplength=500, justify="left", font=("Arial", 12))
        info_label.pack(pady=20, padx=50)

        close_button = ttk.Button(self, text="关闭向导", command=controller.destroy)
        close_button.pack(pady=30, ipadx=10)

if __name__ == "__main__":
    app = GPUWizardApp()
    app.mainloop() 