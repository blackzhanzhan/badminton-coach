import os
import urllib.request
import sys
import shutil
import requests
import time

def download_file(url, destination, use_requests=False):
    """
    下载文件并显示进度条
    
    Args:
        url: 下载链接
        destination: 保存路径
        use_requests: 是否使用requests库下载
    """
    print(f"正在下载 {os.path.basename(destination)}...")
    
    if use_requests:
        # 使用requests库下载大文件
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
                            sys.stdout.write(f"\r下载进度: {percent}% [{downloaded} / {total_size}]")
                            sys.stdout.flush()
                
                print(f"\n下载完成: {destination}")
                return True
        except Exception as e:
            print(f"\n使用requests下载失败: {e}")
            return False
    else:
        # 使用urllib下载
        try:
            def progress_bar(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
                sys.stdout.write(f"\r下载进度: {percent}% [{count * block_size} / {total_size}]")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, destination, progress_bar)
            print(f"\n下载完成: {destination}")
            return True
        except Exception as e:
            print(f"\n使用urllib下载失败: {e}")
            return False

def main():
    """主函数"""
    # 创建模型目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 定义模型文件URL和目标路径 (包括国内镜像源)
    models = {
        # prototxt文件 (模型结构定义)
        "pose_deploy_linevec.prototxt": [
            # 主要链接
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt",
            # 国内镜像链接
            "https://gitee.com/mirrors/openpose/raw/master/models/pose/mpi/pose_deploy_linevec.prototxt",
            # 备用链接
            "https://raw.fastgit.org/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt"
        ],
        
        # caffemodel文件 (模型权重)
        "pose_iter_440000.caffemodel": [
            # 主要链接 - MPI模型
            "https://pan.baidu.com/s/1UM-1JGgseixBpUzXUvN-hw?pwd=1234",  # 百度网盘链接 (需手动下载)
            # 备用链接 - COCO模型 (更准确但更大)
            "https://pan.baidu.com/s/1UM-1JGgseixBpUzXUvN-hw?pwd=1234"   # 百度网盘链接 (需手动下载)
        ]
    }
    
    # 尝试下载模型文件
    for filename, urls in models.items():
        destination = os.path.join(model_dir, filename)
        
        # 如果文件已存在，检查文件大小是否合理
        if os.path.exists(destination):
            file_size = os.path.getsize(destination)
            if filename.endswith('.prototxt') and file_size > 100:  # prototxt文件通常很小
                print(f"{filename} 已存在，跳过下载")
                continue
            elif filename.endswith('.caffemodel') and file_size > 10000000:  # caffemodel文件通常很大 (>10MB)
                print(f"{filename} 已存在，跳过下载")
                continue
            else:
                print(f"{filename} 已存在但可能不完整，尝试重新下载")
        
        # 尝试从多个源下载
        success = False
        for url in urls:
            if "pan.baidu.com" in url:
                # 百度网盘链接需要手动下载
                print(f"请手动从百度网盘下载 {filename}:")
                print(f"链接: {url}")
                if filename == "pose_iter_440000.caffemodel":
                    print("提取码: 1234")
                    print("下载后请将文件放到以下目录:")
                    print(f"  {model_dir}")
                continue
            
            print(f"尝试从 {url} 下载...")
            # 对大文件使用requests库下载
            use_requests = filename.endswith('.caffemodel')
            if download_file(url, destination, use_requests):
                success = True
                break
            else:
                print(f"从 {url} 下载失败，尝试下一个源...")
                # 等待一会儿再尝试下一个源
                time.sleep(1)
        
        if not success and not os.path.exists(destination):
            print(f"\n警告: 无法下载 {filename}，请尝试手动下载。")
    
    # 提供备用模型信息
    print("\n=== 模型下载信息 ===")
    print("如果自动下载失败，请尝试以下方法:")
    print("1. 访问以下链接手动下载模型文件:")
    print("   - pose_deploy_linevec.prototxt (模型结构):")
    print("     https://gitee.com/mirrors/openpose/raw/master/models/pose/mpi/pose_deploy_linevec.prototxt")
    print("   - pose_iter_440000.caffemodel (模型权重):")
    print("     百度网盘: https://pan.baidu.com/s/1UM-1JGgseixBpUzXUvN-hw 提取码: 1234")
    print(f"2. 将下载的文件放到 {model_dir} 目录下")
    print("\n3. 或者使用其他预训练模型:")
    print("   - MediaPipe Pose (更轻量级): https://google.github.io/mediapipe/solutions/pose.html")
    print("   - PoseNet (浏览器运行): https://github.com/tensorflow/tfjs-models/tree/master/posenet")
    
    # 检查模型文件是否存在
    prototxt_path = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
    caffemodel_path = os.path.join(model_dir, "pose_iter_440000.caffemodel")
    
    if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
        print("\n✅ 所有模型文件已准备就绪!")
    else:
        print("\n⚠️ 部分模型文件缺失，系统将使用备用检测方法。")

if __name__ == "__main__":
    main() 