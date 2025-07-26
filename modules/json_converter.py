import os
import json
import glob
import requests
from typing import List, Dict, Any

class JsonConverter:
    """
    JSON智能体：自动将output文件夹中的原始JSON转换为staged_templates格式
    """
    
    def __init__(self, output_dir="d:\\羽毛球项目\\output", 
                 staged_dir="d:\\羽毛球项目\\staged_templates",
                 template_path="d:\\羽毛球项目\\staged_templates\\击球动作模板.json"):
        """
        初始化JSON转换器
        
        Args:
            output_dir: 原始JSON文件目录
            staged_dir: 输出staged JSON的目录
            template_path: 参考模板路径
        """
        self.output_dir = output_dir
        self.staged_dir = staged_dir
        self.template_path = template_path
        self.api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.api_key = os.environ.get('VOLCENGINE_API_KEY', '')
        
    def get_latest_output_json(self) -> str:
        """
        获取output文件夹中最新的JSON文件
        
        Returns:
            最新JSON文件的完整路径
        """
        json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError("output文件夹中没有找到JSON文件")
        
        # 按修改时间排序，返回最新的
        latest_file = max(json_files, key=os.path.getmtime)
        return latest_file
    
    def load_template(self) -> List[Dict[str, Any]]:
        """
        加载staged模板文件
        
        Returns:
            模板数据
        """
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"加载模板文件失败: {e}")
    
    def convert_to_staged_format(self, input_json_path: str, output_filename: str = None) -> str:
        """
        将原始JSON转换为staged格式
        
        Args:
            input_json_path: 输入的原始JSON文件路径
            output_filename: 输出文件名（可选）
            
        Returns:
            输出文件的完整路径
        """
        try:
            # 加载原始数据和模板
            with open(input_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            template_data = self.load_template()
            
            # 生成输出文件名
            if not output_filename:
                base_name = os.path.basename(input_json_path)
                output_filename = f"staged_{base_name}"
            
            output_path = os.path.join(self.staged_dir, output_filename)
            
            # 确保输出目录存在
            os.makedirs(self.staged_dir, exist_ok=True)
            
            # 使用LLM进行转换
            staged_data = self._convert_with_llm(raw_data, template_data)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(staged_data, f, ensure_ascii=False, indent=2)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"转换失败: {e}")
    
    def _convert_with_llm(self, raw_data: List[Dict], template_data: List[Dict]) -> List[Dict]:
        """
        使用LLM将原始数据转换为staged格式
        
        Args:
            raw_data: 原始JSON数据
            template_data: 模板数据
            
        Returns:
            转换后的staged格式数据
        """
        # 分批处理大数据
        batch_size = 300
        staged_result = []
        
        # 准备模板示例（只取前2个阶段作为示例）
        template_example = template_data[:2] if len(template_data) >= 2 else template_data
        template_str = json.dumps(template_example, ensure_ascii=False, indent=2)
        
        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i+batch_size]
            
            # 构建提示词
            prompt = f"""
你是一个专业的羽毛球动作分析专家。请将用户提供的原始JSON数据转换为阶段化的分析格式。

任务要求：
1. 分析原始数据中的time_ms和landmarks信息
2. 将动作分为5个阶段：准备、移动/接近、后摆、击球/前挥、收势
3. 参考以下模板格式输出：

模板示例：
{template_str}

严格要求：
1. 必须包含完整的5个阶段：准备、移动/接近、后摆、击球/前挥、收势
2. 每个阶段必须包含：stage, start_ms, end_ms, description, expected_values, key_landmarks
3. expected_values必须包含关键角度的min/max/ideal值（参考模板中的数值范围）
4. key_landmarks选择每个阶段的代表性时间点
5. 只保留关键landmarks点：0(鼻子), 4(左手腕), 7(右手腕), 8(左髋), 11(右髋)
6. 时间分配建议：准备(0-20%)、移动/接近(20-40%)、后摆(40-60%)、击球/前挥(60-80%)、收势(80-100%)
7. 输出必须是完整的JSON数组格式，包含所有5个阶段

原始数据批次：
{json.dumps(batch[:50], ensure_ascii=False)}  

请直接输出包含5个阶段的完整JSON数组：
"""
            
            try:
                # 调用火山引擎豆包API
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": "doubao-seed-1-6-thinking-250715",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "top_p": 0.9
                }
                
                # 禁用代理
                proxies = {
                    'http': None,
                    'https': None
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=300,  # 5分钟超时
                    proxies=proxies
                )
                response.raise_for_status()
                
                result = response.json()
                response_content = result['choices'][0]['message']['content']
                
                # 清理响应内容，提取JSON部分
                json_start = response_content.find('[')
                json_end = response_content.rfind(']') + 1
                
                if json_start != -1 and json_end != -1:
                    json_content = response_content[json_start:json_end]
                    batch_stages = json.loads(json_content)
                    
                    # 验证和修正数据
                    validated_stages = self._validate_stages(batch_stages)
                    staged_result.extend(validated_stages)
                else:
                    # 如果无法提取JSON，使用默认结构
                    default_stages = self._create_default_stages(batch)
                    staged_result.extend(default_stages)
                    
            except Exception as e:
                print(f"LLM转换失败，使用默认结构: {e}")
                default_stages = self._create_default_stages(batch)
                staged_result.extend(default_stages)
        
        # 合并重复阶段
        return self._merge_stages(staged_result)
    
    def _validate_stages(self, stages: List[Dict]) -> List[Dict]:
        """
        验证和修正阶段数据
        """
        required_fields = ['stage', 'start_ms', 'end_ms', 'description', 'expected_values', 'key_landmarks']
        valid_stages = []
        
        for stage in stages:
            if isinstance(stage, dict) and all(field in stage for field in required_fields):
                valid_stages.append(stage)
        
        return valid_stages
    
    def _create_default_stages(self, batch_data: List[Dict]) -> List[Dict]:
        """
        创建默认的阶段结构
        """
        if not batch_data:
            return []
        
        start_time = batch_data[0].get('time_ms', 0)
        end_time = batch_data[-1].get('time_ms', 1000)
        duration = end_time - start_time
        
        # 计算每个阶段的时间分配
        stage_durations = [0.2, 0.2, 0.2, 0.2, 0.2]  # 每个阶段占20%
        
        stages = [
            {
                "stage": "准备",
                "start_ms": start_time,
                "end_ms": start_time + duration * stage_durations[0],
                "description": "初始准备，连续动作起点，身体调整。",
                "expected_values": {
                    "elbow_angle": {"min": 88, "max": 118, "ideal": 103},
                    "shoulder_angle": {"min": 28, "max": 58, "ideal": 43},
                    "knee_angle": {"min": 128, "max": 158, "ideal": 143},
                    "body_lean": {"min": 68, "max": 88, "ideal": 78}
                },
                "key_landmarks": [self._extract_key_landmarks(batch_data[0])]
            },
            {
                "stage": "移动/接近",
                "start_ms": start_time + duration * sum(stage_durations[:1]),
                "end_ms": start_time + duration * sum(stage_durations[:2]),
                "description": "连续移动，接近球点，保持流畅。",
                "expected_values": {
                    "elbow_angle": {"min": 98, "max": 128, "ideal": 113},
                    "shoulder_angle": {"min": 48, "max": 78, "ideal": 63},
                    "knee_angle": {"min": 138, "max": 168, "ideal": 153},
                    "body_rotation": {"min": 3, "max": 23, "ideal": 13}
                },
                "key_landmarks": [self._extract_key_landmarks(batch_data[len(batch_data)//5]) if len(batch_data) > 5 else self._extract_key_landmarks(batch_data[0])]
            },
            {
                "stage": "后摆",
                "start_ms": start_time + duration * sum(stage_durations[:2]),
                "end_ms": start_time + duration * sum(stage_durations[:3]),
                "description": "在连续动作中后摆蓄力，身体协调转动。",
                "expected_values": {
                    "elbow_angle": {"min": 78, "max": 108, "ideal": 93},
                    "shoulder_angle": {"min": 58, "max": 88, "ideal": 73},
                    "knee_angle": {"min": 118, "max": 138, "ideal": 128},
                    "body_lean": {"min": 58, "max": 78, "ideal": 68}
                },
                "key_landmarks": [self._extract_key_landmarks(batch_data[len(batch_data)*2//5]) if len(batch_data) > 10 else self._extract_key_landmarks(batch_data[0])]
            },
            {
                "stage": "击球/前挥",
                "start_ms": start_time + duration * sum(stage_durations[:3]),
                "end_ms": start_time + duration * sum(stage_durations[:4]),
                "description": "连续挥拍击球，发力流畅，身体前倾。",
                "expected_values": {
                    "elbow_angle": {"min": 148, "max": 178, "ideal": 163},
                    "shoulder_angle": {"min": 88, "max": 118, "ideal": 103},
                    "knee_angle": {"min": 138, "max": 158, "ideal": 148},
                    "wrist_angle": {"min": 158, "max": 178, "ideal": 168}
                },
                "key_landmarks": [self._extract_key_landmarks(batch_data[len(batch_data)*3//5]) if len(batch_data) > 15 else self._extract_key_landmarks(batch_data[0])]
            },
            {
                "stage": "收势",
                "start_ms": start_time + duration * sum(stage_durations[:4]),
                "end_ms": end_time,
                "description": "连续动作收尾，快速恢复，准备下一次击球。",
                "expected_values": {
                    "elbow_angle": {"min": 88, "max": 118, "ideal": 103},
                    "shoulder_angle": {"min": 28, "max": 58, "ideal": 43},
                    "knee_angle": {"min": 128, "max": 158, "ideal": 143},
                    "body_rotation": {"min": 2, "max": 18, "ideal": 10}
                },
                "key_landmarks": [self._extract_key_landmarks(batch_data[-1])]
            }
        ]
        
        return stages
    
    def _extract_key_landmarks(self, frame_data: Dict) -> Dict:
        """
        提取关键landmarks点
        """
        landmarks = frame_data.get('landmarks', {})
        key_points = ['0', '4', '7', '8', '11']
        
        result = {
            "time_ms": frame_data.get('time_ms', 0),
            "landmarks": {}
        }
        
        for point in key_points:
            if point in landmarks:
                result['landmarks'][point] = {
                    "x": landmarks[point].get('x', 0),
                    "y": landmarks[point].get('y', 0)
                }
        
        return result
    
    def _merge_stages(self, stages: List[Dict]) -> List[Dict]:
        """
        合并重复的阶段
        """
        if not stages:
            return []
        
        # 按阶段名称分组
        stage_groups = {}
        for stage in stages:
            stage_name = stage.get('stage', '未知')
            if stage_name not in stage_groups:
                stage_groups[stage_name] = []
            stage_groups[stage_name].append(stage)
        
        # 合并每个组的第一个阶段
        merged_stages = []
        stage_order = ['准备', '移动/接近', '后摆', '击球/前挥', '收势']
        
        for stage_name in stage_order:
            if stage_name in stage_groups and stage_groups[stage_name]:
                merged_stages.append(stage_groups[stage_name][0])
        
        return merged_stages
    
    def auto_convert_latest(self) -> str:
        """
        自动转换最新的output JSON文件
        
        Returns:
            输出文件路径
        """
        latest_file = self.get_latest_output_json()
        print(f"找到最新文件: {latest_file}")
        
        output_path = self.convert_to_staged_format(latest_file)
        print(f"转换完成，输出文件: {output_path}")
        
        return output_path
    
    def convert_all_output_files(self) -> List[str]:
        """
        转换output文件夹中的所有JSON文件
        
        Returns:
            所有输出文件路径列表
        """
        json_files = glob.glob(os.path.join(self.output_dir, "*.json"))
        output_paths = []
        
        for json_file in json_files:
            try:
                output_path = self.convert_to_staged_format(json_file)
                output_paths.append(output_path)
                print(f"已转换: {json_file} -> {output_path}")
            except Exception as e:
                print(f"转换失败 {json_file}: {e}")
        
        return output_paths

# 使用示例
if __name__ == "__main__":
    converter = JsonConverter()
    
    try:
        # 自动转换最新文件
        result = converter.auto_convert_latest()
        print(f"转换成功: {result}")
    except Exception as e:
        print(f"转换失败: {e}")