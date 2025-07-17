import numpy as np
from modules.pose_detector import PoseDetector

class BadmintonAnalyzer:
    def __init__(self):
        """初始化羽毛球动作分析器"""
        self.pose_detector = PoseDetector()
        
        # 关键点索引定义 - 使用与PoseDetector相同的索引
        self.NOSE = 0
        self.NECK = 1
        self.RIGHT_SHOULDER = 2
        self.RIGHT_ELBOW = 3
        self.RIGHT_WRIST = 4
        self.LEFT_SHOULDER = 5
        self.LEFT_ELBOW = 6
        self.LEFT_WRIST = 7
        self.RIGHT_HIP = 8
        self.RIGHT_KNEE = 9
        self.RIGHT_ANKLE = 10
        self.LEFT_HIP = 11
        self.LEFT_KNEE = 12
        self.LEFT_ANKLE = 13
        
        # 接球动作阶段
        self.receive_stage = 0
        self.prev_stage = 0
        self.stage_count = 0
        
        # 接球标准参数
        self.standard_params = {
            'preparation': {
                'right_elbow_angle': (90, 120),
                'right_shoulder_angle': (30, 50),
                'right_knee_angle': (130, 150),
            },
            'approach': {
                'right_elbow_angle': (100, 130),
                'right_shoulder_angle': (50, 80),
            },
            'hit': {
                'right_elbow_angle': (150, 180),
                'right_shoulder_angle': (90, 120),
            },
            'recovery': {
                'right_elbow_angle': (90, 120),
                'right_shoulder_angle': (30, 50),
            }
        }
        
    def analyze_receive(self, landmarks):
        """
        分析接球动作
        
        Args:
            landmarks: 姿势关键点数据
            
        Returns:
            feedback: 动作反馈信息
        """
        if not landmarks:
            return "未检测到人体姿势"
        
        # 检查是否有足够的关键点
        required_points = [self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST, 
                          self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE]
        if not all(point in landmarks for point in required_points):
            return "未能检测到完整的人体姿势，请调整站位"
        
        # 计算关键角度
        right_elbow_angle = self.pose_detector.get_angle(
            landmarks, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST)
        
        right_shoulder_angle = self.pose_detector.get_angle(
            landmarks, self.RIGHT_HIP, self.RIGHT_SHOULDER, self.RIGHT_ELBOW)
        
        right_knee_angle = self.pose_detector.get_angle(
            landmarks, self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE)
        
        # 检测接球阶段
        current_stage = self.detect_receive_stage(landmarks, right_elbow_angle, right_shoulder_angle)
        
        # 如果阶段发生变化，重置计数器
        if current_stage != self.prev_stage:
            self.stage_count = 0
        else:
            self.stage_count += 1
        
        # 只有当同一阶段连续出现多次时才更新当前阶段（减少抖动）
        if self.stage_count > 5:
            self.receive_stage = current_stage
        
        self.prev_stage = current_stage
        
        # 根据当前阶段生成反馈
        feedback = self.generate_feedback(
            self.receive_stage, right_elbow_angle, right_shoulder_angle, right_knee_angle)
        
        return feedback
    
    def detect_receive_stage(self, landmarks, elbow_angle, shoulder_angle):
        """
        检测接球动作的当前阶段
        
        Args:
            landmarks: 姿势关键点数据
            elbow_angle: 右肘角度
            shoulder_angle: 右肩角度
            
        Returns:
            stage: 动作阶段 (0: 准备, 1: 后摆, 2: 前摆, 3: 随挥)
        """
        if elbow_angle is None or shoulder_angle is None:
            return 0
        
        # 获取右手腕的高度
        right_wrist_y = landmarks[self.RIGHT_WRIST]['y'] if self.RIGHT_WRIST in landmarks else 0
        right_shoulder_y = landmarks[self.RIGHT_SHOULDER]['y'] if self.RIGHT_SHOULDER in landmarks else 0
        
        # 准备阶段：肘部弯曲，肩部角度较小
        if 90 <= elbow_angle <= 120 and 30 <= shoulder_angle <= 50:
            return 0
        
        # 后摆阶段：肘部弯曲，肩部角度增大
        elif 100 <= elbow_angle <= 130 and 50 <= shoulder_angle <= 80:
            return 1
        
        # 前摆阶段：肘部伸直，肩部角度大
        elif 150 <= elbow_angle <= 180 and 90 <= shoulder_angle <= 120:
            return 2
        
        # 随挥阶段：肘部稍弯，肩部角度减小，手腕位置低于肩部
        elif 90 <= elbow_angle <= 120 and 30 <= shoulder_angle <= 50 and right_wrist_y > right_shoulder_y:
            return 3
        
        # 默认为准备阶段
        return 0
    
    def generate_feedback(self, stage, elbow_angle, shoulder_angle, knee_angle):
        """
        根据当前阶段和关键角度生成反馈信息
        
        Args:
            stage: 动作阶段
            elbow_angle: 右肘角度
            shoulder_angle: 右肩角度
            knee_angle: 右膝角度
            
        Returns:
            feedback: 反馈信息
        """
        stage_names = ["准备阶段", "后摆阶段", "前摆阶段", "随挥阶段"]
        feedback = f"当前阶段: {stage_names[stage]}\n\n"
        
        if elbow_angle is None or shoulder_angle is None or knee_angle is None:
            return feedback + "无法获取完整姿势数据，请调整站位"
        
        feedback += f"右肘角度: {elbow_angle:.1f}°\n"
        feedback += f"右肩角度: {shoulder_angle:.1f}°\n"
        feedback += f"右膝角度: {knee_angle:.1f}°\n\n"
        
        # 根据不同阶段给出具体反馈
        if stage == 0:  # 准备阶段
            std_elbow = self.standard_params['preparation']['right_elbow_angle']
            std_shoulder = self.standard_params['preparation']['right_shoulder_angle']
            std_knee = self.standard_params['preparation']['right_knee_angle']
            
            feedback += "准备姿势反馈:\n"
            
            if not (std_elbow[0] <= elbow_angle <= std_elbow[1]):
                if elbow_angle < std_elbow[0]:
                    feedback += "- 右肘弯曲过度，请稍微放松\n"
                else:
                    feedback += "- 右肘弯曲不足，请适当弯曲\n"
            
            if not (std_shoulder[0] <= shoulder_angle <= std_shoulder[1]):
                if shoulder_angle < std_shoulder[0]:
                    feedback += "- 右肩角度过小，请抬高手臂\n"
                else:
                    feedback += "- 右肩角度过大，请降低手臂\n"
            
            if not (std_knee[0] <= knee_angle <= std_knee[1]):
                if knee_angle < std_knee[0]:
                    feedback += "- 右膝弯曲过度，请稍微伸直\n"
                else:
                    feedback += "- 右膝弯曲不足，请适当弯曲\n"
                    
        elif stage == 1:  # 后摆阶段
            std_elbow = self.standard_params['approach']['right_elbow_angle']
            std_shoulder = self.standard_params['approach']['right_shoulder_angle']
            
            feedback += "后摆动作反馈:\n"
            
            if not (std_elbow[0] <= elbow_angle <= std_elbow[1]):
                if elbow_angle < std_elbow[0]:
                    feedback += "- 后摆时右肘弯曲过度\n"
                else:
                    feedback += "- 后摆时右肘弯曲不足\n"
            
            if not (std_shoulder[0] <= shoulder_angle <= std_shoulder[1]):
                if shoulder_angle < std_shoulder[0]:
                    feedback += "- 后摆幅度不足，请增大摆动幅度\n"
                else:
                    feedback += "- 后摆幅度过大，请控制摆动幅度\n"
                    
        elif stage == 2:  # 前摆阶段
            std_elbow = self.standard_params['hit']['right_elbow_angle']
            std_shoulder = self.standard_params['hit']['right_shoulder_angle']
            
            feedback += "前摆动作反馈:\n"
            
            if not (std_elbow[0] <= elbow_angle <= std_elbow[1]):
                if elbow_angle < std_elbow[0]:
                    feedback += "- 前摆时右肘未充分伸展，请伸直手臂\n"
                else:
                    feedback += "- 前摆时右肘过度伸展，注意控制\n"
            
            if not (std_shoulder[0] <= shoulder_angle <= std_shoulder[1]):
                if shoulder_angle < std_shoulder[0]:
                    feedback += "- 前摆幅度不足，无法产生足够力量\n"
                else:
                    feedback += "- 前摆幅度过大，可能导致控制不稳\n"
                    
        elif stage == 3:  # 随挥阶段
            std_elbow = self.standard_params['recovery']['right_elbow_angle']
            std_shoulder = self.standard_params['recovery']['right_shoulder_angle']
            
            feedback += "随挥动作反馈:\n"
            
            if not (std_elbow[0] <= elbow_angle <= std_elbow[1]):
                if elbow_angle < std_elbow[0]:
                    feedback += "- 随挥时右肘弯曲过度，影响动作流畅性\n"
                else:
                    feedback += "- 随挥时右肘过度伸直，缺乏缓冲\n"
            
            if not (std_shoulder[0] <= shoulder_angle <= std_shoulder[1]):
                if shoulder_angle < std_shoulder[0]:
                    feedback += "- 随挥不充分，应继续完成动作\n"
                else:
                    feedback += "- 随挥过度，可能影响下一次准备\n"
        
        # 如果没有具体问题，给予积极反馈
        if feedback.count('-') == 0:
            feedback += "动作执行良好，请保持！"
            
        return feedback 