"""
羽毛球发球标准动作数据
包含各个关键阶段的标准姿势参数
"""

# 标准发球动作参数
STANDARD_SERVE_PARAMS = {
    'preparation': {
        'description': '准备阶段：身体略微前倾，持拍手臂弯曲，另一只手拿球',
        'right_elbow_angle': (80, 100),  # 右肘角度范围
        'right_shoulder_angle': (20, 40),  # 右肩角度范围
        'right_knee_angle': (150, 170),  # 右膝角度范围
        'body_lean_angle': (80, 100),  # 身体前倾角度范围
    },
    'backswing': {
        'description': '后摆阶段：拍头向后摆动，肘部保持弯曲，肩部角度增大',
        'right_elbow_angle': (80, 110),  # 右肘角度范围
        'right_shoulder_angle': (70, 100),  # 右肩角度范围
        'right_knee_angle': (150, 170),  # 右膝角度范围
        'body_rotation': (10, 30),  # 身体旋转角度范围
    },
    'forward_swing': {
        'description': '前摆阶段：拍头快速向前摆动，手臂伸直，击打羽毛球',
        'right_elbow_angle': (140, 180),  # 右肘角度范围
        'right_shoulder_angle': (100, 140),  # 右肩角度范围
        'right_knee_angle': (150, 170),  # 右膝角度范围
        'wrist_angle': (160, 180),  # 手腕角度范围
    },
    'follow_through': {
        'description': '随挥阶段：击球后继续完成动作，手臂向前下方挥动',
        'right_elbow_angle': (120, 160),  # 右肘角度范围
        'right_shoulder_angle': (40, 70),  # 右肩角度范围
        'right_knee_angle': (150, 170),  # 右膝角度范围
        'body_rotation': (0, 20),  # 身体旋转角度范围
    }
}

# 各阶段的关键检查点
SERVE_CHECKPOINTS = {
    'preparation': [
        '右手握拍姿势正确',
        '身体重心稍微前倾',
        '右肘适当弯曲',
        '左手持球在身体前方',
        '两脚分开与肩同宽',
    ],
    'backswing': [
        '拍头向后摆动到位',
        '右肘保持弯曲',
        '身体重心向后移动',
        '左手抛球稳定',
        '目光跟随球移动',
    ],
    'forward_swing': [
        '右臂向前摆动流畅',
        '击球点在适当高度',
        '手腕发力恰当',
        '身体重心前移',
        '击球时机准确',
    ],
    'follow_through': [
        '击球后动作完整',
        '手臂自然向前下方随挥',
        '身体恢复平衡',
        '目光跟随球飞行方向',
        '准备下一步移动',
    ]
}

def get_standard_serve_params():
    """获取标准发球动作参数"""
    return STANDARD_SERVE_PARAMS

def get_serve_checkpoints():
    """获取发球动作检查点"""
    return SERVE_CHECKPOINTS 