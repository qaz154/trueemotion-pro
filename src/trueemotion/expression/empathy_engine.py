# -*- coding: utf-8 -*-
"""
EmpathyEngine - 共情回应引擎

不是机械地安慰，而是真正理解用户的感受，
生成"感同身受"式的回应。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class EmpathyResponse:
    """共情回应结构"""
    text: str
    empathy_type: str  # "understanding", "venting", "support", "action"
    intensity_level: str  # "high", "medium", "low"
    follow_up_suggestion: Optional[str] = None  # 后续引导建议


class EmpathyEngine:
    """
    共情引擎

    核心原则：共情不是安慰，是"我懂你"

    共情类型：
    - understanding: 表达理解（"我懂你的感受"）
    - venting: 让用户发泄（"确实气人，换我我也急"）
    - support: 表达支持（"我在，有什么可以帮你的"）
    - action: 引导行动（"咱们想想怎么办"）

    每种情感有3种强度的高质量共情模板
    """

    # 15种情感 × 3种强度 = 45组共情模板
    EMPATHY_RESPONSES = {
        "joy": {
            "high": [
                "太为你高兴了！说说怎么回事！",
                "哇塞！真心替你开心！",
                "太棒了！感觉你肯定特别激动！",
                "恭喜恭喜！这种时刻太难得了！",
            ],
            "medium": [
                "真替你开心！",
                "挺好的，祝贺你！",
                "不错不错，继续保持！",
            ],
            "low": [
                "挺好的～",
                "嗯，挺好的",
                "还行吧",
            ],
        },
        "sadness": {
            "high": [
                "我能感受到你很难过...想聊聊吗？",
                "心疼你...有些事儿确实让人难受",
                "哎...这种时刻最难熬了，我在呢",
                "我能理解你现在的感受，真的",
            ],
            "medium": [
                "心疼你...慢慢说",
                "哎...会好起来的",
                "能理解你的难过",
            ],
            "low": [
                "嗯...",
                "明白",
                "了解",
            ],
        },
        "anger": {
            "high": [
                "确实太气人了！换我我也生气！",
                "太坑了！这事儿搁谁都会急！",
                "我懂，换我我也忍不了！",
                "气死我了！说说怎么回事！",
            ],
            "medium": [
                "能理解你有多气",
                "确实让人来火",
                "哎，别往心里去，但理解你",
            ],
            "low": [
                "呃...",
                "好吧",
                "这样啊",
            ],
        },
        "fear": {
            "high": [
                "听起来真的很担心...我懂",
                "这种事确实让人心里没底，你不是一个人",
                "害怕是正常的，我能理解你的担心",
                "嗯...先别急，咱们理理看",
            ],
            "medium": [
                "担心是正常的",
                "能理解你的紧张",
                "嗯，先冷静一下",
            ],
            "low": [
                "嗯",
                "了解",
                "知道",
            ],
        },
        "anxiety": {
            "high": [
                "DDL压力大是吧...我懂这种焦虑",
                "时间紧确实让人着急，你先别急",
                "我理解，真的很难受这种压力",
                "咱们来想想怎么办，别一个人扛",
            ],
            "medium": [
                "能理解你的着急",
                "时间紧确实让人焦虑",
                "先别急，咱们理理",
            ],
            "low": [
                "嗯",
                "知道",
                "明白",
            ],
        },
        "surprise": {
            "high": [
                "哇！真的没想到！太意外了！",
                "这太出乎意料了！什么情况？",
                "我天！真没想到！说说看！",
                "确实意外！什么感受？",
            ],
            "medium": [
                "确实意外",
                "没想到",
                "这事儿...",
            ],
            "low": [
                "哦？",
                "这样啊",
                "是吗",
            ],
        },
        "love": {
            "high": [
                "好甜啊！感觉你特别幸福！",
                "太让人羡慕了！好好珍惜！",
                "真好...这种时刻太美好了",
                "感觉你都快融化了！真为你们高兴！",
            ],
            "medium": [
                "真好啊",
                "甜蜜蜜的",
                "真不错",
            ],
            "low": [
                "嗯",
                "挺好",
                "不错",
            ],
        },
        "trust": {
            "high": [
                "是啊，有时候就是需要相信",
                "我懂你说的那种放心",
                "能找到可以信赖的人很难得",
                "确实这种感觉很重要",
            ],
            "medium": [
                "理解",
                "是的",
                "明白",
            ],
            "low": [
                "嗯",
                "对",
            ],
        },
        "anticipation": {
            "high": [
                "好期待啊！说说你的计划！",
                "感觉你充满希望！我也想听！",
                "期待是好事！有什么想法？",
                "太棒了，感觉你动力满满！",
            ],
            "medium": [
                "期待是好事",
                "挺好的",
                "加油！",
            ],
            "low": [
                "嗯",
                "好",
                "知道",
            ],
        },
        "optimism": {
            "high": [
                "这种心态太好了！",
                "确实一切都会好起来的！",
                "喜欢这种积极的态度！",
                "太好了！保持这个状态！",
            ],
            "medium": [
                "挺好的",
                "嗯，会好的",
                "不错",
            ],
            "low": [
                "嗯",
                "好",
            ],
        },
        "guilt": {
            "high": [
                "我懂，这种自责很难受",
                "事情已经发生了，别太为难自己",
                "能感觉到你的愧疚，但人都会犯错",
                "你不是故意的，对自己好一点",
            ],
            "medium": [
                "别太自责了",
                "能理解你的感受",
                "人会犯错，也正常",
            ],
            "low": [
                "嗯",
                "知道",
            ],
        },
        "envy": {
            "high": [
                "我懂...看到别人有自己没有确实难受",
                "羡慕是正常的，谁都有这种时候",
                "我理解你的心情",
                "这种感觉确实不好受，但也是人之常情",
            ],
            "medium": [
                "能理解",
                "明白",
                "有这种感觉也正常",
            ],
            "low": [
                "嗯",
                "知道",
            ],
        },
        "contempt": {
            "high": [
                "确实让人看不起",
                "换我也会觉得过分",
                "我懂你的感受",
                "这种事确实让人无语",
            ],
            "medium": [
                "确实不好",
                "无语",
            ],
            "low": [
                "嗯",
            ],
        },
        "despair": {
            "high": [
                "我能感受到你现在很难...但我在这里",
                "哎...先别想太多，有什么想说的就说",
                "我知道很难，但你不是一个人",
                "先缓缓，我陪着你",
            ],
            "medium": [
                "我懂",
                "能理解",
                "先别急",
            ],
            "low": [
                "嗯",
                "知道",
            ],
        },
        "disgust": {
            "high": [
                "确实恶心，换我也受不了",
                "这种事太让人反胃了",
                "我懂你的感受",
                "确实过分",
            ],
            "medium": [
                "确实不好",
                "无语",
                "太过了",
            ],
            "low": [
                "嗯",
            ],
        },
    }

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        user_emotion: str,
        intensity: float,
        context: Optional[str] = None,
    ) -> EmpathyResponse:
        """
        生成共情回应

        Args:
            user_emotion: 用户情感
            intensity: 情感强度 0-1
            context: 可选的上下文信息

        Returns:
            EmpathyResponse对象
        """
        # 确定强度级别
        if intensity > 0.7:
            level = "high"
        elif intensity > 0.3:
            level = "medium"
        else:
            level = "low"

        # 获取对应模板
        emotion_templates = self.EMPATHY_RESPONSES.get(
            user_emotion,
            self.EMPATHY_RESPONSES["sadness"]  # 默认用sadness模板
        )

        templates = emotion_templates.get(level, emotion_templates["low"])
        response_text = self.rng.choice(templates)

        # 确定共情类型
        empathy_type = self._determine_empathy_type(user_emotion, intensity)

        # 生成后续引导建议
        follow_up = self._generate_follow_up(user_emotion, empathy_type)

        return EmpathyResponse(
            text=response_text,
            empathy_type=empathy_type,
            intensity_level=level,
            follow_up_suggestion=follow_up,
        )

    def _determine_empathy_type(
        self, emotion: str, intensity: float
    ) -> str:
        """根据情感和强度决定共情类型"""
        if emotion in ["sadness", "despair"]:
            if intensity > 0.6:
                return "support"  # 需要支持
            return "understanding"  # 需要理解
        elif emotion in ["anger", "contempt", "disgust"]:
            return "venting"  # 需要发泄
        elif emotion in ["anxiety", "fear"]:
            return "action"  # 需要行动指引
        elif emotion in ["joy", "love", "surprise"]:
            return "understanding"  # 需要分享
        return "understanding"

    def _generate_follow_up(
        self, emotion: str, empathy_type: str
    ) -> Optional[str]:
        """生成后续引导建议"""
        follow_ups = {
            "support": "想聊聊具体发生了什么吗？",
            "venting": "还有什么让你气不过的？说出来",
            "action": "咱们看看能做点什么？",
            "understanding": "后来呢？",
        }
        return follow_ups.get(empathy_type)

    def respond_with_empathy(
        self, user_text: str, user_emotion: str, intensity: float
    ) -> str:
        """
        快速生成共情回应

        Args:
            user_text: 用户输入
            user_emotion: 检测到的情感
            intensity: 情感强度

        Returns:
            共情文本
        """
        response = self.generate(user_emotion, intensity)
        return response.text


@dataclass
class ConversationEmpathy:
    """对话中的共情状态"""
    last_empathy_type: str = "understanding"
    empathy_depth: int = 0  # 共情深度计数
    response_count: int = 0


class AdaptiveEmpathyEngine(EmpathyEngine):
    """
    自适应共情引擎

    在基础共情引擎上，增加了：
    - 对话中的共情深度追踪
    - 根据共情深度调整回应策略
    - 避免重复同样的共情表达
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.conversation_state = ConversationEmpathy()
        self._used_templates: List[str] = []

    def generate_adaptive(
        self,
        user_emotion: str,
        intensity: float,
        user_text: Optional[str] = None,
    ) -> EmpathyResponse:
        """
        生成自适应共情回应

        根据对话深度调整共情策略
        """
        # 记录对话数
        self.conversation_state.response_count += 1

        # 生成基础回应
        response = self.generate(user_emotion, intensity)

        # 避免重复使用相同模板
        if response.text in self._used_templates and len(self._used_templates) < 10:
            # 获取另一个随机模板
            alt_response = self.generate(user_emotion, intensity)
            if alt_response.text not in self._used_templates:
                response = alt_response

        # 更新已使用模板
        self._used_templates.append(response.text)
        if len(self._used_templates) > 20:
            self._used_templates.pop(0)

        # 根据共情深度调整
        if self.conversation_state.empathy_depth > 3:
            # 深度共情后，引导到行动
            if response.follow_up_suggestion:
                response.follow_up_suggestion = "咱们看看能做点什么？"

        return response

    def reset_conversation(self):
        """重置对话状态（新对话开始时调用）"""
        self.conversation_state = ConversationEmpathy()
        self._used_templates = []


if __name__ == "__main__":
    # 测试
    engine = AdaptiveEmpathyEngine(seed=42)

    print("=== Empathy Engine Test ===\n")

    # 测试不同情感的共情回应
    test_cases = [
        ("项目又延期了，好烦啊", "anger", 0.8),
        ("考研终于过了！太开心了！", "joy", 0.95),
        ("被裁员了，不知道怎么办...", "sadness", 0.85),
        ("担心考试考不好", "anxiety", 0.6),
        ("老公出轨了...", "despair", 0.9),
    ]

    for text, emotion, intensity in test_cases:
        response = engine.generate_adaptive(emotion, intensity, text)
        print(f"User: {text}")
        print(f"  Emotion: {emotion}, Intensity: {intensity}")
        print(f"  Response: {response.text}")
        print(f"  Type: {response.empathy_type}, Level: {response.intensity_level}")
        if response.follow_up_suggestion:
            print(f"  Follow-up: {response.follow_up_suggestion}")
        print()

    print("=== Conversation Flow Test ===")
    # 模拟连续对话
    engine.reset_conversation()

    user_says = [
        ("工作丢了...", "sadness", 0.8),
        ("是啊，投了很多简历都没消息", "anxiety", 0.7),
        ("不知道还能做什么", "despair", 0.6),
        ("你说的对，我试试看", "optimism", 0.5),
    ]

    for text, emotion, intensity in user_says:
        response = engine.generate_adaptive(emotion, intensity, text)
        print(f"User: {text}")
        print(f"  -> AI: {response.text}")
        print()