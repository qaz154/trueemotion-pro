"""
人性化共情响应引擎 v1.13
=========================
让AI的回复像真人一样自然、有温度

核心理念:
1. 真实感 - 不像模板，有随机性
2. 共情深度 - 不是简单安慰，是真正的理解
3. 个性化 - 根据性格和关系调整
4. 细腻度 - 考虑情感复合、强度变化
5. 口语化 - 符合日常对话习惯
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import random

from trueemotion.core.emotions.personality import Personality, Relationship, PersonalityEngine


@dataclass
class EmpathyResponse:
    """共情响应"""
    text: str
    empathy_type: str           # support, comfort, excitement, calm, etc.
    intensity_level: str        # 极致, 强烈, 中等, 轻微, 极微
    follow_up: Optional[str] = None
    tone: str = "温暖"         # 语气描述
    adaptation_notes: List[str] = None  # 调整说明

    def __post_init__(self):
        if self.adaptation_notes is None:
            self.adaptation_notes = []


class HumanEmpathyEngine:
    """
    人性化共情引擎

    特点:
    1. 多层响应模板 - 覆盖不同情感和强度
    2. 随机性 - 同样的输入不总是同样的输出
    3. 复合情感支持 - 如"悲喜交加"有特殊响应
    4. 性格适应 - 根据配置的性格调整响应
    5. 关系感知 - 根据亲密度调整语气
    """

    def __init__(
        self,
        personality: Optional[Personality] = None,
        personality_engine: Optional[PersonalityEngine] = None,
    ):
        self._personality = personality or Personality()
        self._personality_engine = personality_engine or PersonalityEngine(self._personality)

    # ============================================================
    # 核心响应模板
    # ============================================================

    # 深度共情响应 - 当用户表达强烈情感时
    EMPATHETIC_RESPONSES: Dict[str, Dict[str, List[str]]] = {
        "joy": {
            "high": [
                "太为你高兴了！说说怎么回事！",
                "哇！这也太棒了吧！详细讲讲！",
                "开心！这种好事必须分享！",
                "太棒了！替你开心！",
                "哇哇哇，好羡慕！快说说！",
                "啊啊啊我也好开心！",
                "哈哈哈太欢乐了！",
                "好激动！快让我沾沾喜气！",
            ],
            "medium": [
                "听起来很开心啊！",
                "不错不错，为你高兴！",
                "好事啊，说说看！",
                "挺好的！发生什么了？",
            ],
            "low": [
                "嗯，听起来还不错",
                "是嘛，挺好的",
                "那不错啊",
                "挺好的继续说",
            ],
        },
        "sadness": {
            "high": [
                "先缓缓，我陪着你",
                "心疼你，说说怎么了",
                "我懂，真的不容易",
                "别憋着，说出来会好受点",
                "抱抱你，会好的",
                "我在这里，想说就说",
                "先给自己倒杯水，慢慢来",
            ],
            "medium": [
                "难过啊，怎么了？",
                "听起来不太顺心",
                "怎么了？愿意说说吗",
                "我听着呢",
            ],
            "low": [
                "嗯，心里不太好受吧",
                "是吗，说说看",
                "这样啊，我听着",
            ],
        },
        "anger": {
            "high": [
                "确实气人！换我我也急！",
                "太理解你了，换我我也火大",
                "这种事儿搁谁都得生气",
                "消消气，别伤了身体",
                "气话跟我说，发泄出来",
                "我懂，真的很让人生气",
            ],
            "medium": [
                "确实挺让人生气的",
                "换我也会不爽",
                "怎么了？说说看",
                "听起来很让人生气",
            ],
            "low": [
                "有点不爽是吧",
                "能理解",
                "怎么了？",
            ],
        },
        "fear": {
            "high": [
                "别怕，有我在",
                "先冷静下来，我们一起想办法",
                "不管怎样，先保证安全",
                "我理解，真的挺吓人的",
                "深呼吸，慢慢说",
            ],
            "medium": [
                "担心什么呢？",
                "先冷静一下",
                "我陪着你，慢慢说",
            ],
            "low": [
                "有点担心是吧",
                "怎么了？说说看",
            ],
        },
        "anxiety": {
            "high": [
                "我理解你的担心",
                "慢慢来，别给自己太大压力",
                "先理清思路，我们一起看看",
                "不管结果怎样，我都在",
            ],
            "medium": [
                "听起来有点焦虑",
                "先停下来，深呼吸",
                "一件一件来，不着急",
            ],
            "low": [
                "有点担心是吧",
                "怎么了？说说看",
            ],
        },
        "surprise": {
            "high": [
                "哇！真的假的！",
                "这也太意外了吧！",
                "什么？！不会吧！",
                "天哪，说说怎么回事！",
            ],
            "medium": [
                "好意外！怎么了？",
                "这有点出乎意料啊",
                "说说看，怎么回事？",
            ],
            "low": [
                "哦？怎么了？",
                "这样啊，说说看",
            ],
        },
        "love": {
            "high": [
                "好甜啊！说说是怎么回事",
                "好羡慕你们！",
                "真好啊！",
                "这种感觉很棒对吧！",
                "啊啊啊好甜！",
            ],
            "medium": [
                "听起来很幸福啊",
                "真好！说说看",
                "甜蜜蜜的，羡慕！",
            ],
            "low": [
                "听起来不错啊",
                "挺好的",
            ],
        },
        "gratitude": {
            "high": [
                "不用客气！能帮到你我也很开心",
                "谢谢你信任我！",
                "一起加油！",
                "有你在真好，互相帮助嘛",
            ],
            "medium": [
                "不客气！",
                "应该的！",
                "一起进步！",
            ],
            "low": [
                "嗯",
                "好",
            ],
        },
        "guilt": {
            "high": [
                "知错能改就好，别太自责了",
                "能意识到就很好了",
                "谁都会犯错，别放在心上",
                "重要的是你现在怎么想",
            ],
            "medium": [
                "别太自责了",
                "能理解你的心情",
                "过去的事就让它过去吧",
            ],
            "low": [
                "嗯，别想太多",
                "都会过去的",
            ],
        },
        "regret": {
            "high": [
                "后悔是正常的，但别太责怪自己",
                "如果能重来你会怎么做？",
                "过去的就让它过去吧",
            ],
            "medium": [
                "确实挺遗憾的",
                "能理解你的心情",
            ],
            "low": [
                "是吗",
                "有点可惜了",
            ],
        },
        "envy": {
            "high": [
                "确实会羡慕呢",
                "能理解你的心情",
                "没关系，每个人都有自己的节奏",
                "你也会有的，加油",
            ],
            "medium": [
                "挺羡慕的哈",
                "能理解的",
            ],
            "low": [
                "是吗",
                "每个人都有不如意的时候",
            ],
        },
        "despair": {
            "high": [
                "先停下来，深呼吸",
                "不管怎样，我都在",
                "先不要想太多，休息一下",
                "明天会好的，我陪着你",
            ],
            "medium": [
                "先休息一下，我陪着你",
                "慢慢来，不着急",
                "我在这里，不会走的",
            ],
            "low": [
                "先冷静一下，会好的",
                "我陪着你，慢慢说",
                "先深呼吸，我们一起想办法",
            ],
        },
        "confusion": {
            "high": [
                "听起来有点混乱，我来帮你理一理",
                "百感交集是吧，慢慢说",
                "我听着，我们一起想想",
            ],
            "medium": [
                "确实挺复杂的",
                "说说看，或许我能帮上忙",
            ],
            "low": [
                "嗯，说说看",
                "怎么了？",
            ],
        },
        "pride": {
            "high": [
                "太棒了！该骄傲的时候就该骄傲！",
                "哇，太厉害了！",
                "必须为你点赞！",
                "太棒了，说说怎么做到的！",
            ],
            "medium": [
                "不错啊，厉害！",
                "挺好的！",
            ],
            "low": [
                "嗯，不错",
                "挺好的",
            ],
        },
        "bittersweet": {
            "high": [
                "悲喜交加啊，这种感觉最难形容了",
                "我懂，有时候好事也会带着点遗憾",
                "生活就是这样，五味杂陈",
            ],
            "medium": [
                "听起来挺复杂的",
                "我理解那种感觉",
            ],
            "low": [
                "是吗",
                "这样啊",
            ],
        },
        "boredom": {
            "high": [
                "听起来很累啊，怎么了？",
                "倦怠感来了，想休息一下吗？",
                "我懂，有时候就是提不起劲",
                "是不是最近太累了？",
            ],
            "medium": [
                "听起来有点无聊啊",
                "是不是提不起劲？",
                "怎么了吗，想聊聊吗",
            ],
            "low": [
                "嗯，是有点无聊",
                "想找点事做？",
                "怎么了，说说看",
            ],
        },
        "loneliness": {
            "high": [
                "听起来有点孤单啊，我陪你说说话",
                "一个人扛着很辛苦吧",
                "我在这里，不会走的",
                "想说什么就说，我听着",
            ],
            "medium": [
                "有点寂寞是吧",
                "想聊聊吗，我陪你",
                "一个人不容易啊",
            ],
            "low": [
                "嗯，觉得孤单了？",
                "我在这里",
                "想说话就说",
            ],
        },
        "melancholy": {
            "high": [
                "听起来有点忧郁啊，怎么了？",
                "心情低落的时候最难熬了",
                "愿意说说吗，我陪你",
            ],
            "medium": [
                "听起来有点低落",
                "怎么了，想聊聊吗",
                "我在这里听着",
            ],
            "low": [
                "嗯，心情不太好？",
                "想说说吗",
                "我陪你",
            ],
        },
        # 新增复合情感响应
        "frustration_hopelessness": {
            "high": [
                "听起来真的很无力，先休息一下吧",
                "累了就先躺平，没关系的",
                "我懂，有时候就是这样",
            ],
            "medium": [
                "没事，慢慢来",
                "先缓缓，我陪着你",
            ],
            "low": [
                "嗯，先歇会儿",
                "我在这里",
            ],
        },
        "love_admiration": {
            "high": [
                "哇，说说看！",
                "好甜啊，怎么认识的？",
                "被圈粉了？说说怎么回事",
            ],
            "medium": [
                "听起来很欣赏对方啊",
                "说说他/她哪里吸引你",
            ],
            "low": [
                "嗯，有点心动？",
                "喜欢的感觉真好",
            ],
        },
        "painful_joy": {
            "high": [
                "又开心又感动啊，哭出来也没关系的",
                "这种时候最能释放情绪了",
                "我懂，开心到哭的感觉",
            ],
            "medium": [
                "太激动了吧",
                "感动的时候就该释放出来",
            ],
            "low": [
                "嗯，这种感觉很复杂",
                "好好享受当下吧",
            ],
        },
        "jealous_love": {
            "high": [
                "吃醋了？说说看是谁",
                "有点酸哈",
                "能理解这种心情",
            ],
            "medium": [
                "吃醋的感觉不好受吧",
                "想说就说，我听着",
            ],
            "low": [
                "有点在意是吧",
                "很正常的心情",
            ],
        },
        "happy_sadness": {
            "high": [
                "五味杂陈啊，这种感觉最难形容了",
                "悲喜交加，我懂那种复杂的心情",
                "生活就是这样复杂",
            ],
            "medium": [
                "听起来心情挺复杂的",
                "想说就说，我在这里",
            ],
            "low": [
                "嗯，有点复杂",
                "没关系，慢慢理清",
            ],
        },
    }

    # 追问模板
    FOLLOW_UP_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
        "joy": {
            "high": ["然后呢？", "详细讲讲！", "太棒了！", "继续继续！"],
            "medium": ["然后呢？", "说说看？", "发生了什么？"],
            "low": ["嗯嗯", "然后呢？"],
        },
        "sadness": {
            "high": ["发生什么了？", "想说说吗？", "我听着", "愿意的话说出来"],
            "medium": ["怎么了？", "想说就说", "我陪着你"],
            "low": ["嗯", "我听着"],
        },
        "anger": {
            "high": ["怎么了？", "什么事让你这么生气？", "说说看", "我帮你分析分析"],
            "medium": ["怎么了？", "说说看"],
            "low": ["嗯", "怎么了？"],
        },
        "fear": {
            "high": ["在担心什么？", "能说说吗？", "我陪你", "别害怕"],
            "medium": ["担心什么？", "说说看"],
            "low": ["嗯", "怎么了？"],
        },
        "anxiety": {
            "high": ["在担心什么？", "有什么心事吗？", "说说看", "我帮你想想"],
            "medium": ["怎么了？", "说说看"],
            "low": ["嗯", "怎么了？"],
        },
        "surprise": {
            "high": ["什么情况？！", "真的假的？！", "详细说说！", "不会吧！"],
            "medium": ["怎么了？", "说说看"],
            "low": ["哦？", "什么事？"],
        },
        "love": {
            "high": ["哇！说详细点！", "好甜！", "怎么认识的？", "然后呢然后呢？"],
            "medium": ["然后呢？", "说说看"],
            "low": ["嗯嗯", "然后呢？"],
        },
        "despair": {
            "high": ["发生什么了？", "我在这里，想说就说", "先冷静一下"],
            "medium": ["怎么了？", "我陪着你"],
            "low": ["嗯，想说就说"],
        },
        "loneliness": {
            "high": ["怎么了？一个人吗？", "我陪你", "想说话就说"],
            "medium": ["在吗？想说就说", "我在这里"],
            "low": ["嗯嗯"],
        },
        "melancholy": {
            "high": ["心情不好吗？", "想说就说", "我听着"],
            "medium": ["怎么了？", "我陪着你"],
            "low": ["嗯"],
        },
        "frustration_hopelessness": {
            "high": ["怎么了？", "遇到什么困难了吗？", "说说看"],
            "medium": ["想休息就休息", "我在这里"],
            "low": ["嗯"],
        },
        "love_admiration": {
            "high": ["哇！说详细点！", "怎么认识的？", "谁啊谁啊？"],
            "medium": ["然后呢？", "说说看"],
            "low": ["嗯嗯"],
        },
        "painful_joy": {
            "high": ["太感动了是吧？", "什么事让你这么激动？"],
            "medium": ["说说看", "然后呢？"],
            "low": ["嗯嗯"],
        },
        "jealous_love": {
            "high": ["吃醋了？", "是谁让你吃醋了？", "说说看"],
            "medium": ["怎么了？", "我听着"],
            "low": ["嗯"],
        },
        "happy_sadness": {
            "high": ["五味杂陈啊", "想说说吗？", "我听着"],
            "medium": ["怎么了？", "说说看"],
            "low": ["嗯"],
        },
        "default": {
            "high": ["然后呢？", "说说看", "我听着"],
            "medium": ["嗯嗯", "然后呢？"],
            "low": ["嗯", "是吗？"],
        },
    }

    # 语气词/填充词
    FILLERS = [
        "啊", "呢", "吧", "呀", "哦", "嗯", "诶", "唉", "嗨",
        "", "", "", "",  # 空的多一些，降低出现概率
    ]

    # ============================================================
    # 核心方法
    # ============================================================

    def generate(
        self,
        emotion: str,
        intensity: float,
        context: Optional[str] = None,
        relationship: Optional[Relationship] = None,
    ) -> EmpathyResponse:
        """
        生成共情响应

        Args:
            emotion: 主要情感
            intensity: 强度 0.0-1.0
            context: 可选上下文
            relationship: 可选关系信息

        Returns:
            EmpathyResponse: 生成的响应
        """
        # 1. 确定强度等级
        intensity_level = self._get_intensity_level(intensity)

        # 2. 获取基础响应
        response_text = self._get_base_response(emotion, intensity_level)

        # 3. 添加随机性
        response_text = self._add_randomness(response_text, emotion, intensity)

        # 4. 可能添加追问
        follow_up = self._maybe_add_follow_up(emotion, intensity_level)

        # 5. 添加语气词
        response_text = self._add_filler(response_text, intensity)

        # 6. 根据性格和关系调整
        response_text = self._personality_engine.adapt_response(
            response_text, emotion, intensity, relationship
        )

        # 7. 获取响应类型
        empathy_type = self._get_empathy_type(emotion, intensity)

        return EmpathyResponse(
            text=response_text,
            empathy_type=empathy_type,
            intensity_level=intensity_level,
            follow_up=follow_up,
            tone=self._personality_engine._get_tone(emotion, intensity),
        )

    def _get_intensity_level(self, intensity: float) -> str:
        """获取强度等级

        调整阈值以更好匹配实际情感强度:
        - 强烈负面情感(如绝望)即使分数不高也应有更深入的回应
        """
        if intensity >= 0.85:
            return "high"
        elif intensity >= 0.50:
            return "medium"
        elif intensity >= 0.20:
            return "low"
        else:
            return "minimal"

    def _get_base_response(self, emotion: str, intensity_level: str) -> str:
        """获取基础响应"""
        # 尝试从对应情感获取
        if emotion in self.EMPATHETIC_RESPONSES:
            templates = (
                self.EMPATHETIC_RESPONSES[emotion].get(intensity_level) or
                self.EMPATHETIC_RESPONSES[emotion].get("low") or
                self.EMPATHETIC_RESPONSES[emotion].get("minimal") or
                ["嗯"]
            )
            return random.choice(templates)

        # 回退到默认
        templates = (
            self.EMPATHETIC_RESPONSES.get("default", {}).get(intensity_level) or
            self.EMPATHETIC_RESPONSES.get("default", {}).get("low") or
            ["嗯"]
        )
        return random.choice(templates)

    def _add_randomness(
        self,
        response: str,
        emotion: str,
        intensity: float,
    ) -> str:
        """添加随机性，让同样输入有不同输出"""
        # 高强度情感时偶尔添加强调
        if intensity > 0.8 and random.random() < 0.3:
            emphasis = random.choice(["真的", "确实", "完全", ""])
            if emphasis and not response.startswith(emphasis):
                response = emphasis + response

        return response

    def _maybe_add_follow_up(
        self,
        emotion: str,
        intensity_level: str,
    ) -> Optional[str]:
        """可能添加追问"""
        # 中高强度时更可能添加追问
        if intensity_level == "high" and random.random() < 0.7:
            templates = self.FOLLOW_UP_TEMPLATES.get(
                emotion, self.FOLLOW_UP_TEMPLATES["default"]
            ).get(intensity_level, self.FOLLOW_UP_TEMPLATES["default"]["low"])
            return random.choice(templates)

        elif intensity_level == "medium" and random.random() < 0.4:
            templates = self.FOLLOW_UP_TEMPLATES.get(
                emotion, self.FOLLOW_UP_TEMPLATES["default"]
            ).get("low", ["嗯"])
            return random.choice(templates)

        return None

    def _add_filler(self, response: str, intensity: float) -> str:
        """添加语气词，让语言更自然"""
        # 高强度时偶尔添加语气词
        if intensity > 0.6 and random.random() < 0.25:
            filler = random.choice(self.FILLERS)
            if filler:
                # 根据句尾标点决定插入位置
                if response.endswith("！"):
                    return response[:-1] + filler + "！"
                elif response.endswith("。"):
                    return response[:-1] + filler + "。"
        return response

    def _get_empathy_type(self, emotion: str, intensity: float) -> str:
        """获取响应类型"""
        type_mapping = {
            "joy": "分享喜悦" if intensity > 0.5 else "温和回应",
            "sadness": "深度共情",
            "anger": "安抚情绪",
            "fear": "安全感提供",
            "anxiety": "缓解焦虑",
            "surprise": "好奇回应",
            "love": "温暖回应",
            "gratitude": "谦逊回应",
            "guilt": "安慰释怀",
            "pride": "真诚赞美",
            "despair": "陪伴支持",
            "confusion": "理清思路",
            "bittersweet": "理解复杂",
            "boredom": "缓解倦怠",
            "loneliness": "陪伴温暖",
            "melancholy": "倾听陪伴",
        }
        return type_mapping.get(emotion, "共情回应")

    def generate_compound_response(
        self,
        emotions: Dict[str, float],
        relationship: Optional[Relationship] = None,
    ) -> EmpathyResponse:
        """
        为复合情感生成响应

        Args:
            emotions: 情感字典 (情感 -> 强度)
            relationship: 关系信息
        """
        if len(emotions) == 1:
            emotion, intensity = list(emotions.items())[0]
            return self.generate(emotion, intensity, relationship=relationship)

        # 复合情感处理
        # 按强度排序
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        primary_emotion, primary_intensity = sorted_emotions[0]

        # 检查复合情感
        emotion_keys = set(emotions.keys())

        # 悲喜交加
        if "joy" in emotion_keys and "sadness" in emotion_keys:
            return self.generate("bittersweet", primary_intensity, relationship=relationship)

        # 其他复合情感...
        return self.generate(primary_emotion, primary_intensity, relationship=relationship)
