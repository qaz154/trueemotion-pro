"""
共情响应生成引擎
生成有血有肉的、口语化的情感化回复
"""

from dataclasses import dataclass
from typing import Optional

from trueemotion.core.analysis.output import HumanResponse
from trueemotion.core.emotions.plutchik24 import EMOTION_VAD


# 共情回复模板
RESPONSE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "joy": {
        "support": [
            "太为你高兴了！说说怎么回事！",
            "哇，太棒了！详细讲讲呗～",
            "开心！这种好事必须分享！",
            "太好了！替你开心！",
        ],
        "excited": [
            "太开心了吧！哈哈哈！",
            "哇哇哇，好羡慕！",
            "啊啊啊我也好开心！",
            "哈哈哈太欢乐了！",
        ],
    },
    "sadness": {
        "support": [
            "先缓缓，我陪着你",
            "心疼你，说说怎么了",
            "我懂，真的不容易",
            "别憋着，说出来会好受点",
        ],
        "comfort": [
            "抱抱你，会好的",
            "难过的话先不说，我听着",
            "不管怎样，我都在",
            "先给自己倒杯水，慢慢来",
        ],
    },
    "anger": {
        "support": [
            "确实气人！换我我也急！",
            "太理解你了，换我我也火大",
            "这种事儿搁谁都得生气",
            "消消气，别伤了身体",
        ],
        "calm": [
            "先深呼吸，慢慢说",
            "我听着，骂出来也行",
            "气话跟我说就行，别憋着",
            "发泄一下也好，我陪你",
        ],
    },
    "fear": {
        "support": [
            "别怕，有我在",
            "先冷静下来，我们一起想办法",
            "不管怎样，先保证安全",
            "我理解，真的挺吓人的",
        ],
        "calm": [
            "深呼吸，慢慢说",
            "没事的，我听着呢",
            "先冷静一下",
            "一步一步来，别急",
        ],
    },
    "anxiety": {
        "support": [
            "我理解你的担心",
            "慢慢来，别给自己太大压力",
            "先理清思路，我们一起看看",
            "不管结果怎样，我都在",
        ],
        "calm": [
            "先停下来，深呼吸",
            "一件一件来，不着急",
            "我陪着你，慢慢来",
            "先放下，别想太多",
        ],
    },
    "surprise": {
        "excited": [
            "哇！真的假的！",
            "这也太意外了吧！",
            "什么？！不会吧！",
            "天哪，说说怎么回事！",
        ],
        "curious": [
            "什么情况？快说说！",
            "真的假的？详细讲讲！",
            "这也太惊人了！",
            "等等，让我缓一缓",
        ],
    },
    "love": {
        "warm": [
            "好甜啊！说说是怎么回事",
            "好羡慕你们！",
            "真好啊！",
            "这种感觉很棒对吧！",
        ],
        "excited": [
            "啊啊啊好甜！",
            "说详细点我想听！",
            "好幸福的感觉！",
            "好甜好甜好甜！",
        ],
    },
    "trust": {
        "warm": [
            "能感受到你的信任",
            "谢谢你的信任，我会努力的",
            "一起加油！",
            "有你在真好",
        ],
        "support": [
            "一起面对",
            "我们是一伙的",
            "有我呢",
            "放心交给我",
        ],
    },
    "anticipation": {
        "excited": [
            "期待期待！说说是什么！",
            "好想快点知道！",
            "有这种好事？快说快说！",
            "感觉要有什么好事发生了！",
        ],
        "curious": [
            "是什么什么？好想知道！",
            "说说你的计划？",
            "期待你们的想法",
            "有什么计划吗？",
        ],
    },
    "guilt": {
        "comfort": [
            "知错能改就好",
            "谁都会犯错，别太自责",
            "能意识到就很好",
            "我理解你的感受",
        ],
        "support": [
            "没关系，谁都会这样",
            "重要的是你现在怎么想",
            "过去的事就让它过去吧",
            "下次会更好的",
        ],
    },
    "envy": {
        "warm": [
            "确实会羡慕呢",
            "能理解你的心情",
            "没关系，每个人都有自己的节奏",
            "你也会有的",
        ],
        "support": [
            "加油，你也可以的",
            "相信你也会有的",
            "一起努力",
            "你的日子也会越来越好的",
        ],
    },
    "despair": {
        "comfort": [
            "先停下来，深呼吸",
            "不管怎样，我都在",
            "先不要想太多，休息一下",
            "明天会好的",
        ],
        "support": [
            "我陪着你",
            "慢慢来，不着急",
            "先休息，其他的交给我",
            "我在这里，不会走的",
        ],
    },
    "neutral": {
        "support": [
            "嗯嗯，我听着呢",
            "然后呢？",
            "说说看",
            "怎么想的？",
        ],
        "curious": [
            "然后呢？",
            "还有吗？",
            "想多了解一点",
            "展开说说？",
        ],
    },
}


# 追问模板
FOLLOW_UP_TEMPLATES: dict[str, list[str]] = {
    "joy": ["然后呢？", "详细讲讲！", "太棒了！", "继续继续！"],
    "sadness": ["发生什么了？", "想说说吗？", "我听着", "愿意的话说出来"],
    "anger": ["怎么了？", "什么事让你这么生气？", "说说看", "我帮你分析分析"],
    "fear": ["在担心什么？", "能说说吗？", "我陪你", "别害怕"],
    "anxiety": ["在担心什么？", "有什么心事吗？", "说说看", "我帮你想想"],
    "surprise": ["什么情况？！", "真的假的？！", "详细说说！", "不会吧！"],
    "love": ["哇！说详细点！", "好甜！", "怎么认识的？", "然后呢然后呢？"],
    "trust": ["什么事？", "怎么做到的？", "说说看", "一起分享"],
    "anticipation": ["什么计划？", "然后呢？", "好期待啊！", "说说你的想法"],
    "guilt": ["发生什么了？", "想说说吗？", "我陪你", "别太自责"],
    "envy": ["羡慕什么？", "你也会有更好的", "想聊聊吗？", "一起加油"],
    "despair": ["先休息一下", "我陪着你", "慢慢来", "不着急"],
    "neutral": ["嗯嗯", "然后呢？", "怎么想的？", "说说看"],
}


@dataclass
class EmpathyEngine:
    """
    共情引擎

    生成符合情感的、口语化的、人性化的回复
    """

    def generate(
        self,
        emotion: str,
        intensity: float,
        context: Optional[str] = None,
    ) -> HumanResponse:
        """
        生成共情回复

        Args:
            emotion: 情感类型
            intensity: 强度 0-1
            context: 可选的上下文

        Returns:
            HumanResponse: 生成的回复
        """
        # 获取回复类型
        empathy_type = self._get_empathy_type(emotion, intensity)
        intensity_level = self._get_intensity_level(intensity)

        # 选择模板
        templates = self._get_templates(emotion, empathy_type)
        if not templates:
            templates = self._get_templates("neutral", "support")

        response_text = templates[int(intensity * 1000) % len(templates)]

        # 可能的追问
        follow_up = None
        if intensity > 0.5 and emotion != "neutral":
            follow_ups = FOLLOW_UP_TEMPLATES.get(emotion, FOLLOW_UP_TEMPLATES["neutral"])
            follow_up = follow_ups[int(intensity * 100) % len(follow_ups)]

        return HumanResponse(
            text=response_text,
            empathy_type=empathy_type,
            intensity_level=intensity_level,
            follow_up=follow_up,
        )

    def _get_empathy_type(self, emotion: str, intensity: float) -> str:
        """根据情感和强度确定回复类型"""
        if emotion in ("joy", "surprise", "love", "anticipation"):
            return "excited" if intensity > 0.7 else "support"
        elif emotion in ("sadness", "fear", "despair"):
            return "comfort" if intensity > 0.5 else "support"
        elif emotion == "anger":
            return "calm" if intensity > 0.6 else "support"
        else:
            return "support"

    def _get_intensity_level(self, intensity: float) -> str:
        """获取强度等级"""
        if intensity >= 0.9:
            return "extreme"
        elif intensity >= 0.7:
            return "high"
        elif intensity >= 0.5:
            return "moderate"
        elif intensity >= 0.3:
            return "low"
        return "minimal"

    def _get_templates(self, emotion: str, empathy_type: str) -> list[str]:
        """获取回复模板"""
        emotion_templates = RESPONSE_TEMPLATES.get(emotion, {})
        return emotion_templates.get(
            empathy_type,
            RESPONSE_TEMPLATES.get("neutral", {}).get("support", [])
        )
