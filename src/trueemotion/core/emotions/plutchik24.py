"""
Plutchik 24色情感轮定义
基于Plutchik情感轮理论，扩展为24种精细情感
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


@dataclass(frozen=True)
class EmotionDefinition:
    """情感定义"""
    name: str
    name_cn: str
    vad: Tuple[float, float, float]  # Valence, Arousal, Dominance
    intensity: float  # 1-10 基础强度
    keywords: Tuple[str, ...]  # 中文关键词
    anti_emotion: str  # 对立情感
    parent: str  # 所属主要情感


class PlutchikEmotion(Enum):
    """Plutchik 24色情感轮"""
    # 主要情感 (8种)
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    DISGUST = "disgust"
    SURPRISE = "surprise"
    TRUST = "trust"
    ANTICIPATION = "anticipation"

    # 扩展情感 (16种)
    ECSTASY = "ecstasy"        # 狂喜 (joy强化)
    GRIEF = "grief"            # 悲痛 (sadness强化)
    RAGE = "rage"              # 暴怒 (anger强化)
    TERROR = "terror"          # 恐惧 (fear强化)
    LOATHING = "loathing"      # 厌恶 (disgust强化)
    ASTONISHMENT = "astonishment"  # 惊骇 (surprise强化)
    ADMIRATION = "admiration"  # 钦佩 (trust强化)
    VIGILANCE = "vigilance"    # 警觉 (anticipation强化)

    SERENITY = "serenity"      # 宁静 (joy弱化)
    PENITENCE = "penitence"    # 悔恨 (sadness弱化)
    ANNOYANCE = "annoyance"    #烦恼 (anger弱化)
    APPREHENSION = "apprehension"  # 忧虑 (fear弱化)
    BOREDOM = "boredom"        # 倦怠 (disgust弱化)
    DISTRACTION = "distraction"  # 分心 (surprise弱化)
    ACCEPTANCE = "acceptance"  # 接纳 (trust弱化)
    INTEREST = "interest"      # 兴趣 (anticipation弱化)

    # 复合情感
    LOVE = "love"              # 爱 (joy + trust)
    GUILT = "guilt"            # 内疚 (joy + fear)
    ENVY = "envy"              # 嫉妒 (sadness + anger)
    CONTEMPT = "contempt"      # 鄙视 (anger + disgust)
    OPTIMISM = "optimism"       # 乐观 (joy + anticipation)
    DESPAIR = "despair"        # 绝望 (sadness + fear)
    DESIRE = "desire"          # 渴望 (anticipation + joy)
    PRIDE = "pride"            # 自豪 (joy + anger)

    # 焦虑谱系
    ANXIETY = "anxiety"        # 焦虑
    REMORSE = "remorse"        # 自责


# VAD坐标映射
EMOTION_VAD: Dict[str, Tuple[float, float, float]] = {
    # 主要情感
    "joy": (0.8, 0.5, 0.7),
    "sadness": (-0.8, -0.3, -0.5),
    "anger": (-0.8, 0.7, 0.5),
    "fear": (-0.6, 0.6, -0.4),
    "disgust": (-0.7, -0.1, -0.4),
    "surprise": (0.3, 0.8, 0.3),
    "trust": (0.6, 0.3, 0.7),
    "anticipation": (0.5, 0.6, 0.4),

    # 强化情感
    "ecstasy": (0.95, 0.8, 0.9),
    "grief": (-0.95, -0.6, -0.7),
    "rage": (-0.95, 0.95, 0.7),
    "terror": (-0.8, 0.9, -0.6),
    "loathing": (-0.85, -0.3, -0.5),
    "astonishment": (0.5, 0.95, 0.5),
    "admiration": (0.8, 0.4, 0.9),
    "vigilance": (0.4, 0.8, 0.6),

    # 弱化情感
    "serenity": (0.6, 0.2, 0.5),
    "penitence": (-0.5, 0.0, -0.4),
    "annoyance": (-0.5, 0.4, 0.3),
    "apprehension": (-0.4, 0.4, -0.3),
    "boredom": (-0.3, -0.5, -0.2),
    "distraction": (0.1, 0.3, 0.1),
    "acceptance": (0.4, 0.1, 0.5),
    "interest": (0.4, 0.5, 0.4),

    # 复合情感
    "love": (0.9, 0.4, 0.8),
    "guilt": (0.5, 0.2, -0.4),
    "envy": (-0.4, 0.3, -0.3),
    "contempt": (-0.6, 0.2, 0.4),
    "optimism": (0.7, 0.5, 0.6),
    "despair": (-0.9, 0.3, -0.7),
    "desire": (0.8, 0.7, 0.6),
    "pride": (0.7, 0.5, 0.8),

    # 焦虑谱系
    "anxiety": (-0.5, 0.6, -0.4),
    "remorse": (-0.6, 0.2, -0.5),
}


# 情感关键词映射
EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "joy": ("开心", "高兴", "快乐", "愉快", "喜悦", "欢快", "快乐", "幸福", "美好", "棒", "爽", "耶", "哈哈", "嘻嘻", "乐", "欢欣", "雀跃", "得意", "满足", "欣慰"),
    "sadness": ("难过", "伤心", "悲伤", "悲痛", "失落", "沮丧", "郁闷", "压抑", "绝望", "痛苦", "哀伤", "凄凉", "沉重", "消沉", "低落", "忧郁", "伤感", "想哭", "泪目", "心碎"),
    "anger": ("生气", "愤怒", "气愤", "恼火", "发火", "大火", "暴怒", "气", "怒", "不爽", "讨厌", "可恶", "讨厌", "恨", "气死", "抓狂", "崩溃", "火大", "怒了", "发怒"),
    "fear": ("害怕", "恐惧", "担心", "担忧", "紧张", "怕", "畏", "惶恐", "不安", "惊恐", "惊吓", "胆怯", "心虚", "后怕", "发毛", "哆嗦", "发抖", "惊悚", "可怕", "吓人"),
    "disgust": ("恶心", "厌恶", "讨厌", "反感", "腻", "烦", "腻歪", "嫌弃", "憎恶", "作呕", "反胃", "厌恶", "鄙夷", "不屑", "嫌弃", "恶心", "反感的", "吐了", "油腻", "土"),
    "surprise": ("惊讶", "吃惊", "意外", "震惊", "惊", "哇", "呀", "咦", "诶", "竟然", "居然", "万万没想到", "没想到", "惊讶", "惊呆", "吓", "吓到", "震惊", "惊喜", "惊吓"),
    "trust": ("相信", "信任", "依赖", "放心", "可靠", "托付", "依赖", "信仰", "信念", "认可", "肯定", "确定", "踏实", "安心", "信赖", "依靠", "托付", "交给", "信心", "安心"),
    "anticipation": ("期待", "希望", "盼望", "憧憬", "等待", "期望", "想", "将要", "即将", "马上", "快要", "就要", "指望", "愿望", "向往", "期待", "盼望", "期待", "希望", "憧憬"),

    # 强化情感
    "ecstasy": ("狂喜", "兴奋", "疯了", "太开心了", "爽翻了", "嗨", "起飞", "爆炸", "激动", "热血", "沸腾", "兴奋不已", "癫狂", "狂乐", "欢天喜地", "心花怒放", "手舞足蹈", "欣喜若狂", "乐不可支", "欢欣雀跃"),
    "grief": ("痛哭", "崩溃", "绝望", "心碎", "悲痛欲绝", "哭", "泪流满面", "嚎啕", "嚎啕大哭", "泣不成声", "捶胸顿足", "痛不欲生", "万念俱灰", "心如刀割", "肝肠寸断", "撕心裂肺", "痛彻心扉", "凄惨", "惨淡", "悲恸"),
    "rage": ("暴怒", "狂怒", "怒火中烧", "怒不可遏", "火冒三丈", "大发雷霆", "怒发冲冠", "暴跳如雷", "怒狠狠", "气炸", "气疯了", "恨", "怨恨", "仇", "敌意", "凶狠", "残暴", "狂暴", "凶残", "狰狞"),
    "terror": ("极度恐惧", "吓死", "吓坏", "恐怖", "可怕", "毛骨悚然", "胆战心惊", "心惊肉跳", "魂飞魄散", "惊骇", "恐惧", "惶恐", "惊恐", "震恐", "惧怕", "畏怯", "怯懦", "害怕", "颤抖", "哆嗦"),
    "loathing": ("深恶痛绝", "极其厌恶", "作呕", "恶心至极", "厌恶至极", "憎恨", "唾弃", "不齿", "鄙夷", "蔑视", "看不起", "藐视", "轻视", "小看", "不屑", "恶心", "讨厌", "反感", "腻味", "腻歪"),
    "astonishment": ("惊骇", "震惊", "惊呆", "目瞪口呆", "瞠目结舌", "大惊", "惊呼", "惊叹", "惊讶", "惊异", "惊诧", "诧异", "匪夷所思", "难以置信", "不可思议", "震惊", "惊愕", "惊倒", "呆住", "愣住"),
    "admiration": ("钦佩", "敬佩", "佩服", "崇拜", "仰慕", "羡慕", "欣赏", "赞叹", "赞许", "认可", "赞同", "称颂", "颂扬", "赞美", "表扬", "称赞", "赞赏", "欣赏", "敬仰", "敬重"),
    "vigilance": ("警觉", "警惕", "戒备", "提防", "谨慎", "小心", "注意", "留意", "关注", "盯住", "监视", "守卫", "警戒", "守候", "等候", "警惕", "警觉", "敏锐", "机警", "警觉"),

    # 复合情感
    "love": ("爱", "喜欢", "爱慕", "喜爱", "爱上", "心爱", "亲爱的", "甜蜜", "温馨", "浪漫", "表白", "追求", "想念", "思念", "牵挂", "在乎", "喜欢", "爱", "甜蜜", "幸福"),
    "guilt": ("内疚", "愧疚", "自责", "抱歉", "对不起", "后悔", "懊悔", "悔恨", "惭愧", "心虚", "过意不去", "不安", "歉意", "赔罪", "请罪", "悔过", "自责", "反省", "检讨", "悔改"),
    "envy": ("嫉妒", "羡慕", "眼红", "眼馋", "眼热", "嫉妒", "醋意", "酸", "吃醋", "羡慕", "眼红", "嫉妒", "不忿", "不服气", "凭什么", "眼馋", "攀比", "比较", "羡慕嫉妒", "眼热"),
    "contempt": ("鄙视", "蔑视", "藐视", "轻蔑", "看不起", "不屑", "鄙夷", "轻蔑", "看不起", "藐视", "蔑视", "轻视", "小看", "瞧不起", "不屑一顾", "嗤之以鼻", "鄙夷", "轻慢", "侮蔑", "蔑弃"),
    "optimism": ("乐观", "积极", "阳光", "希望", "向上", "正面", "正面", "充满希望", "有信心", "相信会好", "会好起来的", "加油", "积极", "乐观", "向上", "阳光", "希望", "期待", "憧憬美好", "前途光明"),
    "despair": ("绝望", "崩溃", "无望", "失望", "绝望", "无助", "无奈", "无能为力", "放弃", "自暴自弃", "绝望", "没希望", "没救了", "完了", "绝望", "万念俱灰", "心死", "心灰意冷", "意志消沉", "消沉"),
    "desire": ("渴望", "欲望", "想要", "欲望", "想念", "欲望", "渴望", "想要", "希望得到", "追求", "欲望", "渴求", "贪图", "贪念", "欲念", "邪念", "妄念", "执念", "贪心", "占有欲"),

    # 焦虑谱系
    "anxiety": ("焦虑", "忧虑", "着急", "担忧", "担心", "不安", "紧张", "烦躁", "烦躁不安", "坐立不安", "忐忑", "心神不宁", "慌张", "慌忙", "焦急", "焦虑", "着急", "忧虑", "牵挂", "顾虑", "挂念"),
    "remorse": ("后悔", "悔恨", "自责", "懊悔", "悔过", "反省", "检讨", "愧疚", "歉意", "悔改", "后悔", "悔不当初", "悔悟", "痛悔", "悔过", "自责", "愧疚", "惭愧", "歉疚", "悔过自新"),

    # 弱化情感
    "serenity": ("平静", "安宁", "宁静", "平和", "安静", "轻松", "放松", "悠闲", "自在", "舒适", "惬意", "安心", "踏实", "安稳", "稳定", "平静", "宁静", "安然", "淡定", "从容"),
    "penitence": ("后悔", "悔过", "歉意", "愧疚", "自责", "反省", "检讨", "悔悟", "痛悔", "懊悔", "后悔", "悔不当初", "悔过", "悔改", "悔恨", "悔过自新", "反省", "检讨", "悔过", "悔罪"),
    "annoyance": ("烦躁", "烦恼", "烦", "讨厌", "腻", "腻味", "心烦", "烦躁不安", "不爽", "不痛快", "不悦", "不快", "烦闷", "苦闷", "郁结", "郁闷", "烦恼", "厌烦", "腻烦", "厌恶"),
    "apprehension": ("担忧", "忧虑", "担心", "不安", "紧张", "焦虑", "牵挂", "顾虑", "挂念", "放心不下", "忐忑", "不安", "焦虑", "忧心", "忧愁", "忧虑", "烦恼", "担心", "顾虑重重", "忐忑不安"),
    "boredom": ("无聊", "厌倦", "乏味", "腻", "烦", "没意思", "无聊", "厌倦", "倦怠", "疲惫", "疲倦", "累", "没劲", "无趣", "单调", "枯燥", "无聊", "厌倦", "倦怠", "无聊"),
    "distraction": ("分心", "走神", "注意力分散", "心不在焉", "恍惚", "迷离", "茫然", "恍惚", "迷离恍惚", "神游", "发呆", "愣神", "走神", "分神", "分心", "心不在焉", "恍惚", "失神", "愣住", "发呆"),
    "acceptance": ("接受", "认可", "认同", "同意", "赞成", "肯定", "承认", "容纳", "包容", "理解", "体谅", "谅解", "宽恕", "原谅", "接受", "认可", "认同", "同意", "肯定", "赞许"),
    "interest": ("感兴趣", "有兴趣", "好奇", "好奇心", "想了解", "想知道的", "关注", "留意", "注意", "研究", "探讨", "探索", "钻研", "深入", "好奇", "感兴趣", "趣味", "兴致", "爱好", "喜欢"),

    "pride": ("自豪", "骄傲", "得意", "成就感", "自信", "自负", "骄傲", "自豪", "得意", "成就感", "自信", "自满", "自大", "自尊", "自傲", "傲娇", "自豪感", "成就感", "骄傲", "得意洋洋"),
}


# 对立情感映射
EMOTION_ANTI: Dict[str, str] = {
    "joy": "sadness",
    "sadness": "joy",
    "anger": "fear",
    "fear": "anger",
    "disgust": "trust",
    "trust": "disgust",
    "surprise": "anticipation",
    "anticipation": "surprise",
    "ecstasy": "grief",
    "grief": "ecstasy",
    "rage": "terror",
    "terror": "rage",
    "loathing": "admiration",
    "admiration": "loathing",
    "astonishment": "distraction",
    "distraction": "astonishment",
    "vigilance": "boredom",
    "boredom": "vigilance",
    "serenity": "anxiety",
    "anxiety": "serenity",
    "penitence": "pride",
    "pride": "penitence",
    "annoyance": "interest",
    "interest": "annoyance",
    "apprehension": "acceptance",
    "acceptance": "apprehension",
    "love": "envy",
    "envy": "love",
    "guilt": "pride",
    "pride": "guilt",
    "contempt": "admiration",
    "admiration": "contempt",
    "optimism": "despair",
    "despair": "optimism",
    "desire": "boredom",
    "boredom": "desire",
    "remorse": "pride",
}


# 强度等级
def get_intensity_level(score: float) -> str:
    """根据强度分数返回等级"""
    if score >= 0.9:
        return "extreme"
    elif score >= 0.7:
        return "high"
    elif score >= 0.5:
        return "moderate"
    elif score >= 0.3:
        return "low"
    else:
        return "minimal"
