"""
TrueEmotion - 人性化情感系统 v1.13
=====================================
让AI拥有像人类一样丰富、复杂、真实的情感

核心理念:
1. 情感复合 - 多种情感同时存在，如"喜极而泣"、"带泪的微笑"
2. 情感连续强度 - 情感不是0/1，而是0.0-1.0的连续光谱
3. 情感记忆 - 过去的经历塑造当下的感受
4. 性格建模 - 有人外放、有人内敛
5. 关系感知 - 对不同人情感不同
6. 情境依赖 - 同样的话在不同情境下含义不同
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from enum import Enum


# ============================================================
# 基础情感定义
# ============================================================

class EmotionSpectrum(Enum):
    """情感光谱 - 人类情感的8个基本维度"""
    JOY = "joy"           # 喜悦 - 开心、快乐、幸福
    SADNESS = "sadness"   # 悲伤 - 难过、失落、痛苦
    ANGER = "anger"       # 愤怒 - 生气、恼火、暴怒
    FEAR = "fear"         # 恐惧 - 害怕、担心、焦虑
    DISGUST = "disgust"   # 厌恶 - 讨厌、反感、恶心
    SURPRISE = "surprise"  # 惊讶 - 意外、震惊、惊呆
    TRUST = "trust"       # 信任 - 相信、依赖、放心
    ANTICIPATION = "anticipation"  # 期待 - 希望、盼望、憧憬


# VAD: Valence-Arousal-Dominance 情感坐标
# Valence (效价): -1(负面) ~ +1(正面)
# Arousal (唤醒度): -1(平静) ~ +1(激动)
# Dominance (支配度): -1(顺从) ~ +1(主导)
EMOTION_VAD: Dict[str, Tuple[float, float, float]] = {
    # 主要情感
    "joy": (0.85, 0.50, 0.70),
    "sadness": (-0.85, -0.30, -0.50),
    "anger": (-0.85, 0.70, 0.50),
    "fear": (-0.65, 0.60, -0.40),
    "disgust": (-0.75, -0.10, -0.40),
    "surprise": (0.30, 0.80, 0.30),
    "trust": (0.65, 0.30, 0.70),
    "anticipation": (0.50, 0.60, 0.40),

    # 细腻情感
    "ecstasy": (0.95, 0.85, 0.90),      # 狂喜
    "grief": (-0.95, -0.60, -0.70),     # 悲痛
    "rage": (-0.95, 0.95, 0.70),        # 暴怒
    "terror": (-0.80, 0.90, -0.60),     # 恐惧
    "loathing": (-0.85, -0.30, -0.50),  # 深恶痛绝
    "astonishment": (0.50, 0.95, 0.50),  # 惊骇
    "admiration": (0.80, 0.40, 0.90),   # 钦佩
    "vigilance": (0.40, 0.80, 0.60),    # 警觉

    # 柔和情感
    "serenity": (0.60, 0.15, 0.50),     # 宁静
    "pensiveness": (-0.40, 0.20, -0.30), # 沉思
    "annoyance": (-0.50, 0.40, 0.30),   # 烦恼
    "apprehension": (-0.40, 0.40, -0.30),# 忧虑
    "boredom": (-0.30, -0.50, -0.20),   # 倦怠
    "distraction": (0.10, 0.30, 0.10),   # 分心
    "acceptance": (0.40, 0.10, 0.50),    # 接纳
    "interest": (0.40, 0.50, 0.40),      # 兴趣

    # 复杂情感
    "love": (0.90, 0.40, 0.80),         # 爱
    "guilt": (0.30, 0.25, -0.40),       # 内疚
    "envy": (-0.40, 0.35, -0.30),       # 嫉妒
    "contempt": (-0.60, 0.20, 0.40),    # 鄙视
    "optimism": (0.70, 0.50, 0.60),     # 乐观
    "despair": (-0.90, 0.30, -0.70),    # 绝望
    "shame": (-0.40, 0.40, -0.50),      # 羞耻
    "pride": (0.70, 0.50, 0.80),        # 自豪

    # 细腻复合情感
    "longing": (0.50, 0.60, 0.30),      # 思念/渴望
    "nostalgia": (0.30, 0.20, 0.30),    # 怀旧
    "relief": (0.50, -0.30, 0.40),      # 如释重负
    "hope": (0.60, 0.50, 0.50),         # 希望
    "sorrow": (-0.70, -0.20, -0.40),    # 哀愁
    "melancholy": (-0.50, -0.10, -0.30),# 忧郁
    "gratitude": (0.70, 0.30, 0.60),    # 感激
    "regret": (-0.50, 0.10, -0.40),     # 遗憾
    "compassion": (0.30, 0.30, 0.20),   # 同情
    "amusement": (0.70, 0.70, 0.50),     # 愉悦/趣味
    "contentment": (0.60, 0.10, 0.50),   # 满足
    "confusion": (-0.10, 0.40, -0.10),   # 困惑
    "curiosity": (0.40, 0.60, 0.30),    # 好奇
    "embarrassment": (-0.30, 0.50, -0.40), # 尴尬
    "loneliness": (-0.60, 0.10, -0.40), # 孤独
    "gratitude_love": (0.80, 0.35, 0.70), # 感恩的爱
}


# ============================================================
# 情感关键词
# ============================================================

EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # 喜悦
    "joy": (
        "开心", "高兴", "快乐", "愉快", "喜悦", "欢快", "幸福", "美好", "棒",
        "爽", "耶", "哈哈", "嘻嘻", "乐", "欢欣", "雀跃", "得意", "满足",
        "欣慰", "舒畅", "痛快", "雀跃", "美滋滋", "乐呵呵", "笑", "开心"
    ),
    "ecstasy": (
        "狂喜", "兴奋", "疯了", "太开心了", "爽翻了", "嗨", "起飞", "爆炸",
        "激动", "热血", "沸腾", "癫狂", "欢天喜地", "心花怒放", "手舞足蹈",
        "欣喜若狂", "乐不可支", "欢欣雀跃", "亢奋", "嗨起来", "爽呆", "爆炸开心"
    ),

    # 悲伤
    "sadness": (
        "难过", "伤心", "悲伤", "悲痛", "失落", "沮丧", "郁闷", "压抑", "绝望",
        "痛苦", "哀伤", "凄凉", "沉重", "消沉", "低落", "忧郁", "伤感", "想哭",
        "泪目", "心碎", "哀愁", "凄惨", "苦涩", "酸楚", "悲痛", "酸涩"
    ),
    "grief": (
        "痛哭", "崩溃", "心碎", "悲痛欲绝", "泪流满面", "嚎啕", "泣不成声",
        "捶胸顿足", "痛不欲生", "万念俱灰", "心如刀割", "肝肠寸断", "撕心裂肺",
        "痛彻心扉", "悲恸", "哀恸", "绝哀"
    ),
    "melancholy": (
        "忧郁", "郁闷", "惆怅", "落寞", "寂寥", "萧索", "凄清", "苍凉",
        "感伤", "惘然", "惆怅若失", "百感交集", "思绪万千"
    ),
    "loneliness": (
        "孤独", "寂寞", "孤单", "空落落", "冷清", "孤零零", "寂寥", "落寞",
        "空虚", "茫然", "无所依托", "无所事事", "心里空空的"
    ),

    # 愤怒
    "anger": (
        "生气", "愤怒", "气愤", "恼火", "发火", "大火", "暴怒", "气", "怒",
        "不爽", "讨厌", "可恶", "恨", "气死", "抓狂", "崩溃", "火大", "怒了",
        "发怒", "大发雷霆", "怒不可遏", "恼怒", "愤慨", "怨恨", "憎恶"
    ),
    "rage": (
        "暴怒", "狂怒", "怒火中烧", "火冒三丈", "大发雷霆", "怒发冲冠", "暴跳如雷",
        "怒狠狠", "气炸", "气疯了", "恨之入骨", "怒不可遏", "满腔怒火", "怒形于色"
    ),
    "contempt": (
        "鄙视", "蔑视", "藐视", "轻蔑", "看不起", "不屑", "鄙夷", "轻视",
        "小看", "瞧不起", "不屑一顾", "嗤之以鼻", "轻慢", "侮蔑", "蔑弃", "不屑"
    ),

    # 恐惧
    "fear": (
        "害怕", "恐惧", "担心", "担忧", "紧张", "怕", "畏", "惶恐", "不安",
        "惊恐", "惊吓", "胆怯", "心虚", "后怕", "发毛", "哆嗦", "发抖", "惊悚"
    ),
    "terror": (
        "极度恐惧", "吓死", "吓坏", "恐怖", "可怕", "毛骨悚然", "胆战心惊",
        "心惊肉跳", "魂飞魄散", "惊骇", "惶恐", "震恐", "惧怕", "畏怯", "颤抖"
    ),
    "anxiety": (
        "焦虑", "忧虑", "着急", "担忧", "担心", "不安", "紧张", "烦躁",
        "坐立不安", "忐忑", "心神不宁", "慌张", "慌忙", "焦急", "牵挂", "顾虑"
    ),

    # 厌恶
    "disgust": (
        "恶心", "厌恶", "讨厌", "反感", "腻", "烦", "腻歪", "嫌弃", "憎恶",
        "作呕", "反胃", "鄙夷", "不屑", "恶心", "反感的", "吐了", "油腻", "土"
    ),
    "loathing": (
        "深恶痛绝", "极其厌恶", "作呕", "恶心至极", "厌恶至极", "唾弃", "不齿",
        "鄙夷", "蔑视", "看不起", "藐视", "轻视", "小看", "腻味", "恶心透顶"
    ),
    "boredom": (
        "无聊", "厌倦", "乏味", "腻", "烦", "没意思", "倦怠", "疲惫", "疲倦",
        "累", "没劲", "无趣", "单调", "枯燥", "腻烦", "兴味索然", "百无聊赖"
    ),

    # 惊讶
    "surprise": (
        "惊讶", "吃惊", "意外", "震惊", "惊", "哇", "呀", "咦", "诶", "竟然",
        "居然", "万万没想到", "没想到", "惊呆", "吓", "吓到", "惊喜", "惊吓"
    ),
    "astonishment": (
        "惊骇", "震惊", "惊呆", "目瞪口呆", "瞠目结舌", "大惊", "惊呼", "惊叹",
        "惊异", "惊诧", "诧异", "匪夷所思", "难以置信", "不可思议", "惊愕", "惊倒"
    ),

    # 信任
    "trust": (
        "相信", "信任", "依赖", "放心", "可靠", "托付", "信仰", "信念", "认可",
        "肯定", "确定", "踏实", "安心", "信赖", "依靠", "交给", "信心", "安全感"
    ),
    "admiration": (
        "钦佩", "敬佩", "佩服", "崇拜", "仰慕", "羡慕", "欣赏", "赞叹", "赞许",
        "认可", "赞同", "称颂", "颂扬", "赞美", "表扬", "称赞", "赞赏", "敬仰"
    ),
    "acceptance": (
        "接受", "认可", "认同", "同意", "赞成", "肯定", "承认", "容纳", "包容",
        "理解", "体谅", "谅解", "宽恕", "原谅", "赞许", "首肯"
    ),

    # 期待
    "anticipation": (
        "期待", "希望", "盼望", "憧憬", "等待", "期望", "想", "将要", "即将",
        "马上", "快要", "就要", "指望", "愿望", "向往", "期待", "盼望", "憧憬"
    ),
    "hope": (
        "希望", "盼望", "期待", "憧憬", "向往", "期望", "指望", "愿望", "指望",
        "光明", "曙光", "信心", "相信会好", "会好起来的"
    ),
    "optimism": (
        "乐观", "积极", "阳光", "向上", "正面", "充满希望", "有信心", "加油",
        "前途光明", "积极向上", "乐观其成", "抱持希望"
    ),
    "vigilance": (
        "警觉", "警惕", "戒备", "提防", "谨慎", "小心", "注意", "留意", "关注",
        "盯住", "监视", "守卫", "警戒", "敏锐", "机警"
    ),
    "curiosity": (
        "好奇", "好奇心", "想知道", "想了解", "怎么回事", "为什么", "什么情况",
        "有意思", "有趣", "探索", "钻研", "研究", "探讨"
    ),

    # 复杂情感
    "love": (
        "爱", "喜欢", "爱慕", "喜爱", "爱上", "心爱", "亲爱的", "甜蜜", "温馨",
        "浪漫", "表白", "追求", "想念", "思念", "牵挂", "在乎", "幸福", "甜蜜"
    ),
    "gratitude": (
        "感谢", "谢谢", "感激", "感恩", "多谢", "致谢", "谢意", "感激不尽",
        "没齿难忘", "铭感于心", "感谢不尽", "致以谢意"
    ),
    "guilt": (
        "内疚", "愧疚", "自责", "抱歉", "对不起", "后悔", "懊悔", "悔恨", "惭愧",
        "心虚", "过意不去", "不安", "歉意", "赔罪", "请罪", "悔过", "反省"
    ),
    "shame": (
        "羞耻", "丢脸", "不好意思", "惭愧", "丢人", "脸红", "汗颜", "无地自容",
        "羞愧", "羞耻感", "丢脸", "面子", "丢人现眼"
    ),
    "regret": (
        "后悔", "遗憾", "惋惜", "可惜", "懊悔", "悔不当初", "悔过", "追悔",
        "遗憾", "遗憾", "惋惜", "扼腕", "叹息"
    ),
    "envy": (
        "嫉妒", "羡慕", "眼红", "眼馋", "眼热", "醋意", "酸", "吃醋", "不忿",
        "不服气", "凭什么", "攀比", "羡慕嫉妒", "眼热"
    ),
    "pride": (
        "自豪", "骄傲", "得意", "成就感", "自信", "骄傲", "自豪感", "成就感",
        "得意洋洋", "飘飘然", "自满", "自负", "傲娇", "自尊"
    ),
    "despair": (
        "绝望", "崩溃", "无望", "失望", "无助", "无奈", "无能为力", "放弃",
        "自暴自弃", "没希望", "没救了", "完了", "万念俱灰", "心死", "消沉",
        "没有希望", "希望破灭", "人生没有希望", "前途黑暗", "看不到希望",
        "活着没意思", "生无可恋", "一切都没意义"
    ),
    "relief": (
        "如释重负", "松了一口气", "轻松", "放心", "松了口气", "卸下重担", "轻松愉快",
        "放下心头大石", "松了一口气", "心安"
    ),
    "contentment": (
        "满足", "满意", "知足", "惬意", "舒服", "舒适", "美滋滋", "心满意足",
        "称心如意", "如愿以偿", "满足感", "幸福感"
    ),
    "amusement": (
        "好笑", "有趣", "滑稽", "幽默", "逗", "可乐", "可笑", "搞笑", "逗笑",
        "趣味", "笑死我了", "太逗了", "笑死", "哈哈笑"
    ),
    "confusion": (
        "困惑", "疑惑", "不解", "迷糊", "茫然", "晕", "懵", "搞不懂", "糊涂",
        "不明白", "不清楚", "疑问", "困惑不解", "一头雾水"
    ),
    "embarrassment": (
        "尴尬", "不好意思", "难为情", "脸红", "羞涩", "腼腆", "害羞", "不好意思",
        "窘迫", "尴尬症", "脚趾抠地", "脸红心跳"
    ),
    "compassion": (
        "同情", "心疼", "可怜", "体恤", "怜悯", "心软", "不忍", "怜惜",
        "心有戚戚", "感同身受", "不忍心", "于心不忍"
    ),
    "longing": (
        "想念", "思念", "渴望", "惦记", "牵挂", "眷恋", "怀念", "惦记",
        "向往", "渴望", "期盼", "念想", "思念之情"
    ),
    "nostalgia": (
        "怀旧", "怀念", "回忆", "想念过去", "当年", "时光", "岁月", "追忆",
        "旧时", "往日", "忆当年", "怀念从前", "时光流逝"
    ),
    "sorrow": (
        "哀愁", "忧愁", "忧虑", "愁闷", "愁苦", "哀伤", "忧思", "郁闷",
        "愁绪", "忧伤", "哀怨", "悲戚", "凄然", "惆怅"
    ),

    # 复合情感
    "gratitude_love": (
        "感恩", "感激的爱", "温馨", "感动", "暖心", "窝心", "暖洋洋", "心暖",
        "感动", "感激不尽", "铭感于心", "感恩戴德"
    ),
    "hope_fear": (
        "忐忑", "惶恐", "焦虑", "担忧", "忧心忡忡", "心悬", "不安", "放心不下",
        "战战跟鞋", "如履薄冰", "惴惴不安"
    ),
    # 新增复合情感
    "frustration_hopelessness": (
        "无奈", "无助", "绝望", "无能为力", "算了", "就这样吧", "无所谓",
        "没意思", "无所谓了", "爱咋咋", "躺平", "摆烂"
    ),
    "love_admiration": (
        "崇拜", "仰慕", "钦佩", "心动", "暗恋", "crush", "喜欢", "欣赏",
        "被圈粉", "粉了", "路转粉", "始于颜值", "陷于才华"
    ),
    "painful_joy": (
        "喜极而泣", "喜极而悲", "激动", "热泪盈眶", "感动落泪", "泣不成声",
        "泪流满面", "哭了", "太激动了", "太感动了"
    ),
    "angry_fear": (
        "恐慌", "惊惧", "惧怕", "畏惧", "害怕", "惊恐", "惶恐不安",
        "心惊胆战", "坐立不安", "六神无主"
    ),
    "sadness_guilt": (
        "自责", "愧疚", "对不起", "过意不去", "心里不安", "亏欠",
        "对不起大家", "让大家失望了"
    ),
    "jealous_love": (
        "吃醋", "醋意", "酸了", "嫉妒", "占有欲", "吃醋了", "酸溜溜",
        "凭什么", "不公", "委屈"
    ),
    "shock_denial": (
        "不敢相信", "不可能", "不会吧", "假的吧", "开什么玩笑",
        "说笑吧", "骗人的吧", "我不信"
    ),
    "relief_sadness": (
        "如释重负又难过", "松口气又担心", "放心了但也失落",
        "终于没事了可是", "好了但感觉空落落的"
    ),
    "happy_sadness": (
        "又开心又难过", "悲喜交加", "五味杂陈", "复杂", "说不清",
        "开心也难过", "高兴也难受", "又笑又哭"
    ),
}


# ============================================================
# 情感关系映射
# ============================================================

# 对立情感
EMOTION_ANTONYM: Dict[str, str] = {
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
    "hope": "despair",
    "despair": "hope",
    "pride": "shame",
    "shame": "pride",
}

# 情感转化路径
EMOTION_TRANSITIONS: Dict[str, List[str]] = {
    "joy": ["contentment", "pride", "ecstasy", "love", "gratitude"],
    "sadness": ["grief", "despair", "loneliness", "melancholy", "sorrow"],
    "anger": ["rage", "contempt", "frustration", "annoyance"],
    "fear": ["terror", "anxiety", "apprehension", "worry"],
    "anticipation": ["hope", "hope_fear", "anxiety", "vigilance"],
    "surprise": ["astonishment", "confusion", "curiosity"],
    "trust": ["acceptance", "admiration", "love", "gratitude"],
    "disgust": ["loathing", "contempt", "boredom", "annoyance"],
}


# ============================================================
# 强度计算
# ============================================================

def get_intensity_label(score: float) -> str:
    """将0-1分数映射为人类可读的强度标签"""
    if score >= 0.95:
        return "极致"
    elif score >= 0.85:
        return "强烈"
    elif score >= 0.70:
        return "较高"
    elif score >= 0.50:
        return "中等"
    elif score >= 0.30:
        return "轻微"
    elif score >= 0.15:
        return "微弱"
    else:
        return "极微"


def get_vad_label(v: float, a: float, d: float) -> Dict[str, str]:
    """获取VAD的人类可读标签"""
    valence_label = "正面" if v > 0.3 else ("负面" if v < -0.3 else "中性")
    arousal_label = "激动" if a > 0.3 else ("平静" if a < -0.3 else "温和")
    dominance_label = "主导" if d > 0.3 else ("顺从" if d < -0.3 else "中立")

    return {
        "valence": valence_label,
        "arousal": arousal_label,
        "dominance": dominance_label,
        "valence_value": round(v, 2),
        "arousal_value": round(a, 2),
        "dominance_value": round(d, 2),
    }


# ============================================================
# 复合情感计算
# ============================================================

def calculate_compound_emotion(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    计算复合情感

    人类情感往往是复合的，如:
    - 喜极而泣: joy + sadness
    - 怒其不争: anger + sadness (接近envy)
    - 受宠若惊: surprise + joy + trust
    - 如释重负: relief + joy
    """
    if len(emotions) < 2:
        return emotions

    compound_emotions = {}
    emotion_list = list(emotions.items())

    # 检查常见复合情感
    for i, (e1, v1) in enumerate(emotion_list):
        for e2, v2 in emotion_list[i+1:]:
            combined = min(v1, v2)  # 取较小值作为复合强度

            # joy + sadness = 悲喜交加 ( bittersweet )
            if {e1, e2} == {"joy", "sadness"}:
                compound_emotions["bittersweet"] = combined * 0.8

            # joy + trust = love
            if {e1, e2} == {"joy", "trust"}:
                compound_emotions["love"] = combined * 1.0

            # joy + anticipation = optimism/hope
            if {e1, e2} == {"joy", "anticipation"}:
                compound_emotions["hope"] = combined * 1.0

            # sadness + fear = despair
            if {e1, e2} == {"sadness", "fear"}:
                compound_emotions["despair"] = combined * 0.9

            # sadness + anger = grief/regret (自我相关)
            if {e1, e2} == {"sadness", "anger"}:
                compound_emotions["regret"] = combined * 0.7

            # fear + anticipation = anxiety
            if {e1, e2} == {"fear", "anticipation"}:
                compound_emotions["anxiety"] = combined * 0.9

            # surprise + joy = excitement
            if {e1, e2} == {"surprise", "joy"}:
                compound_emotions["ecstasy"] = combined * 1.1

            # surprise + fear = shock
            if {e1, e2} == {"surprise", "fear"}:
                compound_emotions["terror"] = combined * 1.0

            # trust + joy = gratitude
            if {e1, e2} == {"trust", "joy"}:
                compound_emotions["gratitude"] = combined * 1.0

            # anger + disgust = contempt
            if {e1, e2} == {"anger", "disgust"}:
                compound_emotions["contempt"] = combined * 1.0

            # 新增复合情感组合
            # anger + fear = 恐慌
            if {e1, e2} == {"anger", "fear"}:
                compound_emotions["angry_fear"] = combined * 0.85

            # sadness + guilt = 自责愧疚
            if {e1, e2} == {"sadness", "guilt"}:
                compound_emotions["sadness_guilt"] = combined * 0.9

            # love + envy = 吃醋
            if {e1, e2} == {"love", "envy"}:
                compound_emotions["jealous_love"] = combined * 0.8

            # surprise + sadness = 震惊否认
            if {e1, e2} == {"surprise", "sadness"}:
                compound_emotions["shock_denial"] = combined * 0.85

            # relief + sadness = 如释重负又失落
            if {e1, e2} == {"relief", "sadness"}:
                compound_emotions["relief_sadness"] = combined * 0.75

            # anger + sadness = 愤怒悲伤混合
            if {e1, e2} == {"anger", "sadness"}:
                compound_emotions["sadness_guilt"] = combined * 0.8

            # joy + guilt = 开心但愧疚
            if {e1, e2} == {"joy", "guilt"}:
                compound_emotions["bittersweet"] = combined * 0.7

    # 检查三元复合情感
    if len(emotions) >= 3:
        emotion_set = set(emotions.keys())

        # joy + sadness + trust = painful_joy (喜极而泣)
        if {"joy", "sadness", "trust"}.issubset(emotion_set):
            compound_emotions["painful_joy"] = min(emotions["joy"], emotions["sadness"], emotions["trust"]) * 1.2

        # joy + anticipation + fear = hope_fear
        if {"joy", "anticipation", "fear"}.issubset(emotion_set):
            compound_emotions["hope_fear"] = min(emotions["joy"], emotions["anticipation"], emotions["fear"]) * 0.9

        # love + admiration + joy = love_admiration
        if {"love", "admiration", "joy"}.issubset(emotion_set):
            compound_emotions["love_admiration"] = min(emotions["love"], emotions["admiration"], emotions["joy"]) * 1.1

        # sadness + anger + guilt = frustration_hopelessness
        if {"sadness", "anger", "guilt"}.issubset(emotion_set):
            compound_emotions["frustration_hopelessness"] = min(emotions["sadness"], emotions["anger"], emotions["guilt"]) * 0.85

    return compound_emotions
