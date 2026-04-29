# -*- coding: utf-8 -*-
"""
情感自动进化系统
================

让AI情感系统能够像真人一样从经历中学习和进化。

核心机制：
1. 情感学习 - 从对话中提取情感模式
2. 准则形成 - 将重复验证的模式固化为准则
3. 遗忘机制 - 模拟艾宾浩斯遗忘曲线
4. 自我反思 - AI反思情感反应是否合适
5. 进化追踪 - 记录成长历史
"""

import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import copy


# ==================== 数据结构 ====================

@dataclass
class Experience:
    """经历"""
    timestamp: datetime
    text: str
    emotion: str
    intensity: float
    response: str
    response_emotion: str
    feedback: Optional[float] = None  # 用户反馈 (-1 ~ 1)
    context: List[str] = field(default_factory=list)


@dataclass
class LearnedRule:
    """学习到的准则"""
    id: str
    pattern: str  # 触发模式
    emotion: str  # 对应情感
    confidence: float  # 置信度 (0-1)
    success_count: int  # 成功次数
    failure_count: int  # 失败次数
    created_at: datetime
    last_validated: datetime
    source: str  # "experience", "reflection", "teaching"
    description: str = ""
    is_core: bool = False  # 核心准则（不可遗忘）

    def get_success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def should_upgrade(self) -> bool:
        """是否应该升级为核心准则"""
        return (self.success_count >= 10 and
                self.get_success_rate() > 0.85 and
                not self.is_core)

    def should_deprecate(self) -> bool:
        """是否应该淘汰"""
        return (self.failure_count >= 5 and
                self.get_success_rate() < 0.3)


@dataclass
class EmotionState:
    """当前情感状态"""
    current_emotion: str = "neutral"
    intensity: float = 0.0
    vad: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    recent_emotions: List[Tuple[str, float]] = field(default_factory=list)
    emotion_streak: int = 0  # 情感重复次数
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionSnapshot:
    """进化快照"""
    timestamp: datetime
    rule_count: int
    experience_count: int
    avg_confidence: float
    top_emotions: List[Tuple[str, float]]
    new_rules: List[str]
    deprecated_rules: List[str]


# ==================== 遗忘曲线 ====================

class EbbinghausForgetting:
    """
    艾宾浩斯遗忘曲线模拟

    记忆强度 = 重要性 × e^(-衰减率 × 时间) × 加权系数
    """

    # 艾宾浩斯遗忘曲线时间点（分钟）
    TIME_POINTS = [1, 5, 10, 30, 60, 480, 1440, 10080]  # 1分,5分,10分...1周
    RETENTION_RATES = [0.58, 0.44, 0.36, 0.34, 0.28, 0.25, 0.21, 0.16]

    @classmethod
    def get_retention(cls, minutes_elapsed: float, importance: float = 0.5) -> float:
        """
        计算记忆留存率

        Args:
            minutes_elapsed: 经过的时间（分钟）
            importance: 重要性 (0-1)，高重要性记忆衰减更慢
        """
        if minutes_elapsed <= 0:
            return 1.0

        # 基础衰减
        if minutes_elapsed <= 1:
            return cls.RETENTION_RATES[0]
        elif minutes_elapsed <= 5:
            return cls.RETENTION_RATES[1]
        elif minutes_elapsed <= 10:
            return cls.RETENTION_RATES[2]
        elif minutes_elapsed <= 30:
            return cls.RETENTION_RATES[3]
        elif minutes_elapsed <= 60:
            return cls.RETENTION_RATES[4]
        elif minutes_elapsed <= 480:
            return cls.RETENTION_RATES[5]
        elif minutes_elapsed <= 1440:
            return cls.RETENTION_RATES[6]
        else:
            return cls.RETENTION_RATES[7]

        # 重要性调整：高重要性记忆衰减更慢
        importance_factor = 0.5 + importance * 0.5
        return base_retention * importance_factor

    @classmethod
    def should_forget(cls, rule: LearnedRule, current_time: datetime) -> bool:
        """判断准则是否应该被遗忘"""
        if rule.is_core:
            return False

        minutes_elapsed = (current_time - rule.last_validated).total_seconds() / 60
        retention = cls.get_retention(minutes_elapsed, importance=rule.get_success_rate())

        # 如果留存率低于阈值，考虑遗忘
        return retention < 0.3 and rule.get_success_rate() < 0.6


# ==================== 情感自动进化系统 ====================

class EmotionEvolution:
    """
    情感自动进化系统

    核心功能：
    1. 经历记录 - 记录每个交互
    2. 模式发现 - 从重复经历中发现规律
    3. 准则形成 - 将验证通过的模式固化为准则
    4. 遗忘管理 - 遗忘低价值准则
    5. 自我反思 - AI主动反思情感反应
    6. 进化追踪 - 记录成长历程
    """

    # 核心准则（永不遗忘）
    CORE_RULES = [
        {"pattern": "用户难过", "emotion": "empathy", "description": "用户难过时表达共情"},
        {"pattern": "用户生气", "emotion": "calm", "description": "用户生气时先冷静"},
        {"pattern": "用户开心", "emotion": "share_joy", "description": "用户开心时分享喜悦"},
        {"pattern": "用户害怕", "emotion": "reassure", "description": "用户害怕时安抚"},
        {"pattern": "用户感谢", "emotion": "humble", "description": "用户感谢时谦逊"},
        {"pattern": "用户批评", "emotion": "accept", "description": "用户批评时接受"},
    ]

    def __init__(self, data_dir: str = "./evolution_data"):
        self.data_dir = data_dir

        # 经历记录
        self.experiences: List[Experience] = []

        # 学习到的准则
        self.rules: Dict[str, LearnedRule] = {}

        # 当前情感状态
        self.state = EmotionState()

        # 进化历史
        self.snapshots: List[EvolutionSnapshot] = []

        # 统计
        self.stats = {
            "total_interactions": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "patterns_discovered": 0,
            "rules_formed": 0,
            "rules_deprecated": 0,
        }

        # 初始化核心准则
        self._init_core_rules()

    def _init_core_rules(self):
        """初始化核心准则"""
        for i, core in enumerate(self.CORE_RULES):
            rule_id = f"core_{i}"
            self.rules[rule_id] = LearnedRule(
                id=rule_id,
                pattern=core["pattern"],
                emotion=core["emotion"],
                confidence=0.95,
                success_count=100,
                failure_count=0,
                created_at=datetime.now(),
                last_validated=datetime.now(),
                source="core",
                description=core["description"],
                is_core=True
            )

    # ==================== 核心接口 ====================

    def record_interaction(
        self,
        user_text: str,
        user_emotion: str,
        user_intensity: float,
        response: str,
        response_emotion: str,
        feedback: Optional[float] = None,
        context: Optional[List[str]] = None
    ) -> None:
        """
        记录一次交互

        Args:
            user_text: 用户发言
            user_emotion: 用户情感
            user_intensity: 情感强度
            response: AI回复
            response_emotion: AI回复的情感
            feedback: 用户反馈 (-1 ~ 1)
            context: 上下文
        """
        experience = Experience(
            timestamp=datetime.now(),
            text=user_text,
            emotion=user_emotion,
            intensity=user_intensity,
            response=response,
            response_emotion=response_emotion,
            feedback=feedback,
            context=context or []
        )

        self.experiences.append(experience)
        self.stats["total_interactions"] += 1

        # 记录反馈
        if feedback is not None:
            if feedback > 0:
                self.stats["positive_feedback"] += 1
            elif feedback < 0:
                self.stats["negative_feedback"] += 1

            # 根据反馈更新相关准则
            self._update_rules_from_feedback(experience)

        # 检查是否发现新模式
        self._check_for_pattern(experience)

    def get_response_emotion(self, user_emotion: str, user_text: str, context: List[str]) -> Tuple[str, float]:
        """
        获取合适的响应情感

        Returns:
            (response_emotion, confidence)
        """
        # 1. 先检查核心准则
        for rule in self.rules.values():
            if rule.is_core:
                if self._pattern_matches(rule.pattern, user_text, user_emotion):
                    return rule.emotion, rule.confidence

        # 2. 检查学习到的准则
        best_match = None
        best_confidence = 0.0

        for rule in self.rules.values():
            if self._pattern_matches(rule.pattern, user_text, user_emotion):
                if rule.confidence > best_confidence:
                    best_match = rule
                    best_confidence = rule.confidence

        if best_match:
            return best_match.emotion, best_match.confidence

        # 3. 默认响应
        return self._get_default_response(user_emotion)

    def evolve(self) -> Dict[str, Any]:
        """
        执行一次进化迭代

        Returns:
            进化报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "new_rules": [],
            "deprecated_rules": [],
            "forgotten_rules": [],
            "reflections": []
        }

        # 1. 模式发现
        new_patterns = self._discover_patterns()
        report["actions"].append(f"发现 {len(new_patterns)} 个新模式")
        self.stats["patterns_discovered"] += len(new_patterns)

        # 2. 准则形成
        for pattern in new_patterns:
            rule = self._form_rule(pattern)
            if rule:
                self.rules[rule.id] = rule
                report["new_rules"].append(rule.pattern)
                self.stats["rules_formed"] += 1

        # 3. 遗忘管理
        forgotten = self._manage_forgetting()
        report["forgotten_rules"] = forgotten

        # 4. 准则验证和更新
        self._validate_rules()

        # 5. 自我反思
        reflections = self._self_reflect()
        report["reflections"] = reflections

        # 6. 生成快照
        snapshot = self._create_snapshot(report)
        self.snapshots.append(snapshot)

        return report

    # ==================== 内部方法 ====================

    def _pattern_matches(self, pattern: str, text: str, emotion: str) -> bool:
        """检查模式是否匹配"""
        pattern_lower = pattern.lower()

        # 情感匹配
        if pattern_lower in emotion.lower():
            return True

        # 文本匹配
        if pattern_lower in text.lower():
            return True

        # 关键词匹配
        keywords = {
            "难过": ["难过", "伤心", "悲伤", "失落"],
            "生气": ["生气", "愤怒", "气", "恼火"],
            "开心": ["开心", "高兴", "快乐", "棒"],
            "害怕": ["害怕", "担心", "怕", "慌"],
            "感谢": ["谢谢", "感谢", "感恩"],
            "批评": ["不对", "不好", "错", "问题"],
        }

        for p_kw in keywords:
            if p_kw in pattern_lower:
                for t_kw in keywords[p_kw]:
                    if t_kw in text.lower():
                        return True

        return False

    def _get_default_response(self, user_emotion: str) -> Tuple[str, float]:
        """获取默认响应"""
        defaults = {
            "joy": ("celebrate", 0.5),
            "sadness": ("comfort", 0.5),
            "anger": ("calm", 0.5),
            "fear": ("reassure", 0.5),
            "surprise": ("acknowledge", 0.5),
            "anticipation": ("encourage", 0.5),
            "trust": ("appreciate", 0.5),
            "disgust": ("understand", 0.5),
        }
        return defaults.get(user_emotion, ("neutral", 0.3))

    def _update_rules_from_feedback(self, experience: Experience) -> None:
        """根据反馈更新准则"""
        if experience.feedback is None:
            return

        # 找到相关的准则
        for rule in self.rules.values():
            if self._pattern_matches(rule.pattern, experience.text, experience.emotion):
                if experience.feedback > 0:
                    rule.success_count += 1
                else:
                    rule.failure_count += 1

                rule.last_validated = datetime.now()

                # 更新置信度
                total = rule.success_count + rule.failure_count
                rule.confidence = rule.success_count / total

                # 检查是否应该升级为核心准则
                if rule.should_upgrade():
                    rule.is_core = True

                # 检查是否应该淘汰
                if rule.should_deprecate():
                    rule.confidence = 0.1

    def _check_for_pattern(self, experience: Experience) -> None:
        """检查是否发现新模式"""
        # 简单实现：检查是否有重复的经历
        for exp in self.experiences[:-1]:
            # 检查相似的情感+强度组合
            if (exp.emotion == experience.emotion and
                abs(exp.intensity - experience.intensity) < 0.2 and
                exp.response_emotion == experience.response_emotion):
                # 发现重复模式！
                pass

    def _discover_patterns(self) -> List[Dict]:
        """从经历中发现模式"""
        patterns = []

        # 按情感分组
        by_emotion = defaultdict(list)
        for exp in self.experiences:
            by_emotion[exp.emotion].append(exp)

        # 对每种情感，找最常见的响应
        for emotion, exps in by_emotion.items():
            if len(exps) < 3:
                continue

            # 统计响应情感
            response_counts = Counter(e.response_emotion for e in exps)
            most_common = response_counts.most_common(1)[0]

            # 如果某种响应模式出现超过3次，认为是模式
            if most_common[1] >= 3:
                patterns.append({
                    "pattern": f"{emotion}_response",
                    "trigger_emotion": emotion,
                    "response_emotion": most_common[0],
                    "confidence": most_common[1] / len(exps),
                    "count": most_common[1]
                })

        return patterns

    def _form_rule(self, pattern: Dict) -> Optional[LearnedRule]:
        """将模式固化为准则"""
        # 检查是否已存在相同的模式
        for rule in self.rules.values():
            if (not rule.is_core and
                rule.pattern == pattern["pattern"] and
                rule.emotion == pattern["response_emotion"]):
                # 已存在，更新计数
                rule.success_count += pattern["count"]
                rule.last_validated = datetime.now()
                rule.confidence = min(0.99, rule.get_success_rate())
                return None  # 不创建新准则

        pattern_id = f"learned_{len(self.rules)}"
        rule = LearnedRule(
            id=pattern_id,
            pattern=pattern["pattern"],
            emotion=pattern["response_emotion"],
            confidence=pattern["confidence"],
            success_count=pattern["count"],
            failure_count=0,
            created_at=datetime.now(),
            last_validated=datetime.now(),
            source="experience",
            description=f"从{pattern['count']}次经历中学习"
        )
        return rule

    def _manage_forgetting(self) -> List[str]:
        """遗忘管理"""
        forgotten = []
        current_time = datetime.now()

        for rule_id in list(self.rules.keys()):
            rule = self.rules[rule_id]

            if EbbinghausForgetting.should_forget(rule, current_time):
                del self.rules[rule_id]
                forgotten.append(rule.pattern)

        return forgotten

    def _validate_rules(self) -> None:
        """验证准则"""
        for rule in self.rules.values():
            if rule.is_core:
                continue

            # 验证成功率和置信度
            if rule.failure_count > 0:
                success_rate = rule.get_success_rate()
                rule.confidence = success_rate * 0.9

    def _self_reflect(self) -> List[str]:
        """自我反思"""
        reflections = []

        if len(self.experiences) < 5:
            return reflections

        recent = self.experiences[-10:]

        # 检查是否有负面反馈
        negative_count = sum(1 for e in recent if e.feedback and e.feedback < 0)
        if negative_count >= 3:
            reflections.append(f"近期有{negative_count}次负面反馈，需要反思响应策略")

        # 检查是否有情感重复但反馈不同的情况
        emotion_feedbacks = defaultdict(list)
        for e in recent:
            if e.feedback is not None:
                emotion_feedbacks[e.emotion].append(e.feedback)

        for emotion, feedbacks in emotion_feedbacks.items():
            if len(feedbacks) >= 2:
                avg = sum(feedbacks) / len(feedbacks)
                if avg < 0.3:
                    reflections.append(f"对{emotion}情感的响应需要改进，平均反馈仅{avg:.2f}")

        # 检查准则置信度是否下降
        low_confidence = [r for r in self.rules.values()
                         if not r.is_core and r.confidence < 0.4]
        if len(low_confidence) >= 3:
            reflections.append(f"有{len(low_confidence)}条准则置信度偏低，考虑更新或遗忘")

        return reflections

    def _create_snapshot(self, report: Dict) -> EvolutionSnapshot:
        """创建进化快照"""
        # 统计情感
        emotion_counts = Counter(e.emotion for e in self.experiences[-50:])
        top_emotions = emotion_counts.most_common(5)

        # 新准则和废弃准则
        new_rules = report.get("new_rules", [])
        deprecated_rules = report.get("deprecated_rules", [])

        avg_confidence = sum(r.confidence for r in self.rules.values()) / len(self.rules) if self.rules else 0

        return EvolutionSnapshot(
            timestamp=datetime.now(),
            rule_count=len(self.rules),
            experience_count=len(self.experiences),
            avg_confidence=avg_confidence,
            top_emotions=top_emotions,
            new_rules=new_rules,
            deprecated_rules=deprecated_rules
        )

    # ==================== 工具方法 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_interactions": self.stats["total_interactions"],
            "positive_feedback": self.stats["positive_feedback"],
            "negative_feedback": self.stats["negative_feedback"],
            "patterns_discovered": self.stats["patterns_discovered"],
            "rules_formed": self.stats["rules_formed"],
            "rules_deprecated": self.stats["rules_deprecated"],
            "current_rules": len(self.rules),
            "core_rules": sum(1 for r in self.rules.values() if r.is_core),
            "learned_rules": sum(1 for r in self.rules.values() if not r.is_core),
        }

    def get_rules(self) -> List[Dict]:
        """获取所有准则"""
        return [
            {
                **asdict(rule),
                "created_at": rule.created_at.isoformat(),
                "last_validated": rule.last_validated.isoformat(),
                "success_rate": rule.get_success_rate()
            }
            for rule in self.rules.values()
        ]

    def get_recent_experiences(self, n: int = 10) -> List[Dict]:
        """获取最近的经历"""
        recent = self.experiences[-n:]
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "text": e.text,
                "emotion": e.emotion,
                "intensity": e.intensity,
                "response": e.response,
                "response_emotion": e.response_emotion,
                "feedback": e.feedback
            }
            for e in reversed(recent)
        ]

    def save(self, path: Optional[str] = None) -> None:
        """保存进化状态"""
        if path is None:
            path = f"{self.data_dir}/evolution_state.json"

        data = {
            "rules": self.get_rules(),
            "stats": self.stats,
            "snapshot_count": len(self.snapshots)
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: Optional[str] = None) -> None:
        """加载进化状态"""
        if path is None:
            path = f"{self.data_dir}/evolution_state.json"

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.stats = data.get("stats", self.stats)
        except FileNotFoundError:
            pass

    def reset(self) -> None:
        """重置进化状态"""
        self.experiences.clear()
        self.snapshots.clear()
        self.rules.clear()
        self._init_core_rules()
        self.stats = {k: 0 for k in self.stats}


# ==================== 自动进化管理器 ====================

class EvolutionManager:
    """
    自动进化管理器

    自动化执行进化循环
    """

    def __init__(self):
        self.evolution = EmotionEvolution()

        # 进化配置
        self.auto_evolve = True
        self.evolve_interval = 10  # 每N次交互后进化一次
        self.last_evolve_time = time.time()

    def process_interaction(
        self,
        user_text: str,
        user_emotion: str,
        user_intensity: float,
        response: str,
        response_emotion: str,
        feedback: Optional[float] = None
    ) -> str:
        """
        处理交互并自动进化

        Returns:
            AI回复
        """
        # 1. 记录交互
        self.evolution.record_interaction(
            user_text=user_text,
            user_emotion=user_emotion,
            user_intensity=user_intensity,
            response=response,
            response_emotion=response_emotion,
            feedback=feedback
        )

        # 2. 检查是否需要自动进化
        if self.auto_evolve:
            interactions = self.evolution.stats["total_interactions"]
            if interactions % self.evolve_interval == 0:
                report = self.evolution.evolve()
                if report["new_rules"] or report["reflections"]:
                    print(f"  [进化] 新准则: {len(report['new_rules'])}, 反思: {len(report['reflections'])}")

        # 3. 获取响应情感
        response_emotion, confidence = self.evolution.get_response_emotion(
            user_emotion, user_text, []
        )

        return response_emotion

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "evolution": self.evolution.get_stats(),
            "config": {
                "auto_evolve": self.auto_evolve,
                "evolve_interval": self.evolve_interval
            }
        }


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("情感自动进化系统测试")
    print("=" * 70)

    manager = EvolutionManager()

    # 模拟对话
    dialogues = [
        ("我今天工作很累...", "sadness", 0.7, "辛苦了，要注意休息啊", "care"),
        ("项目又延期了，好烦", "anger", 0.8, "确实挺烦的，说说怎么回事", "understand"),
        ("项目又延期了，好烦", "anger", 0.8, "别烦了，抱怨没用", "dismiss", -0.5),  # 差评
        ("项目又延期了，好烦", "anger", 0.8, "理解你的感受，我们一起想办法", "understand", 0.8),  # 好评
        ("太棒了！终于完成了！", "joy", 0.9, "恭喜恭喜！太厉害了！", "celebrate"),
        ("担心明天的考试...", "fear", 0.6, "别太紧张，相信自己可以的", "reassure"),
    ]

    print("\n模拟对话进化：")
    for user_text, emotion, intensity, response, resp_emotion in dialogues:
        feedback = None
        if len(dialogues[0]) > 5:
            feedback = dialogues[0][-1]

        result = manager.process_interaction(
            user_text=user_text,
            user_emotion=emotion,
            user_intensity=intensity,
            response=response,
            response_emotion=resp_emotion,
            feedback=feedback
        )
        print(f"  用户: {user_text[:20]}... [{emotion}]")
        print(f"  AI响应情感: {result}")
        print()

    # 显示统计
    print("\n进化统计：")
    stats = manager.evolution.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 显示准则
    print("\n当前准则：")
    rules = manager.evolution.get_rules()
    for rule in rules[:5]:
        print(f"  [{rule['pattern']}] -> {rule['emotion']} (置信度:{rule['confidence']:.2f}, 核心:{rule['is_core']})")

    # 手动进化
    print("\n执行手动进化...")
    report = manager.evolution.evolve()
    print(f"  新准则: {report['new_rules']}")
    print(f"  反思: {report['reflections']}")
