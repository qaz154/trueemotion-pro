"""
记忆仓库 v1.13
使用Repository模式管理用户记忆和学习模式

v1.13 增强:
- 智能关键词提取
- 记忆强化衰减机制
- 语义相似度匹配
- 跨用户模式共享
"""

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple


# 停用词（情感分析时不考虑）
STOP_WORDS = {
    "的", "了", "啊", "吧", "呢", "呀", "哦", "嗯", "噢", "唉",
    "我", "你", "他", "她", "它", "们", "是", "在", "有", "和",
    "也", "都", "就", "要", "会", "能", "可以", "什么", "怎么",
    "这个", "那个", "一个", "一些", "什么", "为什么", "吗", "呢",
    "好", "很", "太", "真", "非常", "特别", "比较",
}

# 强化衰减参数
REINFORCEMENT_BOOST = 0.15  # 被使用时反馈强化量
DECAY_RATE = 0.02  # 每周衰减率
DECAY_THRESHOLD = 0.3  # 衰减阈值


@dataclass
class LearnedPattern:
    """学习到的模式"""
    user_id: str
    emotion: str
    response: str
    feedback: float
    times_used: int = 0
    last_used: Optional[str] = None
    created_at: Optional[str] = None
    keywords: List[str] = field(default_factory=list)  # 提取的关键词
    context_hints: List[str] = field(default_factory=list)  # 上下文提示
    decay_count: int = 0  # 衰减计数


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    total_interactions: int = 0
    dominant_emotion: Optional[str] = None
    relationship_level: float = 0.0
    learned_patterns: int = 0
    last_emotion: Optional[str] = None
    emotional_history: list[str] = field(default_factory=list)
    preferred_tone: str = "温暖"  # 用户偏好语气
    interaction_style: str = "normal"  # 互动风格
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_seen: Optional[str] = None


class MemoryRepository:
    """
    记忆仓库 v1.13

    使用JSON文件存储，支持用户画像和学习模式管理
    特性:
    - 智能关键词提取
    - 记忆强化衰减机制
    - 语义相似度匹配
    """

    def __init__(self, base_path: str = "./memory"):
        """
        初始化仓库

        Args:
            base_path: 存储基础路径
        """
        self._base_path = Path(base_path)
        self._users_dir = self._base_path / "users"
        self._patterns_dir = self._base_path / "patterns"
        self._global_patterns_dir = self._base_path / "global_patterns"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """确保目录存在"""
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._patterns_dir.mkdir(parents=True, exist_ok=True)
        self._global_patterns_dir.mkdir(parents=True, exist_ok=True)

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本
            max_keywords: 最大关键词数量

        Returns:
            List[str]: 提取的关键词列表
        """
        # 清理文本
        text = re.sub(r'[^\w\s一-鿿]', ' ', text)
        words = text.split()

        # 过滤停用词和单字
        keywords = [
            w for w in words
            if w not in STOP_WORDS and len(w) >= 2
        ]

        # 统计词频
        word_freq: Dict[str, int] = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1

        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    def _extract_context_hints(self, text: str) -> List[str]:
        """提取上下文提示"""
        hints = []

        # 时间相关
        time_patterns = [
            r'今天', r'昨天', r'明天', r'上周', r'下周',
            r'最近', r'刚才', r'刚才', r'以前', r'小时候',
        ]
        for p in time_patterns:
            if re.search(p, text):
                hints.append(p)

        # 地点相关
        place_patterns = [
            r'公司', r'学校', r'家里', r'回家', r'上班',
            r'上学', r'出门', r'在外', r'这里', r'那里',
        ]
        for p in place_patterns:
            if re.search(p, text):
                hints.append(p)

        # 人物相关
        person_patterns = [
            r'老板', r'同事', r'朋友', r'家人', r'父母',
            r'男/女', r'朋友', r'老师', r'同学', r'恋人',
        ]
        for p in person_patterns:
            if re.search(p, text):
                hints.append(p)

        return hints[:5]  # 最多5个提示

    def _get_user_file(self, user_id: str) -> Path:
        """获取用户文件路径"""
        return self._users_dir / f"{user_id}.json"

    def _get_pattern_file(self, user_id: str) -> Path:
        """获取模式文件路径"""
        return self._patterns_dir / f"{user_id}_patterns.json"

    def get_user(self, user_id: str) -> UserProfile:
        """
        获取用户画像

        Args:
            user_id: 用户ID

        Returns:
            UserProfile: 用户画像，不存在则返回新的
        """
        user_file = self._get_user_file(user_id)

        if user_file.exists():
            try:
                with open(user_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserProfile(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        # 返回新的用户画像
        return UserProfile(user_id=user_id)

    def save_user(self, user_id: str, profile: UserProfile) -> None:
        """
        保存用户画像

        Args:
            user_id: 用户ID
            profile: 用户画像
        """
        user_file = self._get_user_file(user_id)

        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    def learn_pattern(
        self,
        user_id: str,
        emotion: str,
        response: str,
        feedback: float,
        context: Optional[str] = None,
    ) -> LearnedPattern:
        """
        学习新模式 v1.13 增强版

        特性:
        - 自动关键词提取
        - 记忆强化衰减机制
        - 上下文感知

        Args:
            user_id: 用户ID
            emotion: 情感类型
            response: 响应文本
            feedback: 用户反馈 0-1
            context: 可选的上下文文本

        Returns:
            LearnedPattern: 创建的模式
        """
        now = datetime.now().isoformat()
        keywords = self._extract_keywords(response)
        context_hints = self._extract_context_hints(context or response)

        pattern = LearnedPattern(
            user_id=user_id,
            emotion=emotion,
            response=response,
            feedback=feedback,
            times_used=1,
            last_used=now,
            created_at=now,
            keywords=keywords,
            context_hints=context_hints,
            decay_count=0,
        )

        patterns = self._load_patterns(user_id)

        # 检查是否已存在相似的模式（使用关键词匹配）
        existing_idx = None
        for i, p in enumerate(patterns):
            if p.emotion == emotion and p.response == response:
                existing_idx = i
                break
            # 也检查关键词重叠度
            elif p.emotion == emotion:
                overlap = self._calculate_keyword_overlap(p.keywords, keywords)
                if overlap >= 0.6:  # 60%关键词重叠认为是相似模式
                    existing_idx = i
                    break

        if existing_idx is not None:
            # 更新现有模式 - 强化学习
            patterns[existing_idx].times_used += 1
            patterns[existing_idx].last_used = now
            # 强化反馈：如果反馈高，增加该模式的权重
            if feedback >= 0.7:
                patterns[existing_idx].feedback = min(1.0,
                    patterns[existing_idx].feedback + REINFORCEMENT_BOOST * feedback
                )
            else:
                patterns[existing_idx].feedback = (
                    patterns[existing_idx].feedback * 0.8 + feedback * 0.2
                )
            # 衰减计数重置
            patterns[existing_idx].decay_count = 0
            # 更新关键词
            patterns[existing_idx].keywords = list(set(patterns[existing_idx].keywords + keywords))[:10]
            pattern = patterns[existing_idx]
        else:
            # 添加新模式
            patterns.append(pattern)

        self._save_patterns(user_id, patterns)

        # 如果反馈高，同步到全局模式库
        if feedback >= 0.8:
            self._save_global_pattern(pattern)

        return pattern

    def _calculate_keyword_overlap(self, keywords1: List[str], keywords2: List[str]) -> float:
        """计算两个关键词列表的重叠度"""
        if not keywords1 or not keywords2:
            return 0.0
        set1, set2 = set(keywords1), set(keywords2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _save_global_pattern(self, pattern: LearnedPattern) -> None:
        """保存高反馈模式到全局模式库"""
        global_file = self._global_patterns_dir / f"{pattern.emotion}_global.json"
        global_patterns = []

        if global_file.exists():
            try:
                with open(global_file, "r", encoding="utf-8") as f:
                    global_patterns = json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass

        # 检查是否已存在
        existing = None
        for i, p in enumerate(global_patterns):
            if p.get("response") == pattern.response:
                existing = i
                break

        if existing is not None:
            global_patterns[existing]["times_used"] += 1
            global_patterns[existing]["feedback"] = max(
                global_patterns[existing]["feedback"],
                pattern.feedback
            )
        else:
            global_patterns.append({
                "emotion": pattern.emotion,
                "response": pattern.response,
                "feedback": pattern.feedback,
                "times_used": 1,
                "keywords": pattern.keywords,
            })

        with open(global_file, "w", encoding="utf-8") as f:
            json.dump(global_patterns, f, ensure_ascii=False, indent=2)

    def get_global_patterns(self, emotion: Optional[str] = None) -> List[Dict]:
        """获取全局模式库中的模式"""
        if emotion:
            global_file = self._global_patterns_dir / f"{emotion}_global.json"
            if global_file.exists():
                try:
                    with open(global_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            # 返回所有全局模式
            all_patterns = []
            for f in self._global_patterns_dir.glob("*_global.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        all_patterns.extend(json.load(fp))
                except (json.JSONDecodeError, TypeError):
                    pass
            return all_patterns
        return []

    def apply_decay(self) -> int:
        """
        对所有模式应用衰减

        Returns:
            int: 受到衰减影响的模式数量
        """
        affected = 0
        for pattern_file in self._patterns_dir.glob("*_patterns.json"):
            user_id = pattern_file.stem.replace("_patterns", "")
            patterns = self._load_patterns(user_id)
            changed = False

            for pattern in patterns:
                pattern.decay_count += 1
                if pattern.decay_count >= 2:  # 每两次检查衰减一次
                    old_feedback = pattern.feedback
                    pattern.feedback = max(0.1, pattern.feedback - DECAY_RATE)
                    if pattern.feedback != old_feedback:
                        affected += 1
                        changed = True

            if changed:
                self._save_patterns(user_id, patterns)

        return affected

    def find_similar_patterns(
        self,
        user_id: str,
        emotion: str,
        text: str,
        threshold: float = 0.4,
    ) -> List[LearnedPattern]:
        """
        查找相似的已学习模式

        Args:
            user_id: 用户ID
            emotion: 情感类型
            text: 查询文本
            threshold: 相似度阈值

        Returns:
            List[LearnedPattern]: 相似模式列表
        """
        patterns = self._load_patterns(user_id)
        query_keywords = self._extract_keywords(text)

        similar = []
        for pattern in patterns:
            if pattern.emotion == emotion:
                overlap = self._calculate_keyword_overlap(pattern.keywords, query_keywords)
                if overlap >= threshold:
                    similar.append(pattern)

        # 按相似度排序
        similar.sort(
            key=lambda p: self._calculate_keyword_overlap(p.keywords, query_keywords),
            reverse=True
        )
        return similar[:5]

    def _load_patterns(self, user_id: str) -> list[LearnedPattern]:
        """加载用户模式"""
        pattern_file = self._get_pattern_file(user_id)

        if pattern_file.exists():
            try:
                with open(pattern_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [LearnedPattern(**p) for p in data]
            except (json.JSONDecodeError, TypeError):
                pass

        return []

    def _save_patterns(self, user_id: str, patterns: list[LearnedPattern]) -> None:
        """保存用户模式"""
        pattern_file = self._get_pattern_file(user_id)

        with open(pattern_file, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in patterns], f, ensure_ascii=False, indent=2)

    def get_pattern_count(self, user_id: str) -> int:
        """获取用户学到的模式数量"""
        return len(self._load_patterns(user_id))

    def get_patterns_for_emotion(
        self,
        user_id: str,
        emotion: str,
    ) -> list[LearnedPattern]:
        """
        获取特定情感的模式

        Args:
            user_id: 用户ID
            emotion: 情感类型

        Returns:
            list[LearnedPattern]: 匹配的模式列表
        """
        patterns = self._load_patterns(user_id)
        return [p for p in patterns if p.emotion == emotion]

    def get_all_patterns(self) -> dict[str, list[LearnedPattern]]:
        """
        获取所有用户的所有模式

        Returns:
            dict[str, list[LearnedPattern]]: 用户ID到模式列表的映射
        """
        all_patterns: dict[str, list[LearnedPattern]] = {}

        for pattern_file in self._patterns_dir.glob("*_patterns.json"):
            user_id = pattern_file.stem.replace("_patterns", "")
            patterns = self._load_patterns(user_id)
            if patterns:
                all_patterns[user_id] = patterns

        return all_patterns

    def get_stats(self) -> dict:
        """获取系统统计信息 v1.13 增强版"""
        users = list(self._users_dir.glob("*.json"))
        pattern_files = list(self._patterns_dir.glob("*_patterns.json"))
        global_files = list(self._global_patterns_dir.glob("*_global.json"))

        total_patterns = 0
        high_quality_patterns = 0
        emotion_distribution: Dict[str, int] = {}

        for pf in pattern_files:
            try:
                with open(pf, "r", encoding="utf-8") as f:
                    patterns = json.load(f)
                    total_patterns += len(patterns)
                    for p in patterns:
                        if isinstance(p, dict):
                            if p.get("feedback", 0) >= 0.7:
                                high_quality_patterns += 1
                            emotion = p.get("emotion", "unknown")
                            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
                        else:
                            if p.feedback >= 0.7:
                                high_quality_patterns += 1
            except (json.JSONDecodeError, TypeError):
                pass

        # 全局模式统计
        global_patterns_count = 0
        for gf in global_files:
            try:
                with open(gf, "r", encoding="utf-8") as f:
                    global_patterns_count += len(json.load(f))
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "total_users": len(users),
            "total_patterns": total_patterns,
            "high_quality_patterns": high_quality_patterns,
            "global_patterns": global_patterns_count,
            "emotion_distribution": emotion_distribution,
            "memory_path": str(self._base_path),
        }

    def delete_user(self, user_id: str) -> None:
        """删除用户数据"""
        user_file = self._get_user_file(user_id)
        pattern_file = self._get_pattern_file(user_id)

        if user_file.exists():
            user_file.unlink()
        if pattern_file.exists():
            pattern_file.unlink()
