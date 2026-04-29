# -*- coding: utf-8 -*-
"""
TrueEmotion Pro v3.1.0 - 新一代真实情感AI系统
============================================

核心功能：
1. 24种情感识别（HybridEmotionAnalyzer - 增强版规则+神经网络）
2. 自动进化学习（EvolutionManager）
3. 长期记忆（MemorySystem）
4. 有血有肉的情感化回复（TrueEmotionLife Expression）

版本：v3.1.0
"""

__version__ = "3.1.0"

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .models.hybrid_emotion import HybridEmotionAnalyzer, RuleBasedEmotionDetector
from .evolution.emotion_evolution import EvolutionManager, Experience
from .emotion.emotion_output import EmotionOutput
from .emotion.plutchik24 import EMOTION_DEFINITIONS

# TrueEmotionLife 表达模块
from .expression.empathy_engine import AdaptiveEmpathyEngine
from .expression.personality import PersonalityTraits, PersonalityExpressor
from .expression.nl_generator import NaturalLanguageGenerator, NLGConfig


# ==================== 记忆系统 ====================

class MemorySystem:
    """
    长期记忆系统

    管理用户信息、对话历史和关系温度，
    实现跨会话的记忆保持。
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            home = os.path.expanduser("~")
            storage_dir = os.path.join(home, ".openclaw", "data", "trueemotion_memory")

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.user_profiles: Dict[str, Dict] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.learned_patterns: Dict[str, Dict] = {}

        self._load_all()

    def _get_user_file(self, user_id: str) -> Path:
        return self.storage_dir / f"user_{user_id}.json"

    def _get_patterns_file(self) -> Path:
        return self.storage_dir / "learned_patterns.json"

    def _load_all(self):
        """加载所有数据"""
        # 加载用户画像
        for f in self.storage_dir.glob("user_*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    user_id = data.get("user_id", f.stem.replace("user_", ""))
                    self.user_profiles[user_id] = data
            except Exception:
                pass

        # 加载学习到的模式
        patterns_file = self._get_patterns_file()
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as fp:
                    self.learned_patterns = json.load(fp)
            except Exception:
                pass

    def _save_user(self, user_id: str):
        """保存用户数据"""
        if user_id in self.user_profiles:
            user_file = self._get_user_file(user_id)
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles[user_id], f, ensure_ascii=False, indent=2)

    def _save_patterns(self):
        """保存学习到的模式"""
        patterns_file = self._get_patterns_file()
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self.learned_patterns, f, ensure_ascii=False, indent=2)

    def get_or_create_user(self, user_id: str = "default") -> Dict:
        """获取或创建用户"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "conversation_count": 0,
                "total_interactions": 0,
                "common_emotions": {},
                "topics_of_interest": [],
                "sensitive_topics": [],
                "relationship_level": 0.5,
                "preferences": {
                    "warmth": 0.7,
                    "directness": 0.5,
                    "humor": 0.4
                }
            }
            self._save_user(user_id)
        else:
            self.user_profiles[user_id]["last_seen"] = datetime.now().isoformat()

        return self.user_profiles[user_id]

    def record_interaction(self, user_id: str, user_text: str, user_emotion: str,
                          ai_response: str, feedback: float = None):
        """记录一次交互"""
        user = self.get_or_create_user(user_id)

        # 更新用户统计
        user["total_interactions"] += 1

        # 更新情感统计
        emotion_stats = user.get("common_emotions", {})
        emotion_stats[user_emotion] = emotion_stats.get(user_emotion, 0) + 1
        user["common_emotions"] = emotion_stats

        # 记录对话
        session_id = datetime.now().strftime("%Y%m%d")
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []

        self.conversation_history[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "user_emotion": user_emotion,
            "ai_response": ai_response,
            "feedback": feedback
        })

        # 如果对话结束（超过10条），保存会话
        if len(self.conversation_history[session_id]) >= 10:
            user["conversation_count"] += 1
            self._save_user(user_id)

        # 学习模式：如果多次出现相同情感表达
        if emotion_stats.get(user_emotion, 0) >= 3:
            pattern_key = f"{user_emotion}_{user_text[:10]}"
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "emotion": user_emotion,
                    "text_pattern": user_text[:20],
                    "count": 1,
                    "learned_at": datetime.now().isoformat()
                }
            else:
                self.learned_patterns[pattern_key]["count"] += 1

        self._save_user(user_id)
        self._save_patterns()

    def get_user_summary(self, user_id: str = "default") -> Dict:
        """获取用户摘要"""
        user = self.get_or_create_user(user_id)
        return {
            "user_id": user_id,
            "total_interactions": user.get("total_interactions", 0),
            "conversation_count": user.get("conversation_count", 0),
            "dominant_emotion": max(user.get("common_emotions", {}).items(), key=lambda x: x[1])[0] if user.get("common_emotions") else "neutral",
            "relationship_level": user.get("relationship_level", 0.5),
            "last_seen": user.get("last_seen", "never")
        }

    def get_recent_history(self, user_id: str = "default", limit: int = 5) -> List[Dict]:
        """获取最近的对话历史"""
        user = self.get_or_create_user(user_id)
        all_history = []
        for session_id in sorted(self.conversation_history.keys(), reverse=True):
            all_history.extend(self.conversation_history[session_id])
            if len(all_history) >= limit:
                break
        return all_history[:limit]

    def add_learned_pattern(self, pattern: str, emotion: str, confidence: float):
        """添加学习到的模式到检测器"""
        patterns_file = self._get_patterns_file()
        if patterns_file.exists():
            with open(patterns_file, 'r', encoding='utf-8') as f:
                patterns = json.load(f)
        else:
            patterns = {}

        patterns[pattern] = {
            "emotion": emotion,
            "confidence": confidence,
            "learned_at": datetime.now().isoformat(),
            "source": "interaction"
        }

        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)


# ==================== TrueEmotionPro 主类 ====================

class TrueEmotionPro:
    """
    TrueEmotion Pro - 新一代真实情感AI系统

    核心功能：
    - 混合情感检测（规则+神经网络）
    - 自动进化学习
    - 长期记忆
    - 有血有肉回复
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.version = __version__
        self.config = config or {}

        # 核心组件
        self.emotion_analyzer = HybridEmotionAnalyzer()  # 混合情感检测
        self.evolution_manager = EvolutionManager()  # 自动进化
        self.memory_system = MemorySystem()  # 长期记忆

        # TrueEmotionLife 表达组件
        self.empathy_engine = AdaptiveEmpathyEngine()
        self.personality_expressor = PersonalityExpressor()
        self.nl_generator = NaturalLanguageGenerator(
            config=NLGConfig(colloquial_level=0.8, randomness=0.3)
        )

        # 默认个性配置
        self.default_personality = PersonalityTraits(
            warmth=0.75,
            directness=0.65,
            humor=0.45,
            formality=0.2,
            emotionality=0.6,
            patience=0.7
        )

        # 当前用户
        self.current_user_id = "default"

        # 元信息
        self.components = [
            "HybridEmotionAnalyzer (Rules + Neural)",
            "EvolutionManager (Auto-learning)",
            "MemorySystem (Long-term Memory)",
            "TrueEmotionLife Expression (Empathy + Personality + NLG)",
        ]

        # 从记忆系统加载学习到的模式
        self._refresh_learned_patterns()

    def analyze(self, text: str, context: Optional[List[str]] = None,
                learn: bool = False, response: Optional[str] = None,
                feedback: Optional[float] = None,
                user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        综合分析文本

        Args:
            text: 输入文本
            context: 对话上下文
            learn: 是否学习
            response: AI回复（用于学习）
            feedback: 用户反馈
            user_id: 用户ID（用于记忆）

        Returns:
            综合分析结果
        """
        if user_id:
            self.current_user_id = user_id

        # 1. 情感检测
        emotion_result = self.emotion_analyzer.analyze(text)

        if emotion_result is None:
            primary_emotion = "neutral"
            intensity = 0.0
            vad = (0.0, 0.0, 0.0)
            confidence = 0.0
        else:
            primary_emotion = emotion_result.get("primary_emotion", "neutral")
            intensity = emotion_result.get("confidence", 0.5)
            vad = emotion_result.get("vad", (0.0, 0.0, 0.0))
            confidence = emotion_result.get("confidence", 0.5)

        is_irony = False
        irony_prob = 0.0

        # 2. 学习（如果启用）
        if learn and response and self.evolution_manager:
            self._learn(text, primary_emotion, intensity, response, feedback)

        # 3. 记忆（如果启用）
        if learn:
            self.memory_system.record_interaction(
                self.current_user_id, text, primary_emotion,
                response or "", feedback
            )

        # 4. 构建响应
        suggestions = []
        emotion_response = self._generate_emotion_response(primary_emotion, intensity)
        suggestions.append(f"情感响应: {emotion_response}")

        if confidence < 0.5:
            suggestions.append(f"注意: 置信度较低({confidence:.2f})，结果仅供参考")

        # 5. 生成有血有肉回复
        human_response = self._generate_human_response(primary_emotion, intensity, vad, text)

        return {
            "version": self.version,
            "engine": "hybrid",
            "emotion": {
                "primary": primary_emotion,
                "intensity": intensity,
                "vad": vad,
                "is_irony": is_irony,
                "irony_info": None,
                "complex": [],
                "confidence": confidence,
            },
            "personality_safe": True,
            "suggestions": suggestions,
            "human_response": human_response,
            "user_summary": self.memory_system.get_user_summary(self.current_user_id),
        }

    def _learn(self, text: str, emotion: str, intensity: float,
               response: str, feedback: Optional[float]):
        """从交互中学习"""
        # 推断响应情感
        response_emotion = self._infer_response_emotion(response)

        # 记录学习
        self.evolution_manager.process_interaction(
            user_text=text,
            user_emotion=emotion,
            user_intensity=intensity,
            response=response,
            response_emotion=response_emotion,
            feedback=feedback
        )

        # 获取学习到的规则
        rules = self.evolution_manager.evolution.get_rules()
        for rule in rules[-5:]:  # 最近5条
            if rule.get("pattern") and rule.get("emotion"):
                self.memory_system.add_learned_pattern(
                    rule["pattern"], rule["emotion"], rule.get("confidence", 0.5)
                )

        # 反哺：将新学习的模式加载到检测器
        self._refresh_learned_patterns()

    def _refresh_learned_patterns(self):
        """从记忆系统加载学习到的模式到检测器"""
        patterns = self.memory_system.learned_patterns
        if patterns:
            self.emotion_analyzer.load_patterns_from_dict(patterns)

    def _infer_response_emotion(self, response: str) -> str:
        """推断响应情感"""
        response_lower = response.lower()

        if any(w in response_lower for w in ["心疼", "理解", "难过"]):
            return "empathy"
        if any(w in response_lower for w in ["棒", "厉害", "恭喜"]):
            return "celebrate"
        if any(w in response_lower for w in ["加油", "相信"]):
            return "encourage"
        if any(w in response_lower for w in ["别担心", "没事"]):
            return "reassure"
        if any(w in response_lower for w in ["确实", "换我"]):
            return "understand"

        return "neutral"

    def _generate_emotion_response(self, emotion: str, intensity: float) -> str:
        """生成情感响应"""
        responses = {
            "joy": ["太棒了！", "真为你高兴！", "开心！"],
            "sadness": ["我懂你的感受...", "心疼你", "会好起来的"],
            "anger": ["确实很气人...", "可以理解你的感受", "消消气"],
            "fear": ["别担心...", "会没事的", "加油"],
            "surprise": ["哇！真的吗？", "太意外了！", "没想到！"],
            "anticipation": ["期待！", "一定很棒！", "加油！"],
            "trust": ["相信你！", "可以的！", "没问题！"],
            "disgust": ["确实让人不舒服", "理解你的感受", "换谁都会这样"],
            "optimism": ["会好起来的！", "曙光在前！", "加油！"],
            "love": ["真美好！", "爱了！", "好感人！"],
            "guilt": ["知错就好", "没关系", "下次注意"],
            "submission": ["认命吧", "算了", "接受现实"],
            "surprise_complex": ["太震惊了！", "难以置信！", "这...没想到"],
            "disappointment": ["真遗憾...", "可惜了", "哎..."],
            "remorse": ["早知当初...", "后悔没用", "向前看"],
            "envy": ["加油你也可以！", "羡慕啊", "努力！"],
            "suspicion": ["有道理", "确实可疑", "值得怀疑"],
            "aggression": ["消消气", "别冲动", "冷静"],
            "pride": ["骄傲！", "了不起！", "佩服！"],
            "contentment": ["满足", "真好", "惬意"],
            "contempt": ["鄙视", "藐视", "看不起"],
            "cynicism": ["呵呵...", "可笑", "有意思"],
            "morbidness": ["阴暗...", "想太多了", "振作"],
            "sentimentality": ["感慨啊", "怀念", "往事"],
            "anxiety": ["别焦虑", "会好的", "放轻松"],
            "despair": ["别绝望！", "还有希望！", "振作起来！"],
        }

        resp_list = responses.get(emotion, responses.get("neutral", ["了解了"]))
        if intensity > 0.7:
            return resp_list[0] if resp_list else "理解你！"
        elif intensity > 0.4:
            return resp_list[1] if len(resp_list) > 1 else resp_list[0]
        else:
            return resp_list[-1] if len(resp_list) > 1 else resp_list[0]

    def _generate_human_response(
        self,
        emotion: str,
        intensity: float,
        vad: Tuple[float, float, float],
        user_text: str = ""
    ) -> Dict[str, Any]:
        """生成有血有肉的情感化回复"""
        # 1. 共情回复
        empathy_response = self.empathy_engine.generate(emotion, intensity, user_text)

        # 2. 个性化表达
        styled = self.personality_expressor.express(
            base_text=empathy_response.text,
            personality=self.default_personality,
            mood=self._emotion_to_mood(emotion),
            emotion_intensity=intensity,
        )

        # 3. 口语化
        natural = self.nl_generator.generate(
            intent=styled,
            emotion=emotion,
            intensity=intensity,
            personality={
                "warmth": self.default_personality.warmth,
                "humor": self.default_personality.humor,
            },
        )

        # 4. 去AI化
        natural = self._remove_ai_patterns(natural)

        return {
            "text": natural,
            "empathy_type": empathy_response.empathy_type,
            "intensity_level": empathy_response.intensity_level,
            "follow_up": empathy_response.follow_up_suggestion,
            "note": "有血有肉的情感化回复（基于TrueEmotionLife）",
        }

    def _emotion_to_mood(self, emotion: str) -> str:
        """将情感映射到心情状态"""
        mood_map = {
            "joy": "happy",
            "sadness": "sad",
            "anger": "frustrated",
            "fear": "anxious",
            "anxiety": "anxious",
            "surprise": "excited",
            "love": "happy",
            "trust": "calm",
            "anticipation": "excited",
            "optimism": "happy",
            "guilt": "sad",
            "envy": "sad",
            "contempt": "frustrated",
            "despair": "sad",
            "disgust": "frustrated",
        }
        return mood_map.get(emotion, "neutral")

    def _remove_ai_patterns(self, text: str) -> str:
        """移除AI模式"""
        ai_patterns = [
            "此外",
            "然而",
            "综上所述",
            "值得注意的是",
            "从某种意义上说",
            "首先",
            "其次",
            "最后",
            "总之",
        ]
        for pattern in ai_patterns:
            if pattern in text:
                text = text.replace(pattern, "")
        return text.strip()

    def evolve(self) -> Dict[str, Any]:
        """执行进化"""
        if self.evolution_manager:
            return self.evolution_manager.evolve()
        return {"status": "evolution_disabled"}

    def get_user_info(self, user_id: str = None) -> Dict:
        """获取用户信息"""
        return self.memory_system.get_user_summary(user_id or self.current_user_id)

    def get_memory_status(self) -> Dict:
        """获取记忆状态"""
        return {
            "total_users": len(self.memory_system.user_profiles),
            "learned_patterns": len(self.memory_system.learned_patterns),
            "current_user": self.current_user_id,
        }


# ==================== 便捷函数 ====================

_system: Optional[TrueEmotionPro] = None


def get_system() -> TrueEmotionPro:
    """获取系统单例"""
    global _system
    if _system is None:
        _system = TrueEmotionPro()
    return _system


def analyze_pro(text: str, **kwargs) -> Dict[str, Any]:
    """便捷分析函数"""
    return get_system().analyze(text, **kwargs)


# ==================== 测试 ====================

if __name__ == "__main__":
    print("TrueEmotion Pro v3.1.0 - 测试")
    print("=" * 50)

    system = TrueEmotionPro()

    # 测试用例
    test_cases = [
        "工作好累啊，老加班",
        "我升职了！太开心了！",
        "气死了！领导又批评我！",
        "被裁员了，感觉人生没有希望了...",
        "今天吃什么好呢？",
    ]

    for text in test_cases:
        result = system.analyze(text, learn=True, response="我理解你的感受")
        print(f"输入: {text}")
        print(f"  情感: {result['emotion']['primary']} (置信度: {result['emotion']['confidence']:.2f})")
        print(f"  回复: {result['human_response']['text']}")
        print()

    # 测试记忆
    print("\n记忆状态:")
    print(system.get_memory_status())
    print("\n用户信息:")
    print(system.get_user_info())
