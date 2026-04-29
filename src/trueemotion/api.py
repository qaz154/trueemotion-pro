# -*- coding: utf-8 -*-
"""
TrueEmotion Pro - Agent API
===========================

简单的 API 接口，让 Agent 可以调用情感分析功能。

使用示例:
    from trueemotion.api import EmotionAPI

    api = EmotionAPI()
    result = api.analyze("工作好累啊")
    print(result.emotion)      # sadness
    print(result.reply)        # "心疼你..."
"""

import sys
from pathlib import Path

# 确保能导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataclasses import dataclass
from typing import Optional, List
from trueemotion.trueemotion_pro import TrueEmotionPro


@dataclass
class EmotionResult:
    """情感分析结果"""
    emotion: str          # 主要情感
    intensity: float     # 强度 0-1
    confidence: float    # 置信度 0-1
    vad: tuple          # VAD 维度
    reply: str           # 有血有肉的回复
    empathy_type: str    # 共情类型
    user_state: dict    # 用户状态摘要


class EmotionAPI:
    """
    TrueEmotion Pro API for Agents
    封装后的简单接口
    """

    def __init__(self):
        self.pro = TrueEmotionPro()

    def analyze(
        self,
        text: str,
        learn: bool = False,
        response: Optional[str] = None,
        user_id: str = "agent_user"
    ) -> EmotionResult:
        """
        分析文本情感并返回结果

        Args:
            text: 用户输入的文本
            learn: 是否学习这次交互
            response: Agent 的回复
            user_id: 用户标识

        Returns:
            EmotionResult: 包含情感和回复的对象
        """
        result = self.pro.analyze(
            text=text,
            learn=learn,
            response=response,
            user_id=user_id
        )

        emotion_data = result['emotion']
        human_resp = result['human_response']

        return EmotionResult(
            emotion=emotion_data['primary'],
            intensity=emotion_data['intensity'],
            confidence=emotion_data['confidence'],
            vad=emotion_data['vad'],
            reply=human_resp['text'],
            empathy_type=human_resp['empathy_type'],
            user_state=result.get('user_summary', {})
        )

    def get_user_history(self, user_id: str = "agent_user", limit: int = 5) -> List[dict]:
        """获取用户的最近交互历史"""
        return self.pro.memory_system.get_recent_history(user_id, limit)

    def get_user_info(self, user_id: str = "agent_user") -> dict:
        """获取用户信息"""
        return self.pro.get_user_info(user_id)


# ==================== 便捷函数 ====================

_api: Optional[EmotionAPI] = None


def get_api() -> EmotionAPI:
    """获取 API 单例"""
    global _api
    if _api is None:
        _api = EmotionAPI()
    return _api


def analyze(text: str, **kwargs) -> EmotionResult:
    """
    快速分析文本情感

    用法:
        result = analyze("工作好累啊")
        print(result.emotion)  # sadness
        print(result.reply)    # "心疼你..."
    """
    return get_api().analyze(text, **kwargs)


# ==================== CLI 接口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrueEmotion Pro - 情感分析 CLI")
    parser.add_argument("text", help="要分析的文本")
    parser.add_argument("--learn", action="store_true", help="启用学习")
    parser.add_argument("--response", type=str, help="Agent 的回复（用于学习）")

    args = parser.parse_args()

    result = analyze(args.text, learn=args.learn, response=args.response)

    print(f"情感: {result.emotion}")
    print(f"强度: {result.intensity:.2f}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"VAD: {result.vad}")
    print(f"回复: {result.reply}")
    print(f"共情类型: {result.empathy_type}")
