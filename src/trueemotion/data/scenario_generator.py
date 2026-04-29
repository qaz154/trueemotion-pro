# -*- coding: utf-8 -*-
"""
Scenario-Based Training Data Generator
=======================================

基于真实心理场景的情感训练数据生成

不同于简单的关键词组合，我们从真实心理场景出发：
1. 失业打击 -> 悲伤、恐惧、愤怒、焦虑
2. 表白成功 -> 喜悦、期待、紧张
3. 工作压力 -> 焦虑、恐惧、愤怒
...

每个场景包含：
- 场景描述
- 多轮对话模板
- 情感标签（原型 + 复合 + 强度 + VAD）
- 反讽样本（可选）
"""

import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Scenario:
    """场景定义"""
    id: str
    name: str
    description: str
    emotions: List[str]  # 该场景可能的情感
    dialogues: List[Dict]  # 对话模板列表


@dataclass
class DialogueTemplate:
    """对话模板"""
    speaker: str  # "user" or "agent"
    text: str
    emotions: Dict[str, float]  # 原型情感及强度
    complex_emotions: Dict[str, bool]  # 复合情感
    intensity: float  # 情感强度
    vad: Tuple[float, float, float]  # VAD值
    is_irony: bool = False
    irony_surface: Optional[str] = None
    irony_true: Optional[str] = None


class ScenarioBasedDataGenerator:
    """
    基于场景的训练数据生成器

    生成真实、多样、有上下文的情感训练数据
    """

    # ==================== 场景定义 ====================

    SCENARIOS: Dict[str, Scenario] = {
        "job_loss": Scenario(
            id="job_loss",
            name="失业打击",
            description="在公司工作多年后被裁员",
            emotions=["sadness", "fear", "anger", "anxiety", "disappointment"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "我被裁员了...在公司干了5年了。",
                    "emotions": {"sadness": 0.9, "fear": 0.6, "anger": 0.4},
                    "complex_emotions": {"despair": True},
                    "intensity": 0.85,
                    "vad": (-0.7, 0.2, -0.5)
                },
                {
                    "speaker": "user",
                    "text": "太突然了，完全没想到会这样。",
                    "emotions": {"surprise": 0.8, "fear": 0.7},
                    "complex_emotions": {"surprise_complex": True},
                    "intensity": 0.75,
                    "vad": (-0.4, 0.6, -0.3)
                },
                {
                    "speaker": "user",
                    "text": "唉，算了，反正早走晚走都得走。",
                    "emotions": {"sadness": 0.6, "submission": 0.5},
                    "complex_emotions": {"submission": True, "remorse": True},
                    "intensity": 0.55,
                    "vad": (-0.5, -0.2, -0.4)
                },
                {
                    "speaker": "user",
                    "text": "还行吧，重新开始也没什么不好的。",
                    "emotions": {"anticipation": 0.5, "sadness": 0.4},
                    "complex_emotions": {"optimism": True},
                    "intensity": 0.5,
                    "vad": (0.1, 0.1, 0.2),
                    "is_irony": True,
                    "irony_surface": "anticipation",
                    "irony_true": "sadness"
                },
            ]
        ),
        "love_confession": Scenario(
            id="love_confession",
            name="表白场景",
            description="向喜欢的人表白成功或失败",
            emotions=["joy", "anticipation", "anxiety", "fear", "sadness"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "她居然答应了！我整个人都懵了！",
                    "emotions": {"joy": 0.95, "surprise": 0.8, "anticipation": 0.7},
                    "complex_emotions": {"optimism": True, "love": True},
                    "intensity": 0.92,
                    "vad": (0.85, 0.75, 0.7)
                },
                {
                    "speaker": "user",
                    "text": "好紧张啊，心跳得特别快。",
                    "emotions": {"anxiety": 0.7, "anticipation": 0.6},
                    "complex_emotions": {"anxiety": True},
                    "intensity": 0.7,
                    "vad": (0.1, 0.8, 0.1)
                },
                {
                    "speaker": "user",
                    "text": "还行吧，就是表白而已。（其实开心死了）",
                    "emotions": {"joy": 0.8},
                    "complex_emotions": {},
                    "intensity": 0.7,
                    "vad": (0.7, 0.3, 0.5),
                    "is_irony": True,
                    "irony_surface": "neutral",
                    "irony_true": "joy"
                },
                {
                    "speaker": "user",
                    "text": "拒绝了...意料之中吧。",
                    "emotions": {"sadness": 0.8, "disappointment": 0.6},
                    "complex_emotions": {"disappointment": True, "remorse": True},
                    "intensity": 0.75,
                    "vad": (-0.7, -0.1, -0.5)
                },
            ]
        ),
        "work_pressure": Scenario(
            id="work_pressure",
            name="工作压力",
            description="连续加班、项目截止、职场冲突",
            emotions=["anger", "anxiety", "sadness", "fear", "frustration"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "又加班到凌晨三点，我真的快撑不住了。",
                    "emotions": {"sadness": 0.8, "anger": 0.7, "anxiety": 0.6},
                    "complex_emotions": {"despair": True, "aggression": True},
                    "intensity": 0.88,
                    "vad": (-0.75, 0.5, -0.4)
                },
                {
                    "speaker": "user",
                    "text": "领导又改需求了，能不能一次说清楚！",
                    "emotions": {"anger": 0.9, "frustration": 0.7},
                    "complex_emotions": {"aggression": True, "contempt": True},
                    "intensity": 0.82,
                    "vad": (-0.8, 0.7, 0.3)
                },
                {
                    "speaker": "user",
                    "text": "算了，说了也没用，就这样吧。",
                    "emotions": {"sadness": 0.6, "submission": 0.5},
                    "complex_emotions": {"submission": True, "despair": True},
                    "intensity": 0.6,
                    "vad": (-0.6, -0.2, -0.5)
                },
                {
                    "speaker": "user",
                    "text": "挺好的，又学到了新东西。（苦笑）",
                    "emotions": {"sadness": 0.5, "cynicism": 0.4},
                    "complex_emotions": {"cynicism": True},
                    "intensity": 0.5,
                    "vad": (-0.3, 0.1, 0.1),
                    "is_irony": True,
                    "irony_surface": "joy",
                    "irony_true": "sadness"
                },
            ]
        ),
        "friend_conflict": Scenario(
            id="friend_conflict",
            name="朋友冲突",
            description="与朋友发生争执或误解",
            emotions=["anger", "sadness", "anxiety", "disappointment", "remorse"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "我真的没想到她会这样想我。",
                    "emotions": {"surprise": 0.7, "sadness": 0.8},
                    "complex_emotions": {"disappointment": True},
                    "intensity": 0.75,
                    "vad": (-0.5, 0.3, -0.3)
                },
                {
                    "speaker": "user",
                    "text": "我们吵架了，她说了一些很伤人的话。",
                    "emotions": {"anger": 0.8, "sadness": 0.9},
                    "complex_emotions": {"disappointment": True, "contempt": True},
                    "intensity": 0.85,
                    "vad": (-0.75, 0.4, -0.2)
                },
                {
                    "speaker": "user",
                    "text": "可能我也有问题吧...但她太过分了。",
                    "emotions": {"sadness": 0.6, "guilt": 0.4, "anger": 0.5},
                    "complex_emotions": {"guilt": True, "remorse": True},
                    "intensity": 0.6,
                    "vad": (-0.4, 0.1, -0.3)
                },
            ]
        ),
        "academic_pressure": Scenario(
            id="academic_pressure",
            name="学业压力",
            description="考试、升学、论文等学业压力",
            emotions=["anxiety", "fear", "sadness", "anticipation", "anger"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "考研成绩出来了...差了2分。",
                    "emotions": {"sadness": 0.9, "disappointment": 0.8},
                    "complex_emotions": {"disappointment": True, "despair": True},
                    "intensity": 0.85,
                    "vad": (-0.8, -0.1, -0.6)
                },
                {
                    "speaker": "user",
                    "text": "好担心啊，万一毕不了业怎么办。",
                    "emotions": {"anxiety": 0.85, "fear": 0.7},
                    "complex_emotions": {"anxiety": True},
                    "intensity": 0.78,
                    "vad": (-0.5, 0.6, -0.4)
                },
                {
                    "speaker": "user",
                    "text": "无所谓了，大不了延毕呗。",
                    "emotions": {"sadness": 0.6, "submission": 0.5},
                    "complex_emotions": {"submission": True, "despair": True},
                    "intensity": 0.55,
                    "vad": (-0.5, -0.2, -0.5),
                    "is_irony": True,
                    "irony_surface": "neutral",
                    "irony_true": "anxiety"
                },
                {
                    "speaker": "user",
                    "text": "居然过了！太惊喜了！",
                    "emotions": {"joy": 0.9, "surprise": 0.7, "relief": 0.6},
                    "complex_emotions": {"optimism": True, "surprise_complex": True},
                    "intensity": 0.85,
                    "vad": (0.8, 0.7, 0.6)
                },
            ]
        ),
        "family_issue": Scenario(
            id="family_issue",
            name="家庭问题",
            description="与家人的矛盾、沟通问题",
            emotions=["sadness", "anger", "anxiety", "frustration", "guilt"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "我妈又唠叨了，说我这不好那不好。",
                    "emotions": {"anger": 0.6, "sadness": 0.7},
                    "complex_emotions": {"frustration": True},
                    "intensity": 0.65,
                    "vad": (-0.4, 0.3, -0.2)
                },
                {
                    "speaker": "user",
                    "text": "和父母沟通真的好难，他们根本不理解我。",
                    "emotions": {"sadness": 0.8, "frustration": 0.7, "anxiety": 0.5},
                    "complex_emotions": {"disappointment": True, "despair": True},
                    "intensity": 0.75,
                    "vad": (-0.6, 0.2, -0.4)
                },
                {
                    "speaker": "user",
                    "text": "都是为了我好，我知道...但真的很烦。",
                    "emotions": {"sadness": 0.5, "guilt": 0.4, "anger": 0.3},
                    "complex_emotions": {"guilt": True, "remorse": True},
                    "intensity": 0.5,
                    "vad": (-0.3, 0.0, -0.2)
                },
            ]
        ),
        "health_concern": Scenario(
            id="health_concern",
            name="健康担忧",
            description="身体不适、看病、体检",
            emotions=["fear", "anxiety", "sadness", "relief", "anger"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "体检报告有点问题，要进一步检查...好害怕。",
                    "emotions": {"fear": 0.9, "anxiety": 0.85},
                    "complex_emotions": {"anxiety": True, "despair": True},
                    "intensity": 0.88,
                    "vad": (-0.75, 0.7, -0.6)
                },
                {
                    "speaker": "user",
                    "text": "医生说没什么大问题，终于放心了。",
                    "emotions": {"joy": 0.9, "relief": 0.85, "anticipation": 0.3},
                    "complex_emotions": {"optimism": True, "relief": True},
                    "intensity": 0.8,
                    "vad": (0.85, -0.3, 0.6)
                },
                {
                    "speaker": "user",
                    "text": "又感冒了，这都第几次了...身体真差。",
                    "emotions": {"sadness": 0.6, "anger": 0.4},
                    "complex_emotions": {"frustration": True},
                    "intensity": 0.55,
                    "vad": (-0.4, -0.1, -0.3)
                },
            ]
        ),
        "financial_worry": Scenario(
            id="financial_worry",
            name="经济压力",
            description="债务、花呗、理财亏损",
            emotions=["anxiety", "fear", "sadness", "anger", "despair"],
            dialogues=[
                {
                    "speaker": "user",
                    "text": "花呗又还不上了...这个月超支太多了。",
                    "emotions": {"anxiety": 0.8, "fear": 0.6},
                    "complex_emotions": {"anxiety": True, "despair": True},
                    "intensity": 0.75,
                    "vad": (-0.6, 0.5, -0.5)
                },
                {
                    "speaker": "user",
                    "text": "股票又跌了，亏了好几万...真的要崩溃了。",
                    "emotions": {"sadness": 0.8, "anger": 0.6, "despair": 0.7},
                    "complex_emotions": {"despair": True, "aggression": True},
                    "intensity": 0.85,
                    "vad": (-0.8, 0.4, -0.6)
                },
                {
                    "speaker": "user",
                    "text": "没事，大不了分期还，总会有办法的。",
                    "emotions": {"anticipation": 0.5, "sadness": 0.4},
                    "complex_emotions": {"optimism": True},
                    "intensity": 0.45,
                    "vad": (0.1, 0.1, 0.2),
                    "is_irony": True,
                    "irony_surface": "optimism",
                    "irony_true": "anxiety"
                },
            ]
        ),
    }

    # ==================== 数据增强模板 ====================

    AUGMENTATION_TEMPLATES = {
        "intensify": [
            "真的{text}",
            "特别{text}",
            "非常{text}",
            "简直{text}",
        ],
        "weaken": [
            "有点{text}",
            "稍微{text}",
            "略微{text}",
            "好像有点{text}",
        ],
        "question": [
            "{text}吗？",
            "{text}怎么办？",
            "为什么会{text}？",
        ],
        "exclamation": [
            "{text}啊！",
            "太{text}了！",
            "真的{text}！",
        ],
        "add_context": [
            "其实{text}",
            "说真的{text}",
            "说实话{text}",
            "跟你说{text}",
        ],
    }

    def __init__(self, seed: int = 2026):
        random.seed(seed)
        self.scenarios = self.SCENARIOS

    def generate_samples(self, count: int = 1000) -> List[Dict]:
        """
        生成训练样本

        Args:
            count: 目标样本数量

        Returns:
            List[Dict]: 训练样本列表
        """
        samples = []

        # 计算每个场景需要生成多少样本
        samples_per_scenario = count // len(self.scenarios)

        for scenario in self.scenarios.values():
            for _ in range(samples_per_scenario):
                # 随机选择一个对话
                dialogue = random.choice(scenario.dialogues)

                # 应用数据增强
                augmented = self._augment_dialogue(dialogue)

                sample = {
                    "text": augmented["text"],
                    "scenario_id": scenario.id,
                    "scenario_name": scenario.name,
                    "speaker": dialogue["speaker"],
                    "primary_labels": dialogue["emotions"],
                    "complex_labels": dialogue.get("complex_emotions", {}),
                    "intensity_label": dialogue["intensity"],
                    "vad_labels": {
                        "valence": dialogue["vad"][0],
                        "arousal": dialogue["vad"][1],
                        "dominance": dialogue["vad"][2]
                    },
                    "is_irony": dialogue.get("is_irony", False),
                    "surface_emotion": dialogue.get("irony_surface"),
                    "true_emotion": dialogue.get("irony_true"),
                }

                samples.append(sample)

        # 打乱顺序
        random.shuffle(samples)

        return samples

    def _augment_dialogue(self, dialogue: Dict) -> Dict:
        """对对话进行数据增强"""
        original_text = dialogue["text"]
        augmented = dict(dialogue)

        # 30%概率应用增强
        if random.random() < 0.3:
            template_type = random.choice(list(self.AUGMENTATION_TEMPLATES.keys()))
            templates = self.AUGMENTATION_TEMPLATES[template_type]
            template = random.choice(templates)

            # 提取核心情感词
            emotions = dialogue["emotions"]
            primary = max(emotions.items(), key=lambda x: x[1])[0]

            # 用原始情感替换
            new_text = template.replace("{text}", original_text)

            # 如果文本变化不大，保持原样
            if len(new_text) - len(original_text) < 5:
                augmented["text"] = original_text
            else:
                augmented["text"] = new_text
                # 增加情感强度
                if template_type in ["intensify", "exclamation"]:
                    augmented["intensity_label"] = min(1.0, dialogue["intensity"] * 1.2)
                elif template_type == "weaken":
                    augmented["intensity_label"] = dialogue["intensity"] * 0.8

        return augmented

    def generate_with_context(self, scenario_id: str, length: int = 5) -> List[Dict]:
        """
        生成带上下文的对话序列

        Args:
            scenario_id: 场景ID
            length: 对话轮数

        Returns:
            List[Dict]: 对话序列，每个元素包含历史和当前文本
        """
        if scenario_id not in self.scenarios:
            return []

        scenario = self.scenarios[scenario_id]
        dialogues = scenario.dialogues

        samples = []
        for i in range(len(dialogues)):
            # 取前i个作为历史
            history = dialogues[:i] if i > 0 else []
            current = dialogues[i]

            # 构建上下文
            context_texts = [d["text"] for d in history]

            sample = {
                "text": current["text"],
                "context": context_texts,
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "history_length": len(history),
                "primary_labels": current["emotions"],
                "complex_labels": current.get("complex_emotions", {}),
                "intensity_label": current["intensity"],
                "vad_labels": {
                    "valence": current["vad"][0],
                    "arousal": current["vad"][1],
                    "dominance": current["vad"][2]
                },
                "is_irony": current.get("is_irony", False),
            }

            samples.append(sample)

        return samples

    def generate_irony_samples(self, count: int = 200) -> List[Dict]:
        """生成反讽样本"""
        samples = []

        # 反讽场景
        irony_templates = [
            # 克制型
            {"text": "还行吧，就那样。", "emotions": {"disgust": 0.7}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "不错哦，一般般啦。", "emotions": {"disgust": 0.6}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "挺好的，正常发挥。", "emotions": {"disgust": 0.5}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "嗯，还行吧。", "emotions": {"neutral": 0.5}, "is_irony": True, "true_emotion": "sadness"},

            # 愤怒型
            {"text": "真是太感谢了，让我等了三个小时。", "emotions": {"anger": 0.9}, "is_irony": True, "true_emotion": "anger"},
            {"text": "你可真行，又把我的方案否决了。", "emotions": {"anger": 0.85}, "is_irony": True, "true_emotion": "anger"},
            {"text": "行行行，你说的都对。", "emotions": {"anger": 0.7}, "is_irony": True, "true_emotion": "anger"},
            {"text": "你可真是好样的。", "emotions": {"contempt": 0.8}, "is_irony": True, "true_emotion": "contempt"},

            # 悲伤型
            {"text": "太好了，又迟到了。", "emotions": {"sadness": 0.8}, "is_irony": True, "true_emotion": "sadness"},
            {"text": "没事没事，习惯了。", "emotions": {"sadness": 0.7}, "is_irony": True, "true_emotion": "sadness"},
            {"text": "反正也没人在意。", "emotions": {"sadness": 0.8}, "is_irony": True, "true_emotion": "sadness"},
            {"text": "没什么，习惯了。", "emotions": {"sadness": 0.6}, "is_irony": True, "true_emotion": "sadness"},

            # 讽刺型
            {"text": "呵呵，果然不出所料。", "emotions": {"disgust": 0.7}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "笑死，这剧本。", "emotions": {"disgust": 0.6}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "可真是人才啊。", "emotions": {"contempt": 0.8}, "is_irony": True, "true_emotion": "contempt"},

            # 不屑型
            {"text": "好怕怕哦，吓死人了呢。", "emotions": {"disgust": 0.7}, "is_irony": True, "true_emotion": "disgust"},
            {"text": "哇，好厉害啊。", "emotions": {"disgust": 0.6}, "is_irony": True, "true_emotion": "disgust"},
        ]

        # 扩充样本
        for _ in range(count):
            template = random.choice(irony_templates)
            sample = {
                "text": template["text"],
                "scenario_id": "irony",
                "primary_labels": template["emotions"],
                "complex_labels": {},
                "intensity_label": 0.65,
                "vad_labels": {"valence": -0.4, "arousal": 0.2, "dominance": 0.1},
                "is_irony": True,
                "true_emotion": template["true_emotion"],
            }
            samples.append(sample)

        return samples

    def get_train_dev_test_split(
        self,
        total_count: int = 5000,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        获取训练/验证/测试分割

        按场景分割，避免数据泄露
        """
        all_samples = self.generate_samples(total_count)

        # 按场景分组
        from collections import defaultdict
        by_scenario = defaultdict(list)
        for sample in all_samples:
            by_scenario[sample["scenario_id"]].append(sample)

        train, dev, test = [], [], []

        for scenario_id, samples in by_scenario.items():
            n = len(samples)
            n_train = int(n * train_ratio)
            n_dev = int(n * dev_ratio)

            train.extend(samples[:n_train])
            dev.extend(samples[n_train:n_train + n_dev])
            test.extend(samples[n_train + n_dev:])

        return train, dev, test

    def get_scenario_test_set(self, unseen_scenarios: List[str]) -> List[Dict]:
        """
        获取场景外测试集

        用于测试模型在新场景上的泛化能力
        """
        samples = []

        for scenario_id in unseen_scenarios:
            if scenario_id in self.scenarios:
                scenario = self.scenarios[scenario_id]
                for dialogue in scenario.dialogues:
                    sample = {
                        "text": dialogue["text"],
                        "scenario_id": scenario.id,
                        "scenario_name": scenario.name,
                        "primary_labels": dialogue["emotions"],
                        "complex_labels": dialogue.get("complex_emotions", {}),
                        "intensity_label": dialogue["intensity"],
                        "vad_labels": {
                            "valence": dialogue["vad"][0],
                            "arousal": dialogue["vad"][1],
                            "dominance": dialogue["vad"][2]
                        },
                        "is_irony": dialogue.get("is_irony", False),
                    }
                    samples.append(sample)

        return samples


if __name__ == "__main__":
    # 测试数据生成器
    generator = ScenarioBasedDataGenerator()

    print("=" * 70)
    print("场景驱动的情感训练数据生成测试")
    print("=" * 70)

    # 生成样本
    samples = generator.generate_samples(count=100)
    print(f"\n生成了 {len(samples)} 个训练样本")

    # 按场景统计
    from collections import Counter
    scenario_counts = Counter(s["scenario_id"] for s in samples)
    print("\n按场景分布:")
    for scenario_id, count in scenario_counts.items():
        scenario_name = generator.scenarios[scenario_id].name
        print(f"  {scenario_name}: {count}")

    # 显示一些样本
    print("\n样本示例:")
    for i, sample in enumerate(samples[:5]):
        print(f"\n{i+1}. [{sample['scenario_name']}] {sample['text']}")
        print(f"   情感: {sample['primary_labels']}")
        print(f"   复合: {sample['complex_labels']}")
        print(f"   强度: {sample['intensity_label']:.2f}")
        if sample.get('is_irony'):
            print(f"   反讽: 真实情感={sample.get('true_emotion')}")

    # 生成反讽样本
    print("\n" + "=" * 70)
    print("反讽样本测试")
    print("=" * 70)

    irony_samples = generator.generate_irony_samples(count=20)
    print(f"\n生成了 {len(irony_samples)} 个反讽样本")

    for sample in irony_samples[:5]:
        print(f"\n  {sample['text']}")
        print(f"    真实情感: {sample.get('true_emotion')}")

    # 测试场景外泛化
    print("\n" + "=" * 70)
    print("场景外泛化测试集")
    print("=" * 70)

    unseen = ["health_concern", "financial_worry"]
    test_set = generator.get_scenario_test_set(unseen)
    print(f"\n未见场景 {unseen} 的测试样本数: {len(test_set)}")

    for sample in test_set[:3]:
        print(f"\n  [{sample['scenario_name']}] {sample['text']}")
        print(f"    情感: {sample['primary_labels']}")
