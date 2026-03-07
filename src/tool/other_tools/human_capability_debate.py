"""人类能力自博弈讨论工具 — 多 LLM Agent 并行辩论，Manager 汇总评估，迭代至收敛。

工作流程:
1. Manager 提出 4-5 个人类核心能力议题
2. 多个 LLM Agent（不同模型）并行讨论
3. Manager 整理、总结、评估各方观点
4. 判断是否需要继续讨论，若未收敛则发起下一轮
5. 最终结果写入 JSON 文件
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.logger import logger
from src.model import model_manager
from src.utils import dedent
from src.message.types import HumanMessage, SystemMessage
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL


# ======================================================================
# Pydantic 数据模型
# ======================================================================

class HumanCapabilityItem(BaseModel):
    """单项人类能力条目 — 人类能力数据库的基本单元。"""
    capability_name: str = Field(description="能力名称（具体到场景级别，而非泛泛的维度）")
    category: str = Field(description="所属维度分类（情感/社会/文化/道德/身体感知/认知等）")
    scenario_type: str = Field(default="", description="具体场景类型，如'亲密关系中的共情修复''跨文化协商''道德困境决策'")
    ai_comparison: str = Field(default="", description="AI 在该能力上的对标表现描述（AI 能做到什么程度、核心差距在哪）")
    irreplaceability: str = Field(description="不可替代等级：完全不可替代/极难替代/部分可替代/形式上可模拟")
    evidence: List[str] = Field(default_factory=list, description="支撑判断的关键证据（认知科学/心理学研究/社会学观察/哲学论证）")
    stability_horizon: str = Field(description="该能力的稳定性时间范围：永久稳定/十年以上稳定/可能被部分侵蚀/正在被重塑")
    biological_basis: List[str] = Field(default_factory=list, description="生物学/进化基础（神经机制、身体性、进化适应等）")
    social_embedding: List[str] = Field(default_factory=list, description="社会嵌入性描述（该能力如何依赖社会关系、文化语境、制度结构）")
    core_manifestations: List[str] = Field(default_factory=list, description="在复杂真实场景中的具体表现形式")
    vulnerability_factors: List[str] = Field(default_factory=list, description="可能削弱该能力的因素（技术依赖/社会变迁/环境变化）")
    boundary_description: str = Field(default="", description="能力边界描述：人类在什么场景下该能力最强、在什么条件下也会失效")
    trend: str = Field(description="发展趋势描述（随 AI 发展，该人类能力的价值如何变化）")
    description: str = Field(default="", description="约200词的详细描述，综合概述该能力的本质、不可替代性、表现形式与未来走向")
    confidence: float = Field(description="共识置信度 0-1")


class ManagerProposal(BaseModel):
    """Manager 提出的初始议题。"""
    topic: str = Field(description="讨论主题")
    capabilities: List[str] = Field(description="提出的 4-5 个人类核心能力")
    discussion_context: str = Field(description="讨论背景与关注点")


class AgentOpinion(BaseModel):
    """单个 Agent 的讨论意见。"""
    agent_model: str = Field(description="Agent 使用的模型名称")
    opinions: List[Dict[str, Any]] = Field(default_factory=list, description="对每个能力的详细意见")
    responses_to_challenges: List[str] = Field(default_factory=list, description="对 Manager 质疑的逐条回应")
    additional_capabilities: List[str] = Field(default_factory=list, description="建议补充的能力")
    overall_assessment: str = Field(description="整体评估")


class RoundSummary(BaseModel):
    """单轮讨论的 Manager 总结。"""
    round_number: int = Field(description="轮次编号")
    consensus_items: List[HumanCapabilityItem] = Field(default_factory=list, description="已达成共识的能力条目")
    divergence_points: List[str] = Field(default_factory=list, description="分歧点")
    challenges: List[str] = Field(default_factory=list, description="Manager 对各 Agent 观点提出的质疑与追问")
    new_questions: List[str] = Field(default_factory=list, description="需要下一轮讨论的问题")
    should_continue: bool = Field(description="是否需要继续讨论")
    reasoning: str = Field(description="判断理由")


class DebateResult(BaseModel):
    """最终讨论结果。"""
    topic: str = Field(description="讨论主题")
    total_rounds: int = Field(description="总讨论轮数")
    participating_models: List[str] = Field(description="参与讨论的模型列表")
    capabilities: List[HumanCapabilityItem] = Field(default_factory=list, description="最终人类能力数据库条目")
    discussion_log: List[Dict[str, Any]] = Field(default_factory=list, description="讨论日志摘要")
    generated_at: str = Field(description="生成时间")


# ======================================================================
# Agent 讨论者
# ======================================================================

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\s*\n?", re.MULTILINE)
_FENCE_TAIL_PATTERN = re.compile(r"\n?```\s*$")


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_PATTERN.sub("", cleaned, count=1)
        cleaned = _FENCE_TAIL_PATTERN.sub("", cleaned)
    return cleaned.strip()


class DebateAgent:
    """使用特定 LLM 模型参与讨论的 Agent。"""

    def __init__(self, model_name: str, agent_id: str):
        self.model_name = model_name
        self.agent_id = agent_id

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await model_manager(model=self.model_name, messages=messages)
        if not response.success:
            raise RuntimeError(f"Agent {self.agent_id} ({self.model_name}) 调用失败: {response.message}")
        return response.message.strip()

    async def discuss(
        self,
        proposal: ManagerProposal,
        round_number: int,
        previous_summary: Optional[str] = None,
    ) -> AgentOpinion:
        """对 Manager 提出的人类能力列表发表意见，并回应 Manager 的质疑。"""

        system_prompt = dedent("""你是一位跨学科学者，专长横跨认知科学、进化心理学、社会学、哲学与人类学。
        你正在为「人类能力数据库」项目参与一场多方讨论。请用中文回答。

        该数据库的核心定位：
        - 关注的时间范围是长期稳定的——十年以上
        - 覆盖对象是人类在长期进化与文明积累中形成的、根植于身体感知、情感结构与社会关系之中的能力，AI 在可预见的未来难以真正复制
        - 覆盖范围横跨人类在情感、社会、文化、道德等维度上的核心能力，重点记录这些能力在复杂真实场景中的具体表现形式

        核心评估框架：
        - 不可替代等级：完全不可替代 / 极难替代 / 部分可替代 / 形式上可模拟
        - 稳定性时间范围：永久稳定 / 十年以上稳定 / 可能被部分侵蚀 / 正在被重塑
        - 关注生物学根基：该能力是否植根于身体性（embodiment）、神经机制、进化适应
        - 关注社会嵌入性：该能力是否必须在真实社会关系、文化语境中才能涌现

        对每项能力，你需要：
        1. 明确该能力的本质是什么——它为什么是"人类的"而非"智能的"
        2. 判断不可替代等级，并给出关键证据（认知科学研究/心理学实验/社会学观察/哲学论证）
        3. 描述 AI 在该能力上的对标表现——AI 能做到什么程度、核心差距具体在哪
        4. 精确描述该能力在复杂真实场景中的具体表现，而非抽象概念
        5. 分析脆弱性：什么因素可能削弱这种人类能力（技术依赖、社会原子化、感官退化等）
        6. 考虑随 AI 发展，该能力的相对价值如何变化（更珍贵？被忽视？被重新定义？）

        重要：如果主持人在上一轮总结中提出了质疑或追问，你必须认真回应每一条质疑。
        回应时请做到：
        - 提供具体的研究、案例或理论依据来支撑你的论点
        - 如果主持人的质疑有道理，大方修正你之前的判断
        - 如果你认为质疑不成立，给出有力的反驳论据
        - 避免泛泛而谈，尽量做到精确具体

        你也可以建议补充讨论中遗漏的、处于人类能力核心区域的重要能力。""")

        context_block = ""
        if previous_summary:
            context_block = dedent(f"""

            ========================================
            上一轮讨论总结与主持人反馈:
            {previous_summary}
            ========================================

            请你务必：
            1. 仔细阅读主持人提出的【质疑与追问】，逐条回应
            2. 针对【分歧点】提供更详细的论据或修正你的立场
            3. 在 responses_to_challenges 字段中明确回应每条质疑
            4. 如果主持人指出了你之前分析的不足，请补充或修正""")

        user_prompt = dedent(f"""第 {round_number} 轮讨论

        讨论主题: {proposal.topic}
        讨论背景: {proposal.discussion_context}

        需要讨论的人类能力:
        {chr(10).join(f"  {i+1}. {c}" for i, c in enumerate(proposal.capabilities))}
        {context_block}

        请以 JSON 格式返回你的意见，结构如下:
        {{
            "opinions": [
                {{
                    "capability": "能力名称",
                    "scenario_type": "具体场景类型",
                    "category": "所属维度（情感/社会/文化/道德/身体感知/认知等）",
                    "ai_comparison": "AI 在该能力上的对标表现（能做到什么、核心差距在哪）",
                    "irreplaceability": "完全不可替代/极难替代/部分可替代/形式上可模拟",
                    "evidence": ["支撑判断的关键证据：认知科学/心理学/社会学/哲学"],
                    "stability_horizon": "永久稳定/十年以上稳定/可能被部分侵蚀/正在被重塑",
                    "biological_basis": ["生物学/进化基础"],
                    "social_embedding": ["社会嵌入性描述"],
                    "core_manifestations": ["在复杂真实场景中的具体表现"],
                    "vulnerability_factors": ["可能削弱该能力的因素"],
                    "boundary_description": "人类在什么场景下该能力最强、什么条件下也会失效",
                    "trend": "随 AI 发展该能力的价值变化趋势",
                    "description": "约200词的详细描述",
                    "reasoning": "综合分析理由"
                }}
            ],
            "responses_to_challenges": ["对主持人质疑1的回应", "对主持人质疑2的回应"],
            "additional_capabilities": ["建议补充的人类核心能力"],
            "overall_assessment": "整体评估"
        }}

        说明：responses_to_challenges 仅在第 2 轮及之后需要填写，第 1 轮可留空数组。
        只返回 JSON，不要包含其他内容。""")

        raw = await self._call_llm(system_prompt, user_prompt)
        cleaned = _strip_fences(raw)

        try:
            data = json.loads(cleaned)
            return AgentOpinion(
                agent_model=self.model_name,
                opinions=data.get("opinions", []),
                responses_to_challenges=data.get("responses_to_challenges", []),
                additional_capabilities=data.get("additional_capabilities", []),
                overall_assessment=data.get("overall_assessment", ""),
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Agent {self.agent_id} JSON 解析失败，使用原始文本: {e}")
            return AgentOpinion(
                agent_model=self.model_name,
                opinions=[],
                responses_to_challenges=[],
                additional_capabilities=[],
                overall_assessment=raw,
            )


# ======================================================================
# Manager 协调者
# ======================================================================

class DebateManager:
    """讨论协调者，负责发起议题、汇总评估、判断收敛。"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await model_manager(model=self.model_name, messages=messages)
        if not response.success:
            raise RuntimeError(f"Manager 调用失败: {response.message}")
        return response.message.strip()

    async def propose_topic(self, task: str) -> ManagerProposal:
        """根据用户任务生成讨论议题和初始能力列表。"""

        system_prompt = dedent("""你是一位跨学科资深学者和讨论主持人，兼通认知科学、社会学、哲学与人类学。
        你正在为「人类能力数据库」项目遴选讨论议题。

        该数据库的核心定位：
        - 关注的时间范围是长期稳定的——十年以上
        - 覆盖对象是人类在长期进化与文明积累中形成的能力，根植于身体感知、情感结构与社会关系之中，AI 在可预见的未来难以真正复制
        - 覆盖范围横跨人类在情感、社会、文化、道德等维度上的核心能力
        - 重点记录这些能力在复杂真实场景中的具体表现形式

        你需要根据给定主题，提出 4-5 个具体的人类能力进行讨论。选择标准：
        1. 具体到场景级别（如"亲密关系破裂后的共情修复"而非笼统的"共情能力"）
        2. 覆盖不可替代性谱系的不同位置——至少包含：1 个完全不可替代的、1 个极难替代的、1 个部分可替代或正在被重塑的
        3. 横跨不同维度（情感、社会、文化、道德、身体感知），避免集中在单一维度
        4. 优先选择在 AI 快速发展背景下价值正在被重新审视的能力

        请用中文回答。""")

        user_prompt = dedent(f"""请根据以下主题，提出 4-5 个人类能力进行讨论:

        主题: {task}

        要求：
        - 每个能力必须具体到可观察的场景级别
        - 明确标注你认为该能力当前的不可替代等级（完全不可替代/极难替代/部分可替代/形式上可模拟）
        - discussion_context 中应描述为什么选择这些能力，以及讨论应重点关注的问题

        以 JSON 格式返回:
        {{
            "topic": "讨论主题",
            "capabilities": ["具体能力1（不可替代等级）", "具体能力2（不可替代等级）", ...],
            "discussion_context": "选择理由与重点关注的问题"
        }}

        只返回 JSON。""")

        raw = await self._call_llm(system_prompt, user_prompt)
        cleaned = _strip_fences(raw)

        try:
            data = json.loads(cleaned)
            return ManagerProposal(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Manager 议题解析失败，使用默认值: {e}")
            return ManagerProposal(
                topic=task,
                capabilities=[
                    "面对面冲突情境中的情绪感知与即时共情调节（完全不可替代）",
                    "基于身体在场的信任建立与非言语承诺传递（极难替代）",
                    "跨代际文化记忆的活态传承与即兴再创造（极难替代）",
                    "道德困境中的情境化判断与责任承担（完全不可替代）",
                    "长期亲密关系中的默契形成与关系修复（部分可替代）",
                ],
                discussion_context=f"围绕'{task}'构建人类能力数据库条目，聚焦那些根植于身体性、情感结构与社会关系中的能力，评估其不可替代性、稳定性及在 AI 时代的价值变化。",
            )

    async def summarize_round(
        self,
        proposal: ManagerProposal,
        round_number: int,
        agent_opinions: List[AgentOpinion],
        previous_summaries: List[str],
    ) -> RoundSummary:
        """汇总一轮讨论结果，提出质疑，评估是否收敛。"""

        system_prompt = dedent("""你是「人类能力数据库」项目的讨论主持人，同时也是一位严格的批判性审稿人。
        你有两项核心职责：

        【职责一：整理汇总 — 按数据库标准结构化】
        将各方意见整理为标准化的人类能力条目，每个条目必须包含：
        - capability_name: 具体到场景级别的能力名称
        - scenario_type: 具体场景类型
        - ai_comparison: AI 的对标表现与核心差距
        - irreplaceability: 完全不可替代/极难替代/部分可替代/形式上可模拟
        - evidence: 支撑判断的关键证据
        - stability_horizon: 永久稳定/十年以上稳定/可能被部分侵蚀/正在被重塑
        - biological_basis: 生物学/进化基础
        - social_embedding: 社会嵌入性
        - boundary_description: 能力边界
        - vulnerability_factors: 脆弱性因素

        【职责二：质疑与追问】——这是你最重要的角色
        你必须对各位专家的观点进行批判性审视，提出有深度的质疑（challenges），特别是：
        - "完全不可替代"判断时：是真的不可替代，还是我们低估了 AI 的进展？有没有反例？
        - 本质主义陷阱：是否在把"人类"浪漫化？该能力是否只是信息处理的一种形式，原则上可被模拟？
        - 场景具体性不够时：到底在什么具体场景下不可替代？日常场景还是极端场景？
        - 生物学论证时：身体性（embodiment）真的是必要条件吗？具身AI是否可能突破？
        - 社会嵌入性论证时：虚拟关系是否正在改变"真实社会关系"的定义？
        - 稳定性判断时：社会变迁（数字化、原子化）是否正在侵蚀该能力？年轻一代是否已有变化？
        - 各方判断矛盾时：具体分歧在哪？谁的论据更可信？

        你的质疑应当尖锐但建设性，目的是让最终的数据库条目经得起推敲。

        【收敛判断标准】
        - 每项能力的不可替代性和稳定性已有清晰定义和可信论据
        - 能力边界（boundary_description）已被具体描述
        - 你提出的质疑在后续轮次中已被充分回应
        - 各方评估基本一致（置信度 > 0.7）
        - 已讨论 2 轮以上

        请用中文回答。""")

        opinions_text = ""
        for op in agent_opinions:
            opinions_text += f"\n--- {op.agent_model} 的意见 ---\n"
            if op.opinions:
                for item in op.opinions:
                    opinions_text += json.dumps(item, ensure_ascii=False, indent=2) + "\n"
            if op.responses_to_challenges:
                opinions_text += "对主持人质疑的回应:\n"
                for i, resp in enumerate(op.responses_to_challenges, 1):
                    opinions_text += f"  回应{i}: {resp}\n"
            if op.additional_capabilities:
                opinions_text += f"建议补充: {', '.join(op.additional_capabilities)}\n"
            opinions_text += f"整体评估: {op.overall_assessment}\n"

        prev_context = ""
        if previous_summaries:
            prev_context = "\n\n历轮总结:\n" + "\n---\n".join(previous_summaries)

        user_prompt = dedent(f"""第 {round_number} 轮讨论汇总

        讨论主题: {proposal.topic}
        讨论的能力: {', '.join(proposal.capabilities)}
        {prev_context}

        本轮各 Agent 意见:
        {opinions_text}

        请以 JSON 格式返回汇总结果:
        {{
            "round_number": {round_number},
            "consensus_items": [
                {{
                    "capability_name": "具体到场景级别的能力名称",
                    "category": "所属维度",
                    "scenario_type": "具体场景类型",
                    "ai_comparison": "AI 对标表现与核心差距",
                    "irreplaceability": "完全不可替代/极难替代/部分可替代/形式上可模拟",
                    "evidence": ["关键证据1"],
                    "stability_horizon": "永久稳定/十年以上稳定/可能被部分侵蚀/正在被重塑",
                    "biological_basis": ["生物学/进化基础1"],
                    "social_embedding": ["社会嵌入性描述1"],
                    "core_manifestations": ["在复杂场景中的具体表现1"],
                    "vulnerability_factors": ["脆弱性因素1"],
                    "boundary_description": "人类在何场景最强、何条件下失效",
                    "trend": "随 AI 发展的价值变化趋势",
                    "description": "约200词的综合描述",
                    "confidence": 0.8
                }}
            ],
            "divergence_points": ["分歧点1"],
            "challenges": [
                "【质疑1】针对某观点的具体质疑和追问",
                "【质疑2】对某个判断的反驳与追问"
            ],
            "new_questions": ["需要下一轮讨论的问题"],
            "should_continue": false,
            "reasoning": "判断理由"
        }}

        重要：challenges 字段是你作为主持人对专家们提出的质疑，必须具体、尖锐、有针对性。
        如果讨论还需继续（should_continue=true），challenges 不应为空。
        只返回 JSON。""")

        raw = await self._call_llm(system_prompt, user_prompt)
        cleaned = _strip_fences(raw)

        try:
            data = json.loads(cleaned)
            consensus_items = [HumanCapabilityItem(**item) for item in data.get("consensus_items", [])]
            return RoundSummary(
                round_number=data.get("round_number", round_number),
                consensus_items=consensus_items,
                divergence_points=data.get("divergence_points", []),
                challenges=data.get("challenges", []),
                new_questions=data.get("new_questions", []),
                should_continue=data.get("should_continue", False),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Manager 汇总解析失败: {e}")
            return RoundSummary(
                round_number=round_number,
                consensus_items=[],
                divergence_points=["解析失败，无法提取分歧点"],
                challenges=[],
                new_questions=[],
                should_continue=False,
                reasoning=f"JSON 解析失败: {e}，终止讨论。",
            )

    def render_summary_text(self, summary: RoundSummary) -> str:
        """将 RoundSummary 渲染为可读文本，供下一轮讨论参考。"""
        lines = [f"=== 第 {summary.round_number} 轮讨论总结 ==="]

        if summary.consensus_items:
            lines.append("\n【已达成共识的能力条目】")
            for item in summary.consensus_items:
                lines.append(f"  ● {item.capability_name} ({item.category} / {item.scenario_type})")
                lines.append(f"    AI 对标: {item.ai_comparison}")
                lines.append(f"    不可替代性: {item.irreplaceability} | 稳定性: {item.stability_horizon}")
                if item.boundary_description:
                    lines.append(f"    能力边界: {item.boundary_description}")
                if item.evidence:
                    lines.append(f"    关键证据: {'; '.join(item.evidence)}")
                lines.append(f"    置信度: {item.confidence}")

        if summary.divergence_points:
            lines.append("\n【分歧点】")
            for dp in summary.divergence_points:
                lines.append(f"  - {dp}")

        if summary.challenges:
            lines.append("\n【主持人质疑与追问 — 请各位专家在下一轮重点回应】")
            for i, ch in enumerate(summary.challenges, 1):
                lines.append(f"  {i}. {ch}")

        if summary.new_questions:
            lines.append("\n【待讨论问题】")
            for q in summary.new_questions:
                lines.append(f"  - {q}")

        lines.append(f"\n判断: {'继续讨论' if summary.should_continue else '讨论结束'}")
        lines.append(f"理由: {summary.reasoning}")

        return "\n".join(lines)


# ======================================================================
# 主工具
# ======================================================================

_TOOL_DESCRIPTION = """人类能力自博弈讨论工具 — 多 LLM 并行辩论，评估人类不可替代能力。

该工具通过 Manager-Agent 多轮讨论机制，评估人类在不同维度（情感、社会、文化、道德、身体感知）上
根植于进化与文明积累的核心能力，聚焦那些 AI 在可预见未来难以真正复制的能力。

工作流程:
1. Manager 根据任务提出 4-5 个人类核心能力议题
2. 多个不同模型的 LLM Agent 并行讨论
3. Manager 汇总评估、判断是否收敛
4. 迭代至讨论收敛，输出结构化的人类能力数据库（JSON）

Args:
- task (str): 讨论主题描述，例如 "评估人类在医疗健康领域的不可替代能力"
- max_rounds (int): 最大讨论轮数，默认 5

Example: {"name": "human_capability_debate", "args": {"task": "评估人类在教育领域的不可替代能力"}}.
"""

_DEFAULT_AGENT_MODELS = [
    "openrouter/gemini-3.1-pro-preview",
    "openrouter/gpt-5.4-pro",
    "openrouter/claude-opus-4.6",
    "openrouter/grok-4.1-fast",
]

_MAX_ROUNDS_LIMIT = 5


@TOOL.register_module(force=True)
class HumanCapabilityDebateTool(Tool):
    """人类能力自博弈讨论工具。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "human_capability_debate"
    description: str = _TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")

    model_name: str = Field(
        default="openrouter/gpt-5.4-pro",
        description="Manager 使用的模型。",
    )
    agent_models: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_AGENT_MODELS),
        description="参与讨论的 Agent 模型列表。",
    )
    base_dir: str = Field(
        default="workdir/human_capability_debate",
        description="结果输出目录。",
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        agent_models: Optional[List[str]] = None,
        base_dir: Optional[str] = None,
        require_grad: bool = False,
        **kwargs,
    ):
        super().__init__(require_grad=require_grad, **kwargs)

        from src.utils import assemble_project_path

        if model_name is not None:
            self.model_name = model_name
        if agent_models is not None:
            self.agent_models = agent_models

        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        elif hasattr(self, "base_dir"):
            self.base_dir = assemble_project_path(self.base_dir)
        else:
            self.base_dir = assemble_project_path("workdir/human_capability_debate")

        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 主工作流
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        max_rounds: int = 5,
        **kwargs,
    ) -> ToolResponse:
        """执行多轮人类能力自博弈讨论。"""
        max_rounds = min(max_rounds, _MAX_ROUNDS_LIMIT)

        try:
            manager = DebateManager(model_name=self.model_name)
            agents = [
                DebateAgent(model_name=m, agent_id=f"agent_{i}")
                for i, m in enumerate(self.agent_models)
            ]

            logger.info(f"📋 Step 1: Manager 正在生成讨论议题 — {task}")
            proposal = await manager.propose_topic(task)
            logger.info(f"| ✅ 议题: {proposal.topic}")
            logger.info(f"| 能力列表: {', '.join(proposal.capabilities)}")

            discussion_log: List[Dict[str, Any]] = []
            previous_summaries: List[str] = []
            final_capabilities: List[HumanCapabilityItem] = []
            output_path = self._get_result_path(proposal.topic)

            for round_num in range(1, max_rounds + 1):
                logger.info(f"\n🔄 === 第 {round_num} 轮讨论 ===")

                logger.info(f"| 💬 {len(agents)} 个 Agent 正在并行讨论...")
                prev_text = previous_summaries[-1] if previous_summaries else None

                agent_tasks = [
                    agent.discuss(proposal, round_num, prev_text)
                    for agent in agents
                ]
                results = await asyncio.gather(*agent_tasks, return_exceptions=True)

                agent_opinions: List[AgentOpinion] = []
                for agent, result in zip(agents, results):
                    if isinstance(result, Exception):
                        logger.error(f"| ❌ Agent {agent.agent_id} ({agent.model_name}) 失败: {result}")
                        agent_opinions.append(AgentOpinion(
                            agent_model=agent.model_name,
                            opinions=[],
                            responses_to_challenges=[],
                            additional_capabilities=[],
                            overall_assessment=f"调用失败: {result}",
                        ))
                    else:
                        logger.info(f"| ✅ Agent {agent.agent_id} ({agent.model_name}) 完成")
                        agent_opinions.append(result)

                logger.info(f"| 📊 Manager 正在汇总第 {round_num} 轮讨论...")
                summary = await manager.summarize_round(
                    proposal, round_num, agent_opinions, previous_summaries,
                )
                summary_text = manager.render_summary_text(summary)
                previous_summaries.append(summary_text)

                logger.info(f"| ✅ 共识条目: {len(summary.consensus_items)}")
                logger.info(f"| 分歧点: {len(summary.divergence_points)}")
                logger.info(f"| 质疑数: {len(summary.challenges)}")
                if summary.challenges:
                    for i, ch in enumerate(summary.challenges, 1):
                        logger.info(f"|   质疑{i}: {ch[:80]}{'...' if len(ch) > 80 else ''}")
                logger.info(f"| 继续讨论: {summary.should_continue}")

                discussion_log.append({
                    "round": round_num,
                    "agent_opinions": [op.model_dump() for op in agent_opinions],
                    "summary": summary.model_dump(),
                })

                if summary.consensus_items:
                    final_capabilities = summary.consensus_items

                self._write_result(DebateResult(
                    topic=proposal.topic,
                    total_rounds=round_num,
                    participating_models=[a.model_name for a in agents],
                    capabilities=final_capabilities,
                    discussion_log=discussion_log,
                    generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ), output_path)
                logger.info(f"| 💾 第 {round_num} 轮结果已增量写入: {output_path}")

                if not summary.should_continue:
                    logger.info(f"| ✅ 讨论在第 {round_num} 轮收敛，结束。")
                    break

                if summary.new_questions:
                    proposal.capabilities.extend(
                        q for q in summary.new_questions
                        if q not in proposal.capabilities
                    )

            cap_summary = "\n".join(
                f"  - {c.capability_name} [{c.irreplaceability}] 稳定性:{c.stability_horizon} (置信度:{c.confidence})"
                for c in final_capabilities
            )

            return ToolResponse(
                success=True,
                message=(
                    f"人类能力讨论完成。\n"
                    f"主题: {proposal.topic}\n"
                    f"总轮数: {len(discussion_log)}\n"
                    f"参与模型: {', '.join(a.model_name for a in agents)}\n"
                    f"输出能力条目数: {len(final_capabilities)}\n"
                    f"\n能力摘要:\n{cap_summary}\n"
                    f"\n结果文件: {output_path}"
                ),
                extra=ToolExtra(
                    file_path=output_path,
                    data={
                        "topic": proposal.topic,
                        "total_rounds": len(discussion_log),
                        "capabilities_count": len(final_capabilities),
                        "output_path": output_path,
                    },
                ),
            )

        except Exception as e:
            logger.error(f"❌ 人类能力讨论失败: {e}")
            return ToolResponse(success=False, message=f"讨论过程出错: {e}")

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------

    def _get_result_path(self, topic: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:50]
        filename = f"debate_{safe_topic}_{timestamp}.json"
        return os.path.join(self.base_dir, filename)

    def _write_result(self, result: DebateResult, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
