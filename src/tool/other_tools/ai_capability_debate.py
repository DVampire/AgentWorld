"""AI 能力自博弈讨论工具 — 多 LLM Agent 并行辩论，Manager 汇总评估，迭代至收敛。

工作流程:
1. Manager 提出 4-5 个 AI 能力议题
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
from typing import List, Dict, Any, Optional, Tuple

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

class AICapabilityItem(BaseModel):
    """单项 AI 能力条目 — AI 能力数据库的基本单元。"""
    capability_name: str = Field(description="能力名称（具体到任务级别，而非泛泛的领域）")
    category: str = Field(description="所属行业/领域分类")
    task_type: str = Field(default="", description="具体任务类型，如'合同审查''代码生成''肺结节检测'")
    human_baseline: str = Field(default="", description="人类平均水平基准描述（用什么指标衡量、人类表现如何）")
    current_level: str = Field(description="相对人类平均水平：已超越/已持平/快速逼近/明显落后")
    evidence: List[str] = Field(default_factory=list, description="支撑当前水平判断的关键证据（benchmark/论文/产品数据）")
    maturity_timeline: str = Field(description="预计达到或超越人类平均水平的时间线（已达到/1-3年/3-5年/5-10年/10年以上）")
    key_milestones: List[str] = Field(default_factory=list, description="达到下一水平所需的关键里程碑或技术突破")
    strengths: List[str] = Field(default_factory=list, description="AI 在此能力上相对人类的优势")
    limitations: List[str] = Field(default_factory=list, description="AI 在此能力上相对人类的局限")
    boundary_description: str = Field(default="", description="能力边界描述：AI 能做什么、不能做什么的分界线在哪")
    trend: str = Field(description="发展趋势描述")
    external_factors: List[str] = Field(default_factory=list, description="影响发展的外部因素（监管/算力/数据/伦理等）")
    description: str = Field(default="", description="约200词的详细描述，综合概述该能力的现状、边界、趋势与关键挑战")
    confidence: float = Field(description="共识置信度 0-1")


class ManagerProposal(BaseModel):
    """Manager 提出的初始议题。"""
    topic: str = Field(description="讨论主题")
    capabilities: List[str] = Field(description="提出的 4-5 个 AI 能力")
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
    consensus_items: List[AICapabilityItem] = Field(default_factory=list, description="已达成共识的能力条目")
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
    capabilities: List[AICapabilityItem] = Field(default_factory=list, description="最终 AI 能力数据库条目")
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
        """对 Manager 提出的能力列表发表意见，并回应 Manager 的质疑。"""

        system_prompt = dedent("""你是一位 AI 技术专家，正在为「AI 能力数据库」项目参与一场多方讨论。
        请用中文回答。你的分析应当客观、专业、有数据支撑。

        该数据库的核心框架：
        - 核心标尺：以「人类平均水平」为基准线
        - 评级体系：已超越人类 / 已持平 / 快速逼近 / 明显落后
        - 时间线：已达到 / 1-3年 / 3-5年 / 5-10年 / 10年以上
        - 关注边界：AI 能做什么与不能做什么的分界线具体在哪里

        对每项能力，你需要：
        1. 明确「人类平均水平」的基准是什么（用什么指标衡量、人类大概什么表现）
        2. 判断 AI 当前处于谱系的哪个位置（已超越/已持平/快速逼近/明显落后），并给出关键证据（benchmark、论文、产品数据）
        3. 精确描述能力边界：AI 能做到什么程度、在什么情况下会失败
        4. 预测时间线，并说明达到下一水平需要什么技术突破
        5. 考虑外部因素：监管政策、算力瓶颈、数据壁垒、伦理约束如何影响发展

        重要：如果主持人在上一轮总结中提出了质疑或追问，你必须认真回应每一条质疑。
        回应时请做到：
        - 提供具体的数据、案例或技术依据来支撑你的论点
        - 如果主持人的质疑有道理，大方修正你之前的判断
        - 如果你认为质疑不成立，给出有力的反驳论据
        - 避免泛泛而谈，尽量做到精确具体

        你也可以建议补充讨论中遗漏的、处于能力边界附近的重要 AI 能力。""")

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

        需要讨论的 AI 能力:
        {chr(10).join(f"  {i+1}. {c}" for i, c in enumerate(proposal.capabilities))}
        {context_block}

        请以 JSON 格式返回你的意见，结构如下:
        {{
            "opinions": [
                {{
                    "capability": "能力名称",
                    "task_type": "具体任务类型",
                    "category": "所属行业/领域",
                    "human_baseline": "人类平均水平基准（用什么指标、人类表现如何）",
                    "current_level": "已超越/已持平/快速逼近/明显落后",
                    "evidence": ["支撑判断的关键证据：benchmark/论文/产品数据"],
                    "boundary_description": "能力边界：AI 能做到什么、在什么情况下失败",
                    "maturity_timeline": "已达到/1-3年/3-5年/5-10年/10年以上",
                    "key_milestones": ["达到下一水平需要的技术突破"],
                    "strengths": ["相对人类的优势"],
                    "limitations": ["相对人类的局限"],
                    "trend": "发展趋势描述",
                    "external_factors": ["监管/算力/数据/伦理等外部影响因素"],
                    "description": "约200词的详细描述，综合概述该能力的现状、边界、趋势与关键挑战",
                    "reasoning": "综合分析理由"
                }}
            ],
            "responses_to_challenges": ["对主持人质疑1的回应", "对主持人质疑2的回应"],
            "additional_capabilities": ["建议补充的处于能力边界附近的能力"],
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

        system_prompt = dedent("""你是一位 AI 研究领域的资深专家和讨论主持人。你正在为「AI 能力数据库」项目遴选讨论议题。

        该数据库的核心定位：
        - 覆盖对象：AI 当前已掌握或在未来 1-10 年内可预期掌握的能力
        - 核心标尺：以「人类平均水平」为基准线——在特定任务上，AI 是已超越、已持平、快速逼近、还是明显落后
        - 覆盖范围：横跨不同行业、不同任务类型中的 AI 表现水平与发展趋势
        - 重点追踪：AI 能力的边界变化——既记录 AI 的强势领域，也记录其当前的局限地带

        你需要根据给定主题，提出 4-5 个具体的 AI 能力进行讨论。选择标准：
        1. 具体到任务级别（如"合同条款审查"而非笼统的"法律能力"）
        2. 覆盖能力谱系的不同位置——至少包含：1 个已超越人类的、1 个快速逼近的、1 个仍明显落后的
        3. 横跨不同行业或任务类型，避免集中在单一领域
        4. 优先选择边界正在快速移动的能力（近 1-2 年有显著进展或突破预期的）

        请用中文回答。""")

        user_prompt = dedent(f"""请根据以下主题，提出 4-5 个 AI 能力进行讨论:

        主题: {task}

        要求：
        - 每个能力必须具体到可衡量的任务级别
        - 明确标注你认为该能力当前大致处于谱系的哪个位置（已超越/已持平/快速逼近/明显落后）
        - discussion_context 中应描述为什么选择这些能力，以及讨论应重点关注的边界问题

        以 JSON 格式返回:
        {{
            "topic": "讨论主题",
            "capabilities": ["具体能力1（当前位置）", "具体能力2（当前位置）", ...],
            "discussion_context": "选择理由与重点关注的边界问题"
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
                    "标准化代码生成与补全（已超越）",
                    "复杂多步数学推理与证明（快速逼近）",
                    "开放域长文本事实性写作（快速逼近）",
                    "实时多模态场景理解与决策（明显落后）",
                    "跨领域常识推理与因果判断（明显落后）",
                ],
                discussion_context=f"围绕'{task}'构建 AI 能力数据库条目，以人类平均水平为基准线，评估各项能力的当前水平、能力边界和发展时间线。",
            )

    async def summarize_round(
        self,
        proposal: ManagerProposal,
        round_number: int,
        agent_opinions: List[AgentOpinion],
        previous_summaries: List[str],
    ) -> RoundSummary:
        """汇总一轮讨论结果，提出质疑，评估是否收敛。"""

        system_prompt = dedent("""你是「AI 能力数据库」项目的讨论主持人，同时也是一位严格的批判性审稿人。
        你有两项核心职责：

        【职责一：整理汇总 — 按数据库标准结构化】
        将各方意见整理为标准化的 AI 能力条目，每个条目必须包含：
        - capability_name: 具体到任务级别的能力名称
        - task_type: 具体任务类型
        - human_baseline: 人类平均水平是什么、用什么指标衡量
        - current_level: 已超越/已持平/快速逼近/明显落后
        - evidence: 支撑判断的关键证据
        - boundary_description: 能力边界——AI 能做什么、做不到什么
        - maturity_timeline: 已达到/1-3年/3-5年/5-10年/10年以上
        - key_milestones: 突破到下一水平需要什么
        - external_factors: 监管/算力/数据等外部影响因素

        【职责二：质疑与追问】——这是你最重要的角色
        你必须对各位专家的观点进行批判性审视，提出有深度的质疑（challenges），特别是：
        - "人类基准"定义模糊时：到底是和什么水平的人类比？用什么指标？基准是否合理？
        - "已超越"判断时：在什么具体任务和指标上超越？是否有反例？是平均超越还是只在窄场景？
        - "快速逼近"判断时：速度有多快？有什么量化证据？可能遇到什么瓶颈导致减速？
        - 边界描述不清时：AI 到底在哪里失败？失败模式是什么？
        - 时间线预测时：基于什么技术路线图？关键假设是什么？如果假设不成立呢？
        - 外部因素被忽略时：监管收紧/算力瓶颈/数据壁垒/伦理问题如何影响时间线？
        - 各方判断矛盾时：具体分歧在哪？谁的证据更可信？

        你的质疑应当尖锐但建设性，目的是让最终的数据库条目经得起推敲。

        【收敛判断标准】
        - 每项能力的 human_baseline 和 current_level 已有清晰定义和可信证据
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
                    "capability_name": "具体到任务级别的能力名称",
                    "category": "所属行业/领域",
                    "task_type": "具体任务类型",
                    "human_baseline": "人类平均水平基准描述",
                    "current_level": "已超越/已持平/快速逼近/明显落后",
                    "evidence": ["关键证据1"],
                    "maturity_timeline": "已达到/1-3年/3-5年/5-10年/10年以上",
                    "key_milestones": ["关键里程碑1"],
                    "strengths": ["优势1"],
                    "limitations": ["局限1"],
                    "boundary_description": "AI能做什么、做不到什么的分界线",
                    "trend": "发展趋势",
                    "external_factors": ["外部影响因素1"],
                    "description": "约200词的综合描述：该能力的现状、边界、趋势与关键挑战",
                    "confidence": 0.8
                }}
            ],
            "divergence_points": ["分歧点1"],
            "challenges": [
                "【质疑1】针对某观点的具体质疑和追问，要求补充论据或澄清",
                "【质疑2】对某个乐观/悲观判断的反驳与追问"
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
            consensus_items = [AICapabilityItem(**item) for item in data.get("consensus_items", [])]
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
                lines.append(f"  ● {item.capability_name} ({item.category} / {item.task_type})")
                lines.append(f"    人类基准: {item.human_baseline}")
                lines.append(f"    当前水平: {item.current_level} | 时间线: {item.maturity_timeline}")
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

_TOOL_DESCRIPTION = """AI 能力自博弈讨论工具 — 多 LLM 并行辩论，评估 AI 能力边界。

该工具通过 Manager-Agent 多轮讨论机制，评估 AI 在不同行业、不同任务类型中的能力水平与发展趋势。

工作流程:
1. Manager 根据任务提出 4-5 个 AI 能力议题
2. 多个不同模型的 LLM Agent 并行讨论
3. Manager 汇总评估、判断是否收敛
4. 迭代至讨论收敛，输出结构化的 AI 能力数据库（JSON）

Args:
- task (str): 讨论主题描述，例如 "评估 AI 在金融行业的能力边界"
- max_rounds (int): 最大讨论轮数，默认 5

Example: {"name": "ai_capability_debate", "args": {"task": "评估 AI 在软件工程领域的能力现状与未来趋势"}}.
"""

_DEFAULT_AGENT_MODELS = [
    "openrouter/gemini-3.1-pro-preview",
    "openrouter/gpt-5.4-pro",
    "openrouter/claude-opus-4.6",
    "openrouter/grok-4.1-fast",
]

_MAX_ROUNDS_LIMIT = 5


@TOOL.register_module(force=True)
class AICapabilityDebateTool(Tool):
    """AI 能力自博弈讨论工具。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "ai_capability_debate"
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
        default="workdir/ai_capability_debate",
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
            self.base_dir = assemble_project_path("workdir/ai_capability_debate")

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
        """执行多轮 AI 能力自博弈讨论。"""
        max_rounds = min(max_rounds, _MAX_ROUNDS_LIMIT)

        try:
            manager = DebateManager(model_name=self.model_name)
            agents = [
                DebateAgent(model_name=m, agent_id=f"agent_{i}")
                for i, m in enumerate(self.agent_models)
            ]

            # Step 1 — Manager 提出议题
            logger.info(f"📋 Step 1: Manager 正在生成讨论议题 — {task}")
            proposal = await manager.propose_topic(task)
            logger.info(f"| ✅ 议题: {proposal.topic}")
            logger.info(f"| 能力列表: {', '.join(proposal.capabilities)}")

            discussion_log: List[Dict[str, Any]] = []
            previous_summaries: List[str] = []
            final_capabilities: List[AICapabilityItem] = []
            output_path = self._get_result_path(proposal.topic)

            for round_num in range(1, max_rounds + 1):
                logger.info(f"\n🔄 === 第 {round_num} 轮讨论 ===")

                # Step 2 — Agents 并行讨论
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

                # Step 3 — Manager 汇总评估
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

                # 每轮结束后增量写入 JSON
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
                f"  - {c.capability_name} [{c.current_level}] 时间线:{c.maturity_timeline} (置信度:{c.confidence})"
                for c in final_capabilities
            )

            return ToolResponse(
                success=True,
                message=(
                    f"AI 能力讨论完成。\n"
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
            logger.error(f"❌ AI 能力讨论失败: {e}")
            return ToolResponse(success=False, message=f"讨论过程出错: {e}")

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------

    def _get_result_path(self, topic: str) -> str:
        """生成结果文件路径（整个讨论过程使用同一个文件）。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic)[:50]
        filename = f"debate_{safe_topic}_{timestamp}.json"
        return os.path.join(self.base_dir, filename)

    def _write_result(self, result: DebateResult, filepath: str) -> None:
        """将讨论结果写入（或覆盖更新）JSON 文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
