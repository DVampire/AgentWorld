"""未来能力自博弈讨论工具 — 多 LLM Agent 并行辩论，发掘 AI 时代被重塑的人类学习能力。

核心定位：
覆盖对象是因 AI 的出现而使其人类学习价值被根本性重塑的能力乃至 AI 不具备的能力。
时间窗口横跨未来一至二十年。
"重塑"包含两种情形：
  一是此前不存在、由 AI 时代催生的全新能力；
  二是原本存在但长期被低估、在 AI 时代因稀缺性被显著放大的能力。

工作流程:
1. Manager 提出 10+ 个未来能力议题（广撒网）
2. 多个 LLM Agent（不同模型）并行讨论、扩充、论证
3. Manager 汇总、筛选、质疑
4. 迭代至收敛，输出精选的未来能力数据库（JSON）
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

class FutureCapabilityItem(BaseModel):
    """单项未来能力条目 — 未来能力数据库的基本单元。"""
    capability_name: str = Field(description="能力名称（具体到可学习、可培养的技能级别）")
    category: str = Field(description="所属行业/领域分类")
    reshaping_type: str = Field(description="重塑类型：AI时代全新催生 / 长期被低估现被放大 / 人机协作新形态")
    scenario_type: str = Field(default="", description="具体应用场景，如'AI辅助下的哲学思辨教学''现场即兴表演与观众情感共鸣'")
    why_reshaped: str = Field(default="", description="为什么 AI 的出现使该能力的学习价值被根本性重塑——因果链条")
    ai_gap: str = Field(default="", description="AI 在该能力上的核心缺陷或无法触及之处")
    scarcity_driver: str = Field(default="", description="稀缺性驱动因素：是什么让该能力在 AI 时代变得稀缺且珍贵")
    value_trajectory: str = Field(description="价值曲线：急剧上升 / 稳步上升 / 先降后升 / 长期被忽视即将爆发")
    time_horizon: str = Field(description="价值兑现时间窗口：1-3年 / 3-5年 / 5-10年 / 10-20年")
    learning_path: List[str] = Field(default_factory=list, description="培养路径：如何系统性地学习和训练该能力")
    real_world_examples: List[str] = Field(default_factory=list, description="现实中已经出现的信号或案例")
    market_signal: List[str] = Field(default_factory=list, description="市场信号：薪资变化/岗位需求/创业方向等")
    risk_factors: List[str] = Field(default_factory=list, description="风险因素：可能导致该能力价值未如预期的因素")
    synergy_with_ai: str = Field(default="", description="与 AI 的协作模式：该能力如何与 AI 形成互补而非竞争")
    description: str = Field(default="", description="约200词的详细描述，综合阐述该能力为何在 AI 时代被重塑、价值逻辑与发展前景")
    confidence: float = Field(description="共识置信度 0-1")
    priority_rank: int = Field(default=0, description="优先级排名（经筛选后的排序，1 为最高优先）")


class ManagerProposal(BaseModel):
    """Manager 提出的初始议题。"""
    topic: str = Field(description="讨论主题")
    capabilities: List[str] = Field(description="提出的 10+ 个未来能力")
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
    consensus_items: List[FutureCapabilityItem] = Field(default_factory=list, description="已达成共识的能力条目")
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
    capabilities: List[FutureCapabilityItem] = Field(default_factory=list, description="最终未来能力数据库条目")
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
        """对 Manager 提出的未来能力列表发表意见，并回应 Manager 的质疑。"""

        system_prompt = dedent("""你是一位兼通技术趋势、产业经济、教育变革与人文思想的跨界战略家。
        你正在为「未来能力数据库」项目参与一场多方讨论。请用中文回答。

        该数据库的核心定位：
        - 覆盖对象是因 AI 的出现而使其人类学习价值被根本性重塑的能力，乃至 AI 不具备的能力
        - 时间窗口横跨未来一至二十年
        - "重塑"包含两种情形：
          ① 此前不存在、由 AI 时代催生的全新能力
          ② 原本存在但长期被低估、在 AI 时代因稀缺性被显著放大的能力
        - 典型示例：哲学思辨能力（文科复兴）、让人快乐的能力（魔术、杂技、即兴表演）、讲故事的能力

        核心评估框架：
        - 重塑类型：AI时代全新催生 / 长期被低估现被放大 / 人机协作新形态
        - 价值曲线：急剧上升 / 稳步上升 / 先降后升 / 长期被忽视即将爆发
        - 时间窗口：1-3年 / 3-5年 / 5-10年 / 10-20年
        - 关注因果链：为什么 AI 的出现「导致」该能力的价值被重塑？因果机制是什么？
        - 关注稀缺性：是什么让该能力变得稀缺？是 AI 替代了周边能力？是注意力经济的转向？还是人类本能的退化？

        对每项能力，你需要：
        1. 说清楚重塑的因果链——不是"AI 来了所以 X 变重要了"，而是具体的传导机制
        2. 区分该能力是"全新催生"还是"价值放大"，给出判据
        3. AI 在该能力上的核心缺陷具体是什么——不要泛泛说"AI没有情感"
        4. 给出现实中已经出现的信号（市场数据、薪资变化、创业方向、教育改革等）
        5. 描述可行的培养路径——如何系统性地学习和训练该能力
        6. 评估风险——什么情况下该能力的价值可能不如预期

        重要原则：
        - 大胆提出，鼓励发现非显而易见的能力（不要只列AI伦理、提示词工程这类陈词滥调）
        - 每个行业至少提出10个以上能力，宁多勿少，后续再筛选
        - 关注那些"此刻听起来不太正经但未来可能极有价值"的能力

        重要：如果主持人在上一轮总结中提出了质疑或追问，你必须认真回应每一条质疑。""")

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
            4. 如果主持人指出了你之前分析的不足，请补充或修正
            5. 在第二轮及之后，重点补充被遗漏的能力、强化论证薄弱的条目""")

        user_prompt = dedent(f"""第 {round_number} 轮讨论

        讨论主题: {proposal.topic}
        讨论背景: {proposal.discussion_context}

        需要讨论的未来能力:
        {chr(10).join(f"  {i+1}. {c}" for i, c in enumerate(proposal.capabilities))}
        {context_block}

        请以 JSON 格式返回你的意见，结构如下:
        {{
            "opinions": [
                {{
                    "capability": "能力名称（具体到可学习的技能级别）",
                    "category": "所属行业/领域",
                    "reshaping_type": "AI时代全新催生 / 长期被低估现被放大 / 人机协作新形态",
                    "scenario_type": "具体应用场景",
                    "why_reshaped": "为什么 AI 的出现使该能力被根本性重塑——因果链条",
                    "ai_gap": "AI 在该能力上的核心缺陷或无法触及之处",
                    "scarcity_driver": "稀缺性驱动因素",
                    "value_trajectory": "急剧上升/稳步上升/先降后升/长期被忽视即将爆发",
                    "time_horizon": "1-3年/3-5年/5-10年/10-20年",
                    "learning_path": ["培养路径步骤"],
                    "real_world_examples": ["现实中已出现的信号或案例"],
                    "market_signal": ["市场信号"],
                    "risk_factors": ["风险因素"],
                    "synergy_with_ai": "与 AI 的协作互补模式",
                    "description": "约200词的详细描述",
                    "reasoning": "综合分析理由"
                }}
            ],
            "responses_to_challenges": ["对主持人质疑1的回应", "对主持人质疑2的回应"],
            "additional_capabilities": ["建议补充的未来能力（鼓励大胆提出非显而易见的能力）"],
            "overall_assessment": "整体评估"
        }}

        重要：每项能力必须有清晰的因果链（why_reshaped），说明 AI 的出现如何导致其价值被重塑。
        鼓励提出不少于10个能力，宁多勿少。
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
    """讨论协调者，负责发起议题、汇总评估、筛选排序、判断收敛。"""

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
        """根据用户任务生成讨论议题和初始能力列表（10+个）。"""

        system_prompt = dedent("""你是一位横跨科技、人文、产业的未来学家和讨论主持人。
        你正在为「未来能力数据库」项目遴选讨论议题。

        该数据库的核心定位：
        - 覆盖对象是因 AI 的出现而使其人类学习价值被根本性重塑的能力，乃至 AI 不具备的能力
        - 时间窗口横跨未来一至二十年
        - "重塑"包含两种情形：
          ① 此前不存在、由 AI 时代催生的全新能力（如：AI 协作架构设计、算法审计与偏见检测）
          ② 原本存在但长期被低估、在 AI 时代因稀缺性被显著放大的能力（如：哲学思辨、让人快乐的能力、讲故事的能力、手工艺、魔术与杂技）

        你需要根据给定主题，提出 10 个以上具体的未来能力进行讨论。选择标准：
        1. 具体到可学习、可培养的技能级别（如"即兴叙事与现场故事构建"而非笼统的"沟通能力"）
        2. 两种类型都要覆盖：至少 4 个全新催生的 + 至少 4 个价值被放大的
        3. 跨越不同领域：技术、人文、艺术、身体技能、社交、哲学等
        4. 大胆提出：不要局限于"AI 伦理""提示词工程"这些显而易见的答案
        5. 关注"此刻听起来不太正经但未来可能极有价值"的能力

        请用中文回答。""")

        user_prompt = dedent(f"""请根据以下主题，提出 10 个以上未来能力进行讨论:

        主题: {task}

        要求：
        - 每个能力必须具体到可学习的技能级别
        - 标注重塑类型（AI时代全新催生 / 长期被低估现被放大）
        - discussion_context 中应描述选择逻辑和讨论应关注的核心问题
        - 宁多勿少，鼓励大胆提出

        以 JSON 格式返回:
        {{
            "topic": "讨论主题",
            "capabilities": ["具体能力1（重塑类型）", "具体能力2（重塑类型）", ...],
            "discussion_context": "选择逻辑与核心问题"
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
                    "即兴叙事与现场故事构建能力（长期被低估现被放大）",
                    "哲学思辨与第一性原理推演（长期被低估现被放大）",
                    "让人快乐的表演能力——魔术/杂技/即兴喜剧（长期被低估现被放大）",
                    "人机协作系统架构设计（AI时代全新催生）",
                    "AI 输出的批判性审计与偏见检测（AI时代全新催生）",
                    "跨模态感官体验设计——味觉/触觉/嗅觉（长期被低估现被放大）",
                    "深度倾听与非结构化对话引导（长期被低估现被放大）",
                    "手工艺与身体性创造——木工/陶艺/烹饪（长期被低估现被放大）",
                    "复杂利益相关方的面对面谈判与调解（长期被低估现被放大）",
                    "AI 时代的注意力管理与深度专注训练（AI时代全新催生）",
                    "社群编织与线下社区运营（长期被低估现被放大）",
                    "人类价值观对齐与伦理框架构建（AI时代全新催生）",
                ],
                discussion_context=f"围绕'{task}'发掘因 AI 出现而被根本性重塑的能力。两条主线：一是 AI 催生的全新能力，二是长期被低估但因稀缺性被放大的传统能力。",
            )

    async def summarize_round(
        self,
        proposal: ManagerProposal,
        round_number: int,
        agent_opinions: List[AgentOpinion],
        previous_summaries: List[str],
    ) -> RoundSummary:
        """汇总一轮讨论结果，筛选排序，提出质疑，评估是否收敛。"""

        system_prompt = dedent("""你是「未来能力数据库」项目的讨论主持人，同时是一位犀利的战略评审者。
        你有三项核心职责：

        【职责一：整理汇总 — 按数据库标准结构化】
        将各方意见整理为标准化的未来能力条目，每个条目必须包含：
        - capability_name: 具体到可学习技能级别的能力名称
        - reshaping_type: AI时代全新催生 / 长期被低估现被放大 / 人机协作新形态
        - why_reshaped: 因果链条——AI 的出现如何导致该能力价值被重塑
        - ai_gap: AI 的核心缺陷
        - scarcity_driver: 稀缺性驱动因素
        - value_trajectory: 价值曲线走向
        - time_horizon: 价值兑现时间窗口
        - learning_path: 培养路径
        - real_world_examples: 现实信号
        - market_signal: 市场信号
        - risk_factors: 风险因素

        【职责二：筛选与排序】
        从大量候选能力中筛选出真正有价值的条目：
        - 合并重复或高度重叠的能力
        - 淘汰因果链薄弱、"AI 来了所以 X 重要"这种逻辑跳跃的条目
        - 淘汰过于泛泛、不具备可操作培养路径的条目
        - 对保留条目进行优先级排序（priority_rank），1 为最高优先
        - 筛选标准：因果链清晰度 > 现实信号强度 > 稀缺性逻辑 > 培养可行性

        【职责三：质疑与追问】——这是你最重要的角色
        你必须对各位专家的观点进行批判性审视，提出有深度的质疑：
        - 因果链质疑：该能力的价值上升真的是 AI 导致的吗？还是本来就有这个趋势？
        - 稀缺性质疑：该能力真的会稀缺吗？会不会反而因为大家都看到机会而涌入？
        - 时间窗口质疑：价值兑现真的在预期时间内吗？有没有可能是一厢情愿？
        - AI 能力边界质疑：你说 AI 做不到，但 5 年后呢？有没有技术路线图指向突破？
        - 可培养性质疑：说了培养路径，但真的可行吗？多少人能走通？市场容量够吗？
        - 陈词滥调检测："AI 伦理""批判性思维""创造力"这类是否过于空泛？能否更具体？

        【收敛判断标准】
        - 每项保留能力的因果链（why_reshaped）已被论证清楚
        - 能力条目已经过筛选和排序
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
                    "capability_name": "具体到可学习技能级别的能力名称",
                    "category": "所属行业/领域",
                    "reshaping_type": "AI时代全新催生/长期被低估现被放大/人机协作新形态",
                    "scenario_type": "具体应用场景",
                    "why_reshaped": "因果链条",
                    "ai_gap": "AI 的核心缺陷",
                    "scarcity_driver": "稀缺性驱动因素",
                    "value_trajectory": "急剧上升/稳步上升/先降后升/长期被忽视即将爆发",
                    "time_horizon": "1-3年/3-5年/5-10年/10-20年",
                    "learning_path": ["培养路径"],
                    "real_world_examples": ["现实信号"],
                    "market_signal": ["市场信号"],
                    "risk_factors": ["风险因素"],
                    "synergy_with_ai": "与 AI 的协作互补模式",
                    "description": "约200词的综合描述",
                    "confidence": 0.8,
                    "priority_rank": 1
                }}
            ],
            "divergence_points": ["分歧点1"],
            "challenges": [
                "【质疑1】针对某能力因果链的质疑",
                "【质疑2】对某个价值判断的反驳"
            ],
            "new_questions": ["需要下一轮讨论的问题"],
            "should_continue": false,
            "reasoning": "判断理由"
        }}

        重要：
        1. 对所有候选能力进行筛选，淘汰因果链薄弱或过于空泛的条目
        2. 对保留条目按优先级排序（priority_rank）
        3. challenges 字段必须具体、尖锐、有针对性
        4. 如果讨论还需继续（should_continue=true），challenges 不应为空
        只返回 JSON。""")

        raw = await self._call_llm(system_prompt, user_prompt)
        cleaned = _strip_fences(raw)

        try:
            data = json.loads(cleaned)
            consensus_items = [FutureCapabilityItem(**item) for item in data.get("consensus_items", [])]
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
            lines.append("\n【已达成共识的能力条目（按优先级排序）】")
            sorted_items = sorted(summary.consensus_items, key=lambda x: x.priority_rank if x.priority_rank > 0 else 999)
            for item in sorted_items:
                rank_str = f"#{item.priority_rank}" if item.priority_rank > 0 else "未排序"
                lines.append(f"  ● [{rank_str}] {item.capability_name} ({item.category})")
                lines.append(f"    重塑类型: {item.reshaping_type} | 价值曲线: {item.value_trajectory}")
                lines.append(f"    时间窗口: {item.time_horizon}")
                if item.why_reshaped:
                    lines.append(f"    因果链: {item.why_reshaped[:100]}{'...' if len(item.why_reshaped) > 100 else ''}")
                if item.scarcity_driver:
                    lines.append(f"    稀缺性: {item.scarcity_driver[:80]}{'...' if len(item.scarcity_driver) > 80 else ''}")
                if item.real_world_examples:
                    lines.append(f"    现实案例: {'; '.join(item.real_world_examples[:3])}")
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

_TOOL_DESCRIPTION = """未来能力自博弈讨论工具 — 多 LLM 并行辩论，发掘 AI 时代被重塑的人类学习能力。

覆盖对象是因 AI 的出现而使其人类学习价值被根本性重塑的能力。"重塑"包含两种情形：
一是此前不存在、由 AI 时代催生的全新能力；二是原本存在但长期被低估、在 AI 时代因稀缺性被显著放大的能力。

工作流程:
1. Manager 根据任务提出 10+ 个未来能力议题（广撒网）
2. 多个不同模型的 LLM Agent 并行讨论、扩充
3. Manager 汇总、筛选排序、质疑追问
4. 迭代至收敛，输出经筛选排序的未来能力数据库（JSON）

Args:
- task (str): 讨论主题描述，例如 "发掘教育行业中因 AI 出现而被重塑的未来能力"
- max_rounds (int): 最大讨论轮数，默认 5

Example: {"name": "future_capability_debate", "args": {"task": "发掘金融行业中因 AI 出现而被重塑的未来能力"}}.
"""

_DEFAULT_AGENT_MODELS = [
    "openrouter/gemini-3.1-pro-preview",
    "openrouter/gpt-5.4-pro",
    "openrouter/claude-opus-4.6",
    "openrouter/grok-4.1-fast",
]

_MAX_ROUNDS_LIMIT = 5


@TOOL.register_module(force=True)
class FutureCapabilityDebateTool(Tool):
    """未来能力自博弈讨论工具。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "future_capability_debate"
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
        default="workdir/future_capability_debate",
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
            self.base_dir = assemble_project_path("workdir/future_capability_debate")

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
        """执行多轮未来能力自博弈讨论。"""
        max_rounds = min(max_rounds, _MAX_ROUNDS_LIMIT)

        try:
            manager = DebateManager(model_name=self.model_name)
            agents = [
                DebateAgent(model_name=m, agent_id=f"agent_{i}")
                for i, m in enumerate(self.agent_models)
            ]

            logger.info(f"📋 Step 1: Manager 正在生成讨论议题（10+个）— {task}")
            proposal = await manager.propose_topic(task)
            logger.info(f"| ✅ 议题: {proposal.topic}")
            logger.info(f"| 候选能力数: {len(proposal.capabilities)}")
            for i, c in enumerate(proposal.capabilities, 1):
                logger.info(f"|   {i}. {c}")

            discussion_log: List[Dict[str, Any]] = []
            previous_summaries: List[str] = []
            final_capabilities: List[FutureCapabilityItem] = []
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
                        logger.info(f"| ✅ Agent {agent.agent_id} ({agent.model_name}) 完成，提出 {len(result.opinions)} 项能力")
                        agent_opinions.append(result)

                logger.info(f"| 📊 Manager 正在汇总、筛选第 {round_num} 轮讨论...")
                summary = await manager.summarize_round(
                    proposal, round_num, agent_opinions, previous_summaries,
                )
                summary_text = manager.render_summary_text(summary)
                previous_summaries.append(summary_text)

                logger.info(f"| ✅ 筛选后共识条目: {len(summary.consensus_items)}")
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

            sorted_caps = sorted(final_capabilities, key=lambda x: x.priority_rank if x.priority_rank > 0 else 999)
            cap_summary = "\n".join(
                f"  #{c.priority_rank} {c.capability_name} [{c.reshaping_type}] {c.value_trajectory} | {c.time_horizon} (置信度:{c.confidence})"
                for c in sorted_caps
            )

            return ToolResponse(
                success=True,
                message=(
                    f"未来能力讨论完成。\n"
                    f"主题: {proposal.topic}\n"
                    f"总轮数: {len(discussion_log)}\n"
                    f"参与模型: {', '.join(a.model_name for a in agents)}\n"
                    f"输出能力条目数: {len(final_capabilities)}\n"
                    f"\n能力摘要（按优先级排序）:\n{cap_summary}\n"
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
            logger.error(f"❌ 未来能力讨论失败: {e}")
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
