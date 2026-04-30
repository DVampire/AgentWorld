"""SOP (Standard Operating Procedure) agent.

A fork of `tool_calling.py` specialised for running SOP-type skills. The runtime
logic is identical — the differences are:

* a distinct class name (`SopAgent`) and registered name (`sop_agent`) so the
  planner can target it directly,
* a distinct default prompt (`sop`) that teaches the model the phase-by-phase
  SOP execution protocol (see `prompt/template/sop.py`).

The generic `tool_calling_agent` remains unchanged — it continues to serve as a
plain, skill-free tool runner.
"""

import asyncio
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import Field, ConfigDict

from src.message import Message
from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput
from src.logger import logger
from src.utils import assemble_project_path, dedent, parse_tool_args
from src.tool.server import tool_manager
from src.skill.server import skill_manager
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext



@AGENT.register_module(force=True)
class SopAgent(Agent):
    """SOP agent — runs domain-specific SOP skills as a subagent."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="sop_agent", description="The name of the SOP agent.")
    description: str = Field(
        default=(
            "A subagent that loads domain-specific SOP (Standard Operating "
            "Procedure) skills and executes them phase-by-phase via tool calls."
        ),
        description="The description of the SOP agent.",
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the SOP agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        require_grad: bool = False,
        **kwargs,
    ):
        if not prompt_name:
            prompt_name = "sop"

        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs,
        )

    async def initialize(self):
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        await super().initialize()

    async def _get_tracer_and_record(self) -> tuple[Tracer, Record]:
        tracer = Tracer()
        record = Record()
        if os.path.exists(self.tracer_save_path):
            await tracer.load_from_json(self.tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record
        return tracer, record

    async def _get_tool_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
        tool_context = "<tool_context>"
        tool_context += dedent(f"""
            <available_tools>
            {await tool_manager.get_contract()}
            </available_tools>
        """)
        tool_context += "</tool_context>"
        return {"tool_context": tool_context}

    async def _think_and_action(
        self,
        messages: List[Message],
        task_id: str,
        step_number: int,
        record: Record = None,
        ctx: SessionContext = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Think and tool calls for one step."""
        done = False
        result = None
        reasoning = None

        record_data = {
            "thinking": None,
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "actions": [],
        }

        try:
            think_output = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ThinkOutput,
            )
            think_output = think_output.extra.parsed_model

            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.actions

            record_data["thinking"] = thinking
            record_data["evaluation_previous_goal"] = evaluation_previous_goal
            record_data["memory"] = memory
            record_data["next_goal"] = next_goal

            logger.info(f"| 💭 Thinking: {thinking}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {actions}")

            action_results = []

            for i, action in enumerate(actions):
                action_type = action.type
                action_name = action.name
                action_args_str = action.args
                action_args = parse_tool_args(action_args_str) if action_args_str else {}

                logger.info(f"| 📝 Action {i+1}/{len(actions)}: [{action_type}] {action_name}")
                logger.info(f"| 📝 Args: {action_args}")

                if action_type == "skill":
                    response = await skill_manager(
                        name=action_name,
                        input=action_args,
                        ctx=ctx,
                    )
                    action_result = response.message
                    action_extra = response.extra if hasattr(response, 'extra') else None

                    logger.info(f"| ✅ Skill '{action_name}' completed (success={response.success})")
                    logger.info(f"| 📄 Result: {str(action_result)[:500]}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append({"tool_name": action_name, **record_extra})

                else:
                    tool_response = await tool_manager(
                        name=action_name,
                        input=action_args,
                        ctx=ctx,
                    )
                    action_result = tool_response.message
                    action_extra = tool_response.extra if hasattr(tool_response, 'extra') else None

                    logger.info(f"| ✅ Tool '{action_name}' completed")
                    logger.info(f"| 📄 Result: {str(action_result)}")

                    action_dict = action.model_dump()
                    action_dict["output"] = action_result
                    action_results.append(action_dict)

                    record_extra = {}
                    record_extra.update(action_dict)
                    if action_extra is not None:
                        record_extra['extra'] = action_extra.model_dump()
                    record_data["actions"].append({"tool_name": action_name, **record_extra})

                    if action_name == "done_tool":
                        done = True
                        result = action_result
                        reasoning = action_extra.data.get('reasoning', None) if action_extra and action_extra.data else None
                        break

            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "actions": action_results,
            }

            if record is not None:
                record.action = record_data

            memory_name = self.memory_name
            if self.use_memory and memory_name:
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=step_number,
                    event_type=EventType.TOOL_STEP,
                    data=event_data,
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=ctx,
                )

        except Exception as e:
            logger.error(f"| Error in thinking and tool step: {e}")

        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning,
            "step_data": record_data,
        }
        return response_dict

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        """Main entry point for SOP agent through agent_manager."""
        logger.info(f"| 🚀 Starting SopAgent: {task}")

        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()

        tracer_save_path = os.path.join(self.workdir, f"tracer_{ctx.id}.json")
        tracer = Tracer()
        record = Record()
        if os.path.exists(tracer_save_path):
            await tracer.load_from_json(tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record

        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task

        memory_name = self.memory_name
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")

        if self.use_memory and memory_name:
            await memory_manager.start_session(memory_name=memory_name, ctx=ctx)
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=0,
                event_type=EventType.TASK_START,
                data=dict(task=enhanced_task),
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx,
            )
        else:
            logger.info(f"| ⏭️ Memory disabled (use_memory={self.use_memory}), skipping session management")

        messages = await self._get_messages(enhanced_task, ctx=ctx)

        step_number = 0
        trajectory_steps = []

        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")

            response = await self._think_and_action(messages, task_id, step_number, ctx=ctx, record=record)
            step_number += 1
            trajectory_steps.append(response.pop("step_data", {}))

            await tracer.add_record(
                observation=record.observation,
                action=record.action,
                task_id=task_id,
                ctx=ctx,
            )
            await tracer.save_to_json(tracer_save_path)

            messages = await self._get_messages(enhanced_task, ctx=ctx)

            if response["done"]:
                break

        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "The task has not been completed.",
                "reasoning": "Reached the maximum number of steps.",
            }

        memory_name = self.memory_name
        if self.use_memory and memory_name:
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=step_number,
                event_type=EventType.TASK_END,
                data=response,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx,
            )
            await memory_manager.end_session(memory_name=memory_name, ctx=ctx)

        await tracer.save_to_json(tracer_save_path)

        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")

        return AgentResponse(
            success=response["done"],
            message=response["result"],
            extra=AgentExtra(
                data={**response, "trajectory": trajectory_steps},
            ),
        )
