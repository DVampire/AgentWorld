import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.messages import HumanMessage, SystemMessage

from src.logger import logger
from src.optimizers.base_optimizer import BaseOptimizer


REFLECTION_SYSTEM_PROMPT = """You are an expert at analyzing agent execution results and identifying areas for prompt improvement.

Your task is to:
1. Analyze the agent's execution result and identify what went wrong or what could be improved
2. Reflect on how the current prompt might have contributed to these issues
3. Provide specific, actionable feedback on how to improve the prompt

Focus on:
- Clarity and specificity of instructions
- Missing or ambiguous guidance
- Unnecessary or confusing elements
- Better structure or organization

Be constructive and specific. Provide concrete suggestions for improvement."""

IMPROVEMENT_SYSTEM_PROMPT = """You are an expert at improving prompts based on feedback and analysis."""

REFLECTION_PROMPT_TEMPLATE = """# Agent Execution Analysis

## Task
{task}

## Current Prompt to Improve
{prompt_to_improve}

## Agent Execution Result
{execution_result}

## Analysis Instructions
Please analyze the execution result and provide:

1. **What went wrong or could be improved?**
   - Identify specific issues in the agent's behavior or output
   - Note any errors, inefficiencies, or suboptimal outcomes

2. **How did the prompt contribute to these issues?**
   - Identify parts of the prompt that may have led to the problems
   - Note any unclear, ambiguous, or missing instructions

3. **Specific recommendations for improving the prompt**
   - Provide concrete suggestions for changes
   - Focus on making instructions clearer, more specific, and better structured

Please provide your analysis and recommendations."""


IMPROVEMENT_PROMPT_TEMPLATE = """# Prompt Improvement Task

## Original Task
{task}

## Current Prompt
{current_prompt}

## Analysis and Feedback
{reflection_analysis}

## Improvement Instructions
Based on the analysis above, please provide an improved version of the prompt that addresses the identified issues.

Requirements:
1. Keep the core purpose and structure of the original prompt
2. Address all the issues identified in the analysis
3. Make instructions clearer, more specific, and better organized
4. Remove any unnecessary or confusing elements
5. Add any missing guidance that would help the agent perform better

Please provide ONLY the improved prompt text, without any additional commentary or explanation."""


class ReflectionOptimizer(BaseOptimizer):
    """Optimizer that improves agent prompts using the Reflection method."""
    
    def __init__(self, agent, log_dir: str):
        """
        Initialize the optimizer.

        Args:
            agent: Agent instance.
            log_dir: Path to the log directory.
        """
        super().__init__(agent, log_dir)
    
    async def _generate_reflection(self, task: str, prompt_text: str, execution_result: str, model: Any) -> str:
        """
        Generate the reflection analysis.

        Args:
            task: Task description.
            prompt_text: Current prompt text.
            execution_result: Agent execution result.
            model: LLM model instance.

        Returns:
            str: Reflection output.
        """
        reflection_prompt = REFLECTION_PROMPT_TEMPLATE.format(
            task=task,
            prompt_to_improve=prompt_text,
            execution_result=str(execution_result)[:2000]  # Clamp length.
        )
        
        messages = [
            SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
            HumanMessage(content=reflection_prompt)
        ]
        
        logger.info(f"| 🤔 Generating reflection analysis...")
        self.logger.write(f"🤔 生成反思分析...")
        
        try:
            response = await model.ainvoke(messages)
            reflection_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"| ✅ Reflection analysis generated ({len(reflection_text)} chars)")
            self.logger.write(f"✅ 反思分析生成完成 ({len(reflection_text)} 字符)\n")
            self.logger.write(f"反思分析:\n{reflection_text}\n")
            
            return reflection_text
        except Exception as e:
            logger.error(f"| ❌ Error generating reflection: {e}")
            self.logger.write(f"❌ 生成反思分析时出错: {e}\n")
            raise
    
    async def _improve_prompt(self, task: str, current_prompt: str, reflection_analysis: str, model: Any) -> str:
        """
        Improve the prompt using the reflection analysis.

        Args:
            task: Task description.
            current_prompt: Current prompt text.
            reflection_analysis: Reflection analysis output.
            model: LLM model instance.

        Returns:
            str: Improved prompt text.
        """
        improvement_prompt = IMPROVEMENT_PROMPT_TEMPLATE.format(
            task=task,
            current_prompt=current_prompt,
            reflection_analysis=reflection_analysis
        )
        
        messages = [
            SystemMessage(content=IMPROVEMENT_SYSTEM_PROMPT),
            HumanMessage(content=improvement_prompt)
        ]
        
        logger.info(f"| ✨ Generating improved prompt...")
        self.logger.write(f"✨ 生成改进后的提示词...")
        
        try:
            response = await model.ainvoke(messages)
            improved_text = response.content if hasattr(response, 'content') else str(response)
            
            # Strip potential Markdown code fences.
            improved_text = improved_text.strip()
            if improved_text.startswith("```"):
                # Remove code block markers.
                lines = improved_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                improved_text = "\n".join(lines).strip()
            
            logger.info(f"| ✅ Improved prompt generated ({len(improved_text)} chars)")
            self.logger.write(f"✅ 改进后的提示词生成完成 ({len(improved_text)} 字符)\n")
            self.logger.write(f"改进后的提示词:\n{improved_text}\n")
            
            return improved_text
        except Exception as e:
            logger.error(f"| ❌ Error improving prompt: {e}")
            self.logger.write(f"❌ 改进提示词时出错: {e}\n")
            raise
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3
    ):
        """
        Optimize the agent prompt using the Reflection approach.

        Args:
            task: Task description to optimize for.
            files: Optional list of attachments.
            optimization_steps: Number of optimization iterations (execute -> reflect -> improve per iteration).
        """
        logger.info(f"| 🚀 Starting Reflection optimization...")
        logger.info(f"| 📋 Task: {task}")
        logger.info(f"| 🔄 Optimization steps: {optimization_steps}")
        
        self.logger.write(f"\n{'='*60}")
        self.logger.write(f"Reflection优化开始")
        self.logger.write(f"{'='*60}")
        self.logger.write(f"任务: {task}")
        self.logger.write(f"优化迭代次数: {optimization_steps}\n")
        
        # 1. Extract optimizable variables.
        logger.info(f"| 📊 Extracting optimizable variables...")
        self.optimizable_vars, self.var_mapping = self.extract_optimizable_variables()
        
        if not self.optimizable_vars:
            logger.warning("| ⚠️ No optimizable variables found (require_grad=True). Skipping optimization.")
            self.logger.write("⚠️ 未找到可优化的变量 (require_grad=True)，跳过优化\n")
            return
        
        # 2. Retrieve the model.
        # Use the agent's model for reflection and improvement.
        model = getattr(self.agent, 'model', None)
        if model is None:
            raise ValueError("No model available. Please ensure agent has a model attribute with ainvoke method.")
        
        # Ensure the model exposes an `ainvoke` method.
        if not hasattr(model, 'ainvoke'):
            raise ValueError(f"Model {type(model)} does not have ainvoke method. Please use a LangChain-compatible model.")
        
        # 3. Run the optimization loop.
        for opt_step in range(optimization_steps):
            logger.info(f"\n| {'='*60}")
            logger.info(f"| Reflection Optimization Step {opt_step + 1}/{optimization_steps}")
            logger.info(f"| {'='*60}")
            
            self.logger.write(f"\n{'='*60}")
            self.logger.write(f"优化步骤 {opt_step + 1}/{optimization_steps}")
            self.logger.write(f"{'='*60}\n")
            
            try:
                # 3.1 Execute the task.
                logger.info(f"| 🔄 Executing task with current prompt...")
                self.logger.write(f"🔄 使用当前提示词执行任务...\n")
                
                agent_result = await self.agent.ainvoke(task, files=files)
                
                logger.info(f"| ✅ Task execution completed")
                logger.info(f"| 📄 Result: {str(agent_result)[:200]}...")
                self.logger.write(f"✅ 任务执行完成\n")
                self.logger.write(f"执行结果: {str(agent_result)[:500]}...\n")
                
                # 3.2 Reflect on and improve each optimizable variable.
                for orig_var in self.optimizable_vars:
                    var_name = orig_var.name if hasattr(orig_var, 'name') else 'unknown'
                    var_desc = orig_var.description if hasattr(orig_var, 'description') else f"Prompt module: {var_name}"
                    
                    logger.info(f"\n| 📝 Optimizing variable: {var_name}")
                    logger.info(f"| 📋 Description: {var_desc}")
                    self.logger.write(f"\n📝 优化变量: {var_name}\n")
                    self.logger.write(f"描述: {var_desc}\n")
                    
                    # Retrieve the current variable value.
                    current_value = self.get_variable_value(orig_var)
                    
                    # 3.2.1 Generate reflection analysis.
                    reflection_analysis = await self._generate_reflection(
                        task=task,
                        prompt_text=current_value,
                        execution_result=agent_result,
                        model=model
                    )
                    
                    # 3.2.2 Improve the prompt based on the reflection.
                    improved_value = await self._improve_prompt(
                        task=task,
                        current_prompt=current_value,
                        reflection_analysis=reflection_analysis,
                        model=model
                    )
                    
                    # 3.2.3 Update the variable value.
                    self.set_variable_value(orig_var, improved_value)
                    logger.info(f"| ✅ Variable {var_name} updated")
                    self.logger.write(f"✅ 变量 {var_name} 已更新\n")
                
                # 3.3 Clear caches.
                self.clear_prompt_caches()
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                self.logger.write(f"✅ 优化步骤 {opt_step + 1} 完成\n")
                
            except Exception as e:
                logger.error(f"| ❌ Error in optimization step {opt_step + 1}: {e}")
                self.logger.write(f"❌ 优化步骤 {opt_step + 1} 出错: {e}\n")
                import traceback
                logger.error(f"| Traceback: {traceback.format_exc()}")
                self.logger.write(f"错误详情:\n{traceback.format_exc()}\n")
                # Continue with the next iteration.
                continue
        
        logger.info(f"| ✅ Reflection optimization completed!")
        self.logger.write(f"\n{'='*60}\n")
        self.logger.write(f"✅ Reflection优化完成!\n")
        self.logger.write(f"{'='*60}\n")


async def optimize_agent_with_reflection(
    agent,
    task: str,
    files: Optional[List[str]] = None,
    optimization_steps: int = 3,
    log_dir: Optional[str] = None
):
    """
    Convenience wrapper that optimizes an agent prompt using Reflection.

    Args:
        agent: Agent instance (must expose a model with an `ainvoke` method).
        task: Task description.
        files: Optional list of attachments.
        optimization_steps: Number of optimization iterations (execute -> reflect -> improve).
        log_dir: Log directory path (defaults to the agent's `workdir`).
    """
    if log_dir is None:
        # Try to derive the workdir from the agent.
        if hasattr(agent, 'workdir'):
            log_dir = agent.workdir
        else:
            log_dir = "workdir/optimization"
    
    optimizer = ReflectionOptimizer(agent, log_dir)
    try:
        await optimizer.optimize(task, files, optimization_steps)
    finally:
        optimizer.close()
    
    return optimizer

