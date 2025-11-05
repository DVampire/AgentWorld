"""
TextGrad优化器模块
用于使用TextGrad优化Agent的提示词
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# 确保textgrad在Python路径中
# 优先使用项目根目录下的textgrad，如果不存在则使用src/optimizers/textgrad
root_path = Path(__file__).resolve().parents[2]
textgrad_paths = [
    str(root_path / "textgrad"),
    str(root_path / "src" / "optimizers" / "textgrad"),
]

for path in textgrad_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

from ..optimizers import textgrad as tg
from src.logger import logger


class OptimizationLogger:
    """优化日志管理器"""
    
    def __init__(self, log_dir: str):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = log_dir
        self.log_file = None
        self.log_file_path = None
        self._setup_log_file()
    
    def _setup_log_file(self):
        """设置日志文件"""
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(self.log_dir, f"optimization_{timestamp}.log")
        
        # 打开文件（追加模式，UTF-8编码）
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")
        
        logger.info(f"| 📝 Optimization log will be saved to: {self.log_file_path}")
    
    def write(self, message: str):
        """
        写入优化日志文件
        
        Args:
            message: 要写入的消息（不包含换行符）
        """
        if self.log_file:
            # 移除markdown标记，使其在纯文本文件中更易读
            clean_message = message.replace("| ", "").strip()
            self.log_file.write(clean_message + "\n")
            self.log_file.flush()  # 立即写入磁盘
    
    def close(self):
        """关闭优化日志文件"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class TextGradOptimizer:
    """使用TextGrad优化Agent提示词的优化器"""
    
    def __init__(self, agent, log_dir: str):
        """
        初始化优化器
        
        Args:
            agent: Agent实例
            log_dir: 日志目录路径
        """
        self.agent = agent
        self.logger = OptimizationLogger(log_dir)
        self.optimizable_tg_vars = []
        self.var_mapping = {}  # tg_var -> orig_var
        self.prompt_mapping = {}  # tg_var -> prompt_obj (包含该变量的prompt对象)
    
    def find_prompt_objects_with_variables(self) -> List[Tuple[Any, str]]:
        """
        查找Agent中所有包含Variable对象的prompt对象
        
        Returns:
            List[Tuple[prompt_obj, prompt_name]]: (prompt对象, 名称)的列表
        """
        prompt_objects = []
        
        # 从prompt_manager中查找所有prompt对象
        if hasattr(self.agent, 'prompt_manager'):
            pm = self.agent.prompt_manager
            
            # SystemPrompt
            if hasattr(pm, 'system_prompt') and hasattr(pm.system_prompt, 'prompt'):
                prompt_objects.append((pm.system_prompt, 'system_prompt'))
            
            # AgentMessagePrompt
            if hasattr(pm, 'agent_message_prompt') and hasattr(pm.agent_message_prompt, 'prompt'):
                prompt_objects.append((pm.agent_message_prompt, 'agent_message_prompt'))
        
        # 可以扩展：查找其他可能包含Variable的对象
        # 例如：如果有其他类型的prompt对象，在这里添加
        
        return prompt_objects
    
    def extract_optimizable_variables(self) -> Tuple[List[tg.Variable], Dict, Dict]:
        """
        从Agent的所有prompt对象中提取可优化的变量（require_grad=True）
        并转换为textgrad.Variable格式
        
        Returns:
            Tuple[List[tg.Variable], Dict, Dict]: 
                (可优化的变量列表, 变量映射字典, prompt对象映射字典)
                - var_mapping: tg_var -> orig_var
                - prompt_mapping: tg_var -> prompt_obj (包含该变量的prompt对象)
        """
        optimizable_vars = []
        
        # 查找所有包含Variable的prompt对象
        prompt_objects = self.find_prompt_objects_with_variables()
        
        def extract_from_variable(var, parent_name=""):
            """递归提取所有require_grad=True的变量并转换为textgrad.Variable"""
            result = []
            
            # 检查当前变量是否需要梯度
            if hasattr(var, 'require_grad') and var.require_grad:
                # 获取变量的当前值
                var_value = var.get_value() if hasattr(var, 'get_value') else str(var.variables)
                var_desc = var.description if hasattr(var, 'description') else f"Prompt module: {var.name}"
                
                # 创建textgrad.Variable
                tg_var = tg.Variable(
                    value=var_value,
                    requires_grad=True,
                    role_description=var_desc
                )
                result.append((tg_var, var))  # 保存原始变量以便后续更新
            
            # 递归处理子变量
            if isinstance(var.variables, list):
                for child in var.variables:
                    if hasattr(child, 'require_grad'):
                        result.extend(extract_from_variable(child, f"{parent_name}.{var.name}"))
            elif hasattr(var.variables, 'require_grad'):
                result.extend(extract_from_variable(var.variables, f"{parent_name}.{var.name}"))
            
            return result
        
        # 从所有prompt对象中提取变量
        all_optimizable_var_pairs = []
        prompt_mapping = {}  # tg_var -> prompt_obj
        
        for prompt_obj, prompt_name in prompt_objects:
            if hasattr(prompt_obj, 'prompt'):
                prompt_var = prompt_obj.prompt
                optimizable_var_pairs = extract_from_variable(prompt_var)
                all_optimizable_var_pairs.extend(optimizable_var_pairs)
                
                # 记录每个变量属于哪个prompt对象
                for tg_var, orig_var in optimizable_var_pairs:
                    prompt_mapping[tg_var] = prompt_obj
        
        # 分离textgrad变量和原始变量的映射
        tg_vars = [tg_var for tg_var, _ in all_optimizable_var_pairs]
        var_mapping = {tg_var: orig_var for tg_var, orig_var in all_optimizable_var_pairs}
        
        logger.info(f"| 📊 Found {len(tg_vars)} optimizable variables from {len(prompt_objects)} prompt object(s):")
        for tg_var, orig_var in all_optimizable_var_pairs:
            prompt_obj = prompt_mapping.get(tg_var)
            prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__ if prompt_obj else 'unknown'
            logger.info(f"|   - [{prompt_name}] {orig_var.name if hasattr(orig_var, 'name') else 'unknown'}: {tg_var.role_description}")
        
        return tg_vars, var_mapping, prompt_mapping
    
    def clear_prompt_caches(self, tg_vars: Optional[List[tg.Variable]] = None):
        """
        清除所有包含被优化变量的prompt对象的缓存
        
        Args:
            tg_vars: 要清除缓存的变量列表（如果为None，则清除所有已记录的变量对应的prompt）
        
        重新加载位置说明：
        - 清除缓存后，当Agent调用prompt对象的get_message()方法时（通常在_get_messages()中），
          如果reload=False且message为None，会自动重新渲染prompt（调用prompt.render()），
          此时会使用更新后的变量值
        - 具体位置：
          * ToolCallingAgent._get_messages() -> prompt_manager.get_system_message()
          * SystemPrompt.get_message() -> 如果message为None，会执行prompt.render(modules)
        """
        if tg_vars is None:
            tg_vars = self.optimizable_tg_vars
        
        # 收集所有需要清除缓存的prompt对象（去重）
        prompt_objects_to_clear = set()
        for tg_var in tg_vars:
            if tg_var in self.prompt_mapping:
                prompt_obj = self.prompt_mapping[tg_var]
                prompt_objects_to_clear.add(prompt_obj)
        
        # 清除每个prompt对象的缓存
        for prompt_obj in prompt_objects_to_clear:
            # SystemPrompt 和 AgentMessagePrompt 都有 message 属性
            if hasattr(prompt_obj, 'message'):
                prompt_obj.message = None
                prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__
                logger.debug(f"| 🗑️ Cleared cache for {prompt_name}")
        
        if prompt_objects_to_clear:
            logger.info(f"| 🗑️ Cleared cache for {len(prompt_objects_to_clear)} prompt object(s)")
    
    def define_loss_function(self, agent_result: Any, task: str, max_steps: int) -> tg.TextLoss:
        """
        定义损失函数，基于Agent的执行结果
        
        Args:
            agent_result: Agent的执行结果
            task: 原始任务描述
            max_steps: 最大步数
        
        Returns:
            tg.TextLoss: 损失函数对象
        """
        # 基于任务完成情况的评估指令
        if agent_result and "success" in str(agent_result).lower():
            # 任务成功完成
            eval_instruction = (
                f"The agent successfully completed the task: '{task}'. "
                f"The prompt worked well. Identify any remaining areas for improvement "
                f"to make the prompt even more effective and clear."
            )
        elif agent_result and "error" in str(agent_result).lower():
            # 任务失败
            eval_instruction = (
                f"The agent failed to complete the task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Critically analyze what went wrong and provide feedback on how to improve the prompt "
                f"to help the agent better understand and execute the task."
            )
        else:
            # 部分完成或不确定
            eval_instruction = (
                f"Evaluate the agent's performance on task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Provide critical feedback on how to improve the prompt to make it clearer and more actionable."
            )
        
        # 创建TextLoss对象（textgrad的标准API）
        loss_fn = tg.TextLoss(eval_instruction)
        
        return loss_fn
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3,
        optimizer_model: str = "gpt-4o"
    ):
        """
        使用TextGrad优化Agent的提示词
        
        Args:
            task: 任务描述
            files: 附件文件列表
            optimization_steps: 优化迭代次数
            optimizer_model: 用于优化的模型（字符串或引擎对象）
        """
        # 初始化日志
        self.logger.write("="*70)
        self.logger.write("TextGrad 优化日志")
        self.logger.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.write(f"任务: {task}")
        self.logger.write(f"优化步数: {optimization_steps}")
        self.logger.write("="*70)
        self.logger.write("")
        
        try:
            # 1. 设置TextGrad的反向传播引擎
            optimizer_engine = tg.get_engine(optimizer_model)
            tg.set_backward_engine(optimizer_engine, override=True)
            
            # 2. 提取可优化的变量（从所有prompt对象中）
            self.optimizable_tg_vars, self.var_mapping, self.prompt_mapping = self.extract_optimizable_variables()
            
            if not self.optimizable_tg_vars:
                logger.warning("| ⚠️ No optimizable variables found. Skipping optimization.")
                self.logger.write("⚠️ 未找到可优化的变量，跳过优化")
                return
            
            # 3. 创建优化器（textgrad的标准API）
            # 使用更明确的标签，并添加更多约束来提高格式遵循率
            optimizer = tg.TextualGradientDescent(
                parameters=self.optimizable_tg_vars,
                engine=optimizer_engine,
                verbose=1,
                constraints=[
                    "Keep the prompt clear and actionable",
                    "Maintain compatibility with the existing tool calling framework",
                    "Do not change the core agent identity",
                    "The prompt should work with the tool calling agent architecture",
                    "You MUST respond with ONLY the improved text between <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags",
                    "Do not include any explanation, feedback, or other text outside the tags"
                ],
                new_variable_tags=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]  # 明确指定标签
            )
            
            logger.info(f"| 🔄 Starting TextGrad optimization with {optimization_steps} steps...")
            self.logger.write(f"\n🔄 开始TextGrad优化，共 {optimization_steps} 步\n")
            
            # 4. 迭代优化
            for opt_step in range(optimization_steps):
                logger.info(f"\n| {'='*60}")
                logger.info(f"| Optimization Step {opt_step + 1}/{optimization_steps}")
                logger.info(f"| {'='*60}\n")
                
                self.logger.write(f"\n{'='*60}")
                self.logger.write(f"优化步骤 {opt_step + 1}/{optimization_steps}")
                self.logger.write(f"{'='*60}\n")
                
                # 4.1 同步textgrad变量到原始变量（如果有优化过的值）
                if opt_step > 0:
                    for tg_var, orig_var in self.var_mapping.items():
                        if hasattr(orig_var, 'variables'):
                            orig_var.variables = tg_var.value
                    
                    # 清除所有相关prompt对象的缓存，确保使用更新后的变量值
                    # 重新加载会在Agent下次调用get_message()时自动发生（见clear_prompt_caches注释）
                    self.clear_prompt_caches()
                
                # 4.2 使用当前提示词运行Agent
                logger.info(f"| 🚀 Running agent with current prompts...")
                self.logger.write("🚀 使用当前提示词运行Agent...")
                agent_result = await self.agent.ainvoke(task=task, files=files)
                logger.info(f"| 📋 Agent result: {str(agent_result)[:200]}...")
                self.logger.write(f"📋 Agent结果: {str(agent_result)[:500]}...\n")
                
                # 4.3 基于执行结果定义损失函数
                loss_fn = self.define_loss_function(agent_result, task, self.agent.max_steps)
                
                # 4.4 计算损失并进行反向传播
                logger.info(f"| 📉 Computing loss and gradients...")
                self.logger.write("📉 计算损失和梯度...")
                
                # 创建评估用的response变量（代表agent的输出）
                response_var = tg.Variable(
                    value=str(agent_result)[:1000],  # 限制长度
                    requires_grad=True,
                    role_description="Agent execution result"
                )
                
                # 计算损失
                loss = loss_fn(response_var)
                logger.info(f"| 📊 Loss: {loss.value[:200]}...")
                self.logger.write(f"📊 Loss: {loss.value[:500]}...\n")
                
                # 手动将损失反馈添加到提示词变量（因为它们是独立的，没有计算图连接）
                # 我们需要手动为每个可优化变量创建梯度
                loss_feedback = loss.value  # 获取损失值作为反馈
                
                for tg_var in self.optimizable_tg_vars:
                    # 创建梯度变量
                    gradient_var = tg.Variable(
                        value=loss_feedback,
                        requires_grad=False,
                        role_description=f"Feedback for {tg_var.role_description} based on agent performance"
                    )
                    tg_var.gradients.add(gradient_var)
                    logger.info(f"| 📈 Added gradient for {tg_var.role_description[:50]}...")
                
                # 4.5 执行优化步骤（更新提示词）
                logger.info(f"| ✨ Updating prompts with TextGrad...")
                self.logger.write("✨ 使用TextGrad更新提示词...")
                try:
                    optimizer.step()
                except IndexError as e:
                    logger.error(f"| ❌ Optimizer step failed: {e}")
                    self.logger.write(f"❌ 优化步骤失败: {e}")
                    logger.warning(f"| ⚠️ LLM response may not have followed the required format. Trying with a stronger model or retry...")
                    # 可以在这里添加重试逻辑或使用更强的模型
                    raise
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                self.logger.write(f"✅ 优化步骤 {opt_step + 1} 完成\n")
                
                # 4.6 同步优化后的值回原始变量
                for tg_var in self.optimizable_tg_vars:
                    if tg_var in self.var_mapping:
                        orig_var = self.var_mapping[tg_var]
                        if hasattr(orig_var, 'variables'):
                            orig_var.variables = tg_var.value
                
                # 清除所有相关prompt对象的缓存，确保下次调用时使用更新后的变量值
                # 重新加载会在Agent下次调用get_message()时自动发生（见clear_prompt_caches注释）
                self.clear_prompt_caches()
            
            logger.info(f"| 🎉 Optimization completed!")
            self.logger.write("\n🎉 优化完成!")
            
            # 5. 输出最终优化变量摘要
            logger.info(f"| 📊 Final optimized variables (摘要):")
            self.logger.write("\n📊 最终优化变量摘要:")
            for tg_var in self.optimizable_tg_vars:
                logger.info(f"|   - {tg_var.role_description[:60]}: {tg_var.value[:150]}...")
                self.logger.write(f"   - {tg_var.role_description[:60]}: {tg_var.value[:200]}...")
        
        finally:
            # 关闭优化日志文件
            self.logger.write(f"\n{'='*70}")
            self.logger.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.write(f"{'='*70}")
            self.logger.close()
            logger.info(f"| 📝 Optimization log saved and closed")
    
    def get_optimized_variables(self) -> List[tg.Variable]:
        """
        获取优化后的变量列表
        
        Returns:
            List[tg.Variable]: 优化后的变量列表
        """
        return self.optimizable_tg_vars


# 便捷函数：为了保持向后兼容性
async def optimize_agent_with_textgrad(
    agent,
    task: str,
    files: Optional[List[str]] = None,
    optimization_steps: int = 3,
    optimizer_model: str = "gpt-4o",
    log_dir: Optional[str] = None
):
    """
    使用TextGrad优化Agent的提示词（便捷函数）
    
    Args:
        agent: Agent实例
        task: 任务描述
        files: 附件文件列表
        optimization_steps: 优化迭代次数
        optimizer_model: 用于优化的模型（字符串或引擎对象）
        log_dir: 日志目录路径（如果为None，则使用agent的workdir）
    """
    if log_dir is None:
        # 尝试从agent获取workdir
        if hasattr(agent, 'workdir'):
            log_dir = agent.workdir
        else:
            log_dir = "workdir/optimization"
    
    optimizer = TextGradOptimizer(agent, log_dir)
    await optimizer.optimize(task, files, optimization_steps, optimizer_model)
    return optimizer

