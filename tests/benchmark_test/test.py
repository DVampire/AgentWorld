import asyncio
import sys
import os

# --- 导入必要的模块 ---
sys.path.append(os.getcwd())

from src.benchmark.server import BenchmarkManager
from src.benchmark.aime import AIMEBenchmark 

# ✅ 新增：导入模型管理器和消息类型
# 假设你之前的 ModelManager 代码保存在 src/model/manager.py 中
# 如果你的文件路径不同，请修改这里的导入路径
from src.model.manager import model_manager 
from src.message.types import HumanMessage, SystemMessage

# ================= 配置区域 =================
BENCHMARK_NAME = "aime24"
SUBSET = "" 
DATA_PATH = "./datasets/AIME24"
SPLIT = "test"
# ✅ 指定模型名称
TARGET_MODEL = "openrouter/gemini-3-pro-preview"
# ===========================================

async def test_benchmark_flow():
    print(f"🚀 开始测试 BenchmarkManager (Real Inference Mode): {BENCHMARK_NAME}")
    
    # ---------------------------------------------------------
    # 1. 初始化 ModelManager (新增步骤)
    # ---------------------------------------------------------
    print("\n[Step 0] 初始化模型管理器...")
    try:
        await model_manager.initialize()
        print("✅ ModelManager 初始化完成。")
    except Exception as e:
        print(f"❌ ModelManager 初始化失败: {e}")
        return

    # ---------------------------------------------------------
    # 2. 实例化评估逻辑类
    # ---------------------------------------------------------
    print("\n[Step 1] 实例化 AIMEBenchmark 类...")
    aime_impl = AIMEBenchmark(
        benchmark_name=BENCHMARK_NAME,
        path=DATA_PATH,
        split=SPLIT,
        subset=SUBSET
    )
    
    # ---------------------------------------------------------
    # 3. 初始化 Manager 并注入评估函数
    # ---------------------------------------------------------
    print("[Step 2] 初始化 Manager 并注入 aime_impl.eval ...")
    manager = BenchmarkManager(
        benchmark_name=BENCHMARK_NAME,
        path=DATA_PATH,
        split=SPLIT,
        subset=SUBSET,
        evaluate_fn=aime_impl.eval 
    )

    # ---------------------------------------------------------
    # 4. 加载数据
    # ---------------------------------------------------------
    print("\n[Step 3] 正在异步加载数据集...")
    await manager.load()
    df = manager.get_data()
    print(f"📊 数据量: {len(df)} 条")
    
    if len(df) == 0: return

    # ---------------------------------------------------------
    # 5. 真实模型推理 (修改的核心部分)
    # ---------------------------------------------------------
    print("\n[Step 4] 开始真实模型推理...")
    
    # 取第一条数据测试
    item = df.iloc[0]
    task_id = item.get('task_id', 'Unknown')
    
    # ✅ 尝试获取问题文本，AIME 数据集通常字段是 'problem' 或 'question'
    problem_text = item.get('problem', item.get('question', None))
    if not problem_text:
        print("❌ 无法在数据中找到 'problem' 或 'question' 字段")
        return

    real_answer = manager._ground_truth_map.get(str(task_id))
    print(f"👉 Case ID: {task_id}")
    print(f"📚 问题预览: {problem_text[:100]}...")
    print(f"🎯 预期答案 (Ground Truth): {real_answer}")

    # ✅ 构造 Prompt
    # 关键：AIME 评测通常依赖正则表达式提取 \boxed{答案}，所以 System Prompt 必须强调格式
    system_instruction = (
        "You are an expert mathematician participating in the AIME competition. "
        "Solve the following problem carefully. "
        "IMPORTANT: You must put your final answer inside a LaTeX box, like this: \\boxed{answer}. "
        "Do not output 'Answer: \\boxed{...}', just output the boxed value at the end of your reasoning."
    )

    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=problem_text)
    ]

    print(f"🧠 正在调用模型: {TARGET_MODEL} ...")
    
    # ✅ 调用 model_manager 进行推理
    try:
        response = await model_manager(
            model=TARGET_MODEL,
            messages=messages,
            temperature=0.7, # 数学题通常可以使用稍低的温度，或者用默认
            stream=False 
        )
    except Exception as e:
        print(f"❌ 模型调用异常: {e}")
        return

    if not response.success:
        print(f"❌ 模型 API 返回错误: {response.message}")
        return

    prediction_content = response.message
    print(f"📝 模型实际回复 (前200字符):\n{prediction_content[:200]}...")
    print("-" * 40)

    # ---------------------------------------------------------
    # 6. 评估真实结果
    # ---------------------------------------------------------
    print("⏳ 正在调用 manager.eval (使用真实推理结果)...")
    
    # 使用模型生成的文本进行评估
    score = await manager.eval(prediction=prediction_content, task_id=task_id)
    
    print("-" * 40)
    if score == 1.0:
        print(f"🎉 恭喜! 模型回答正确 (Score: {score})")
        print(f"    预期: {real_answer}")
        print(f"    提取: (逻辑在 AIMEBenchmark 内部)")
    else:
        print(f"💔 模型回答错误 (Score: {score})")
        print(f"    预期: {real_answer}")
        print("    可能原因: 计算错误 或 格式未包含 \\boxed{}")

if __name__ == "__main__":
    asyncio.run(test_benchmark_flow())