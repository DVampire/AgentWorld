import asyncio
import sys
import os

# --- 1. 环境配置 ---
# 确保项目根目录在 path 中
sys.path.append(os.getcwd())

from src.benchmark.server import BenchmarkFactory
# 导入你的模型组件
from src.model.manager import model_manager
from src.message.types import HumanMessage, SystemMessage

# --- 2. 评测配置 ---
BENCHMARK_CONFIG = {
    "name": "leetcode",
    "path": "./datasets/leetcode",
    "split": "test",
    "subset": None
}

# 目标模型
TARGET_MODEL = "openrouter/gemini-3-pro-preview"

async def run_benchmark_loop():
    print(f"🚀 开始评测流程 | Target: {TARGET_MODEL} | Dataset: {BENCHMARK_CONFIG['name']}")

    # ------------------------------------------------------------------
    # Step 1: 初始化组件
    # ------------------------------------------------------------------
    print("\n[Init] 初始化 ModelManager...")
    try:
        await model_manager.initialize()
    except Exception as e:
        print(f"❌ ModelManager 初始化失败: {e}")
        return

    print(f"[Init] 创建并加载 Benchmark ({BENCHMARK_CONFIG['name']})...")
    # 使用工厂创建实例
    benchmark = BenchmarkFactory.create(
        name=BENCHMARK_CONFIG["name"],
        path=BENCHMARK_CONFIG["path"],
        split=BENCHMARK_CONFIG["split"],
        subset=BENCHMARK_CONFIG["subset"]
    )
    
    # 异步加载数据 (构建 Ground Truth 索引)
    await benchmark.load()
    
    # 确保从头开始
    benchmark.reset()
    
    total_tasks = benchmark.progress[1]
    print(f"✅ 准备就绪，共 {total_tasks} 个任务。开始流式推理...\n")

    # ------------------------------------------------------------------
    # Step 2: 步进推理循环 (One-by-One Loop)
    # ------------------------------------------------------------------
    while True:
        # 1. 获取下一个任务 (Stateful Step)
        task = benchmark.step()
        
        # 如果 task 为 None，说明遍历结束
        if task is None:
            break
            
        task_id = task["task_id"]
        # 兼容不同数据集字段名 (input/question/problem)
        question_text = task.get("input") or task.get("question") or task.get("problem")
        system_prompt_text = task.get("system_prompt", "You are a helpful assistant.")
        
        # 打印当前进度
        curr_idx, _ = benchmark.progress
        print(f"Processing Task [{curr_idx}/{total_tasks}] (ID: {task_id}) ...")

        # 2. 构造消息 (转换格式)
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=question_text)
        ]
        print(messages)
        # 3. 执行真实推理
        prediction_content = ""
        try:
            # 调用你的 ModelManager
            response = await model_manager(
                model=TARGET_MODEL,
                messages=messages,
                temperature=0.7, # AIME 等数学题建议较低温度
                stream=False
            )
            
            if response.success:
                prediction_content = response.message
                # 打印预览 (可选)
                # print(f"  -> Model Output (first 100 chars): {prediction_content[:100]}...")
            else:
                print(f"  ⚠️ Model API Error: {response.message}")
                prediction_content = "" # 标记为空，稍后评估为 0 分
                
        except Exception as e:
            print(f"  ❌ Critical Inference Error: {e}")
            prediction_content = ""

        # 4. 评估并记录状态
        # benchmark.eval_task 会自动查找 GT，计算分数，并保存到内部 _results 列表
        score = await benchmark.eval_task(prediction=prediction_content, task_id=task_id)
        
        # 实时反馈
        icon = "✅" if score == 1.0 else "❌"
        print(f"  -> {icon} Score: {score}")

    # ------------------------------------------------------------------
    # Step 3: 最终报告
    # ------------------------------------------------------------------
    print("\n" + "="*40)
    stats = benchmark.get_stats()
    print("📊 Final Benchmark Report")
    print("-" * 40)
    print(f"Model: {TARGET_MODEL}")
    print(f"Benchmark: {BENCHMARK_CONFIG['name']}")
    print(f"Total Attempted: {stats['total_attempted']}")
    print(f"Correct: {stats['correct_count']}")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    print("="*40)
    
    # (可选) 如果你想获取详细数据保存为 JSONL
    # all_results = benchmark.get_results()
    # import json
    # with open("benchmark_results.jsonl", "w") as f:
    #     for res in all_results:
    #         f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_benchmark_loop())
    except KeyboardInterrupt:
        print("\n\n🛑 用户手动停止测试。")