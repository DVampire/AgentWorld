"""
简化的离线回测运行脚本 - 快速开始

用法：
    conda activate agentworld
    cd tests/hl_backtest
    python run_backtest_simple.py
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# 添加项目根目录到路径
root_dir = Path(__file__).resolve().parents[1]  # tests/ 的父目录是项目根目录
sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
load_dotenv(verbose=True)

from argparse import Namespace
from src.config import config
from src.logger import logger
from src.agents.protocol import agent_manager
from src.agents.offline_trading_agent import OfflineTradingAgent  # 导入以触发装饰器注册
from src.tools.protocol import tool_manager
from src.environments.protocol import environment_manager
from src.transformation import transformation
from src.models.model_manager import model_manager
from src.environments.backtest_hyperliquid_environment import BacktestHyperliquidEnvironment


async def main():
    """主函数 - 简化的回测运行"""
    
    print("=" * 80)
    print("🚀 LLM 离线回测系统")
    print("=" * 80)
    
    # ============ 配置参数 ============
    # 数据库路径：在 tests/hl_backtest/ 目录下
    db_path = Path(__file__).parent / "hl_backtest" / "database.db"  # 数据库路径
    symbol = "BTC"  # 交易币种
    initial_equity = 1000.0  # 初始资金
    max_leverage = 5.0  # 最大杠杆
    taker_fee_rate = 0.0005  # 手续费率参数（已废弃，实际使用固定0.6块手续费）
    fixed_fee = 0.6  # 固定手续费（开仓和平仓都收取0.6块）
    slippage_bps = 1.0  # 滑点 (0.01%)
    
    # 回测参数（测试模式：减少LLM调用次数）
    steps_per_candle = 1  # 每个K线允许LLM决策的次数
    max_total_steps = 10  # Agent最大步数（每个K线）- 减少以加快测试
    
    # 限制回测的K线数量（测试模式：只处理少量K线）
    max_bars = 10  # 最多处理10根K线（测试用）
    
    # ============ 初始化配置 ============
    print("\n" + "=" * 80)
    print("初始化配置...")
    print("=" * 80)
    
    # 加载配置文件（与 online_trading_agent 完全一致）
    default_config = root_dir / "configs" / "online_trading_agent.py"
    if default_config.exists():
        # config.init_config 需要两个参数：config_path 和 args (Namespace对象)
        # 创建一个空的 Namespace 对象
        empty_args = Namespace()
        empty_args.cfg_options = None
        config.init_config(str(default_config), empty_args)
        logger.init_logger(config)
        logger.info(f"| ✅ 配置加载成功")
    else:
        logger.warning(f"| ⚠️  配置文件不存在，使用默认配置")
        logger.init_logger(config)
    
    # 从配置中读取配置（与 online_trading_agent 一致）
    # 使用 GPT-5 模型（与 online_trading_agent 完全一致）
    model_name = config.online_trading_agent.get('model_name', 'gpt-5') if hasattr(config, 'online_trading_agent') else 'gpt-5'
    prompt_name = config.online_trading_agent.get('prompt_name', 'online_trading') if hasattr(config, 'online_trading_agent') else 'online_trading'
    
    # 工作目录
    workdir = root_dir / "workdir" / "llm_backtest"
    workdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 回测配置:")
    print(f"  数据库: {db_path}")
    print(f"  币种: {symbol}")
    print(f"  初始资金: ${initial_equity:,.2f}")
    print(f"  最大杠杆: {max_leverage}x")
    print(f"  手续费: 固定 ${fixed_fee:.2f}/笔（开仓和平仓都收取）")
    print(f"  滑点: {slippage_bps} bps")
    print(f"  每个K线决策次数: {steps_per_candle}")
    print(f"  Agent最大步数: {max_total_steps}")
    print(f"  最多处理K线数: {max_bars}")
    print(f"  模型: {model_name} (与 online_trading_agent 一致)")
    print(f"  工作目录: {workdir}")
    
    # ============ 初始化回测环境 ============
    print("\n" + "=" * 80)
    print("初始化回测环境...")
    print("=" * 80)
    
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        return
    
    backtest_env = BacktestHyperliquidEnvironment(
        db_path=str(db_path),
        symbol=symbol,
        initial_equity=initial_equity,
        max_leverage=max_leverage,
        taker_fee_rate=taker_fee_rate,
        slippage_bps=slippage_bps
    )
    
    await backtest_env.initialize()
    
    # 注册环境到 ECP（重要：Agent 需要通过 ECP 访问环境）
    # 环境信息应该已经通过装饰器注册到 _registered_environments
    # 现在需要将实例注册到 context manager
    
    # 先检查环境是否已注册（装饰器自动注册）
    logger.info(f"| 🔍 查找环境: {backtest_env.name}")
    logger.info(f"| 🔍 已注册的环境: {list(environment_manager._registered_environments.keys())}")
    
    # 从 _registered_environments 获取环境信息（装饰器注册在这里）
    env_info = None
    if backtest_env.name in environment_manager._registered_environments:
        env_info = environment_manager._registered_environments[backtest_env.name]
        logger.info(f"| ✅ 找到环境信息: {backtest_env.name}")
    else:
        # 如果装饰器没有注册（可能是名称冲突或其他原因），尝试使用已有环境或手动创建
        logger.warning(f"| ⚠️  环境 {backtest_env.name} 未在 _registered_environments 中找到")
        # 尝试使用 build 方法注册实例
        logger.info(f"| 🔧 尝试使用 build 方法注册实例...")
        try:
            def env_factory():
                return backtest_env  # 返回已初始化的实例
            
            # 需要先创建 EnvironmentInfo，但由于装饰器应该已经做了，这里先用已有信息
            # 如果有同名环境，先用那个环境的信息
            existing_env = environment_manager._registered_environments.get("hyperliquid")
            if existing_env:
                env_info = existing_env
                logger.info(f"| ✅ 使用已存在的环境信息: {backtest_env.name}")
            else:
                raise ValueError(f"Environment {backtest_env.name} not found and cannot create")
        except Exception as e:
            logger.error(f"| ❌ 无法创建环境信息: {e}")
            raise
    
    # 设置实例到环境信息并注册到 context manager
    if env_info:
        env_info.instance = backtest_env
        environment_manager.environment_context_manager._environment_info[backtest_env.name] = env_info
        logger.info(f"| ✅ 回测环境已注册到 context manager: {backtest_env.name}")
    else:
        raise ValueError(f"Failed to register environment {backtest_env.name}")
    
    # 限制回测的K线数量
    if len(backtest_env.historical_data) > max_bars:
        logger.info(f"| ⚠️  数据量较大（{len(backtest_env.historical_data)}根），限制为前{max_bars}根")
        backtest_env.historical_data = backtest_env.historical_data.iloc[:max_bars].reset_index(drop=True)
    
    # ============ 初始化模型管理器 ============
    print("\n" + "=" * 80)
    print("初始化模型管理器...")
    print("=" * 80)
    
    use_local_proxy = config.use_local_proxy if hasattr(config, 'use_local_proxy') else False
    await model_manager.initialize(use_local_proxy=use_local_proxy)
    logger.info(f"| ✅ 模型管理器初始化完成: {model_manager.list()}")
    
    # ============ 初始化工具管理器 ============
    print("\n" + "=" * 80)
    print("初始化工具管理器...")
    print("=" * 80)
    
    tool_names = config.tool_names if hasattr(config, 'tool_names') else ['done']
    await tool_manager.initialize(tool_names)
    logger.info(f"| ✅ 工具管理器初始化完成: {tool_manager.list()}")
    
    # ============ 转换环境为工具（重要：与online trading保持一致）============
    print("\n" + "=" * 80)
    print("转换环境为工具（ECP -> TCP）...")
    print("=" * 80)
    
    # 重要：将环境（ECP）转换为工具（TCP），这样Agent才能通过工具调用环境方法
    # 这与 online trading agent 的设置完全一致
    logger.info("| 🔄 Transformation start (E2T)...")
    env_names = [backtest_env.name]  # 使用回测环境的名称
    await transformation.transform(type="e2t", env_names=env_names)
    logger.info(f"| ✅ Transformation completed: {tool_manager.list()}")
    
    # ============ 初始化 LLM Agent ============
    print("\n" + "=" * 80)
    print("初始化 LLM Agent...")
    print("=" * 80)
    
    # 配置 memory_config（与 online_trading_agent 完全一致，但 model_name 使用 deepseek）
    if hasattr(config, 'memory_config'):
        # 从配置中读取，但确保是字典类型
        memory_config = dict(config.memory_config) if isinstance(config.memory_config, dict) else dict(config.memory_config)
    else:
        memory_config = {
            "type": "online_trading_memory_system",
            "model_name": model_name,
            "max_summaries": 20,
            "max_insights": 100
        }
    # memory_config 中的 model_name 使用与 agent 相同的模型
    memory_config["model_name"] = model_name
    
    # 创建 offline_trading_agent 配置（基于 online_trading_agent，但使用 offline_trading agent）
    # 确保配置对象存在
    if not hasattr(config, 'online_trading_agent'):
        config.online_trading_agent = {}
    
    # 从配置中读取 agent 配置（与 online_trading_agent 完全一致）
    agent_config = dict(config.online_trading_agent) if isinstance(config.online_trading_agent, dict) else {}
    
    # 重要：先移除 name 字段（如果存在），然后设置为 offline_trading
    # 这样可以确保 agent 实例使用正确的名称
    if "name" in agent_config:
        del agent_config["name"]
    
    # 使用与 online_trading_agent 相同的模型配置
    agent_config["model_name"] = model_name
    agent_config["workdir"] = str(workdir)
    agent_config["memory_config"] = memory_config
    # 重要：确保 name 字段是 offline_trading，否则 agent 实例名称会错误
    agent_config["name"] = "offline_trading"  # 必须设置为 offline_trading
    
    # 覆盖回测特定的配置（如果需要）
    if max_total_steps > 0:
        agent_config["max_steps"] = max_total_steps
    
    # 创建 offline_trading_agent 配置（用于注册 offline_trading agent）
    # 确保 workdir 在配置中（虽然已经设置过了，但这里确保一下）
    agent_config["workdir"] = str(workdir)
    # 再次确保 name 字段正确（防止被覆盖）
    agent_config["name"] = "offline_trading"
    
    if not hasattr(config, 'offline_trading_agent'):
        config.offline_trading_agent = {}
    config.offline_trading_agent.update(agent_config)
    
    # 最后再次确保 config.offline_trading_agent 中的 name 字段正确
    config.offline_trading_agent["name"] = "offline_trading"
    
    # 确保 config 对象有 get 方法（如果使用 mmengine.config）
    if not hasattr(config, 'get'):
        def config_get(key, default=None):
            attr_name = key
            if hasattr(config, attr_name):
                value = getattr(config, attr_name)
                # 如果值是空字典，也返回它（不是 None）
                if value is not None:
                    return value
            return default
        config.get = config_get
    
    # 验证配置是否正确设置
    test_config = config.get('offline_trading_agent', None)
    logger.info(f"| 🔍 验证配置: config.get('offline_trading_agent') = {test_config}")
    if test_config is None:
        logger.warning(f"| ⚠️ 配置未找到，直接设置到 config 对象")
        # 如果 get 方法还是返回 None，直接设置属性
        config.offline_trading_agent = agent_config
        logger.info(f"| ✅ 直接设置 config.offline_trading_agent = {agent_config}")
    
    # 最后再次确保 config.offline_trading_agent 中的 name 字段正确（防止被覆盖）
    if hasattr(config, 'offline_trading_agent'):
        config.offline_trading_agent["name"] = "offline_trading"
        logger.info(f"| ✅ 最终确认: config.offline_trading_agent['name'] = {config.offline_trading_agent.get('name', 'NOT SET')}")
    
    # 使用 ACP 初始化 Agent（使用 offline_trading agent）
    logger.info("| 🤖 Initializing agents...")
    # 直接传入 agent_names 列表，不依赖 config.agent_names
    agent_names_to_init = ["offline_trading"]
    logger.info(f"| 🔍 准备初始化 agents: {agent_names_to_init}")
    logger.info(f"| 🔍 config.offline_trading_agent 存在: {hasattr(config, 'offline_trading_agent')}")
    if hasattr(config, 'offline_trading_agent'):
        logger.info(f"| 🔍 config.offline_trading_agent 内容: {config.offline_trading_agent}")
    await agent_manager.initialize(agent_names_to_init)
    logger.info(f"| ✅ Agents initialized: {agent_manager.list()}")
    
    # 获取 Agent 实例（使用 offline_trading agent）
    agent_info = agent_manager.get_info("offline_trading")
    if agent_info is None:
        raise ValueError(f"Agent 'offline_trading' not found. Available agents: {agent_manager.list()}")
    agent = agent_info.instance
    
    logger.info(f"| ✅ Agent 初始化完成: {agent.name}")
    logger.info(f"| 🧠 模型: {model_name}")
    
    # ============ 开始回测 ============
    print("\n" + "=" * 80)
    print("开始回测...")
    print("=" * 80)
    
    total_bars = len(backtest_env.historical_data)
    current_bar = 0
    total_agent_steps = 0
    backtest_log = []
    
    try:
        # 从第5根K线开始（保证有足够的历史数据，测试模式用更少）
        start_bar = min(5, total_bars - 1)
        current_bar = start_bar
        
        while current_bar < total_bars:
            # 推进到当前K线
            await backtest_env._advance_to_index(current_bar)
            
            current_time = backtest_env.current_time
            current_price = backtest_env.current_price
            
            print(f"\n{'='*80}")
            print(f"K线 {current_bar + 1}/{total_bars}")
            print(f"时间: {current_time}")
            print(f"价格: ${current_price:,.2f}")
            print(f"权益: ${backtest_env.equity:,.2f}")
            print(f"{'='*80}")
            
            # 每个K线允许LLM决策 steps_per_candle 次
            for step in range(steps_per_candle):
                # 构造任务提示（使用英文，与 online_trading_agent 一致）
                task = f"Start trading on {symbol} and maximize the profit until the environment is done."
                
                # 运行 LLM Agent 进行一轮决策
                logger.info(f"| 🔄 Agent 决策轮次 {step + 1}/{steps_per_candle} (总步数: {total_agent_steps + 1})")
                
                try:
                    # 使用 offline_trading agent（已内置 DeepSeek 格式处理）
                    result = await agent.ainvoke(task)
                    total_agent_steps += 1
                    
                    # 记录日志
                    backtest_log.append({
                        "bar": current_bar,
                        "step": step,
                        "time": current_time.isoformat() if current_time else None,
                        "price": current_price,
                        "equity": backtest_env.equity,
                        "agent_result": str(result)[:500] if result else None,
                        "total_agent_steps": total_agent_steps
                    })
                    
                    # 如果 Agent 调用 done，结束本轮决策
                    if result and "done" in str(result).lower():
                        logger.info(f"| ✅ Agent 决定结束本轮决策")
                        break
                    
                except Exception as e:
                    logger.error(f"| ❌ Agent 执行错误: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 即使出错也继续
                    if "max steps" in str(e).lower():
                        logger.warning(f"| ⚠️  达到最大步数限制，继续下一个K线")
                        break
            
            # 推进到下一个K线
            current_bar += 1
            
            # 提前终止条件
            if backtest_env.equity <= 0:
                logger.warning(f"| ⚠️  账户权益为0，回测终止")
                break
            
            # 显示进度
            if current_bar % 10 == 0:
                progress = (current_bar / total_bars) * 100
                logger.info(f"| 📊 进度: {current_bar}/{total_bars} ({progress:.1f}%)")
    
    except KeyboardInterrupt:
        logger.warning(f"| ⚠️  用户中断回测")
    
    finally:
        # ============ 清理和保存结果 ============
        print("\n" + "=" * 80)
        print("保存回测结果...")
        print("=" * 80)
        
        await backtest_env.cleanup()
        # Agent 不需要 cleanup 方法，context manager 会自动清理
        
        # 获取回测结果
        results = backtest_env.get_backtest_results()
        
        # 显示结果
        print("\n" + "=" * 80)
        print("📊 回测结果")
        print("=" * 80)
        
        print(f"\n初始资金: ${results.get('initial_equity', 0):,.2f}")
        print(f"最终权益: ${results.get('final_equity', 0):,.2f}")
        print(f"总收益: ${results.get('final_equity', 0) - results.get('initial_equity', 0):,.2f}")
        print(f"收益率: {results.get('total_return', 0):.2f}%")
        print(f"最大回撤: {results.get('max_drawdown', 0):.2f}%")
        print(f"总交易次数: {results.get('total_trades', 0)}")
        print(f"总订单数: {results.get('total_orders', 0)}")
        print(f"处理的K线数: {current_bar}/{total_bars}")
        print(f"Agent总步数: {total_agent_steps}")
        
        # 保存结果
        results_dir = workdir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存权益曲线
        if results.get('equity_curve'):
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_file = results_dir / f"equity_curve_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            print(f"\n✅ 权益曲线已保存: {equity_file}")
        
        # 保存交易历史
        if results.get('trades_history'):
            trades_df = pd.DataFrame(results['trades_history'])
            trades_file = results_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ 交易历史已保存: {trades_file}")
        
        # 保存回测日志
        log_file = results_dir / f"backtest_log_{timestamp}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_log, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 回测日志已保存: {log_file}")
        
        # 保存结果摘要
        summary = {
            "initial_equity": results.get('initial_equity'),
            "final_equity": results.get('final_equity'),
            "total_return": results.get('total_return'),
            "max_drawdown": results.get('max_drawdown'),
            "total_trades": results.get('total_trades'),
            "total_orders": results.get('total_orders'),
            "bars_processed": current_bar,
            "total_bars": total_bars,
            "total_agent_steps": total_agent_steps,
            "config": {
                "symbol": symbol,
                "initial_equity": initial_equity,
                "max_leverage": max_leverage,
                "fixed_fee": fixed_fee,
                "taker_fee_rate": taker_fee_rate,  # 已废弃，保留用于兼容性
                "slippage_bps": slippage_bps
            }
        }
        summary_file = results_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 结果摘要已保存: {summary_file}")
        
        print("\n" + "=" * 80)
        print("✅ 回测完成！")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

