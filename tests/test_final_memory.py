#!/usr/bin/env python3
"""Test script for LLM-controlled session and task ID design."""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.memory import MemoryManager, InMemoryStore, ChatEvent, EventType
from src.logger import logger


async def test_llm_controlled_memory():
    """Test the LLM-controlled session and task ID design."""
    
    print("🧠 Testing LLM-Controlled Memory Design")
    print("=" * 50)
    
    # Create memory manager
    memory_manager = MemoryManager(InMemoryStore())
    
    # Simulate LLM determining session and task IDs based on user input
    print("\n📝 Test 1: LLM determines session and task IDs")
    
    # User asks for programming help
    user_task1 = "帮我写一个Python排序函数"
    
    # LLM determines meaningful session and task IDs
    session_id = "programming_help_session_2024_01_15"
    task_id1 = "write_sorting_function_task"
    
    print(f"User task: {user_task1}")
    print(f"LLM determined session_id: {session_id}")
    print(f"LLM determined task_id: {task_id1}")
    
    # Create session and create first task
    memory_manager.create_session(session_id)
    memory_manager.create_task(user_task1, task_id1)
    
    # Add events for this task (note: create_task already added the human message)
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "我来帮你写一个Python排序函数。", agent_name="coding_assistant")
    memory_manager.add_event(EventType.TOOL_CALL, "Calling tool: code_generator", {"tool_name": "code_generator"}, "coding_assistant")
    memory_manager.add_event(EventType.TOOL_RESULT, "✅ code_generator: 生成了排序函数代码", {"tool_name": "code_generator", "success": True}, "coding_assistant")
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "这是你的排序函数：\n```python\ndef bubble_sort(arr):\n    # 实现代码\n```", agent_name="coding_assistant")
    
    # User asks a follow-up question (same session, new task)
    print("\n📝 Test 2: User asks follow-up question (same session, new task)")
    
    user_task2 = "这个函数怎么优化？"
    
    # LLM determines new task ID for the same session
    task_id2 = "optimize_sorting_function_task"
    
    print(f"User task: {user_task2}")
    print(f"LLM determined task_id: {task_id2}")
    print(f"Session remains: {session_id}")
    
    # Create new task in same session
    memory_manager.create_task(user_task2, task_id2)
    
    # Add events for this new task (note: create_task already added the human message)
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "让我分析一下如何优化这个排序函数。", agent_name="coding_assistant")
    memory_manager.add_event(EventType.TOOL_CALL, "Calling tool: code_analyzer", {"tool_name": "code_analyzer"}, "coding_assistant")
    memory_manager.add_event(EventType.TOOL_RESULT, "✅ code_analyzer: 分析完成，建议使用快速排序", {"tool_name": "code_analyzer", "success": True}, "coding_assistant")
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "建议使用快速排序算法来优化性能。", agent_name="coding_assistant")
    
    # Before switching session, show current session stats and task breakdown
    print("\n📊 Session 1 summary (programming)")
    s1_events_all = memory_manager.get_events()
    s1_task1_events = memory_manager.get_events(task_id=task_id1)
    s1_task2_events = memory_manager.get_events(task_id=task_id2)
    print(f"Session 1 total events: {len(s1_events_all)}")
    print(f"Task 1 events: {len(s1_task1_events)} | Task 2 events: {len(s1_task2_events)}")

    # User starts a completely new topic (same session, new task)
    print("\n📝 Test 3: User starts new topic (same session, new task)")
    
    user_task3 = "帮我写一个英语学习计划"
    task_id3 = "create_study_plan_task"
    
    print(f"User task: {user_task3}")
    print(f"LLM determined task_id: {task_id3}")
    print(f"Session remains: {session_id}")
    
    # Create new task in same session
    memory_manager.create_task(user_task3, task_id3)
    
    # Add events for this new session (note: create_task already added the human message)
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "我来帮你制定一个英语学习计划。", agent_name="study_planner")
    memory_manager.add_event(EventType.TOOL_CALL, "Calling tool: plan_generator", {"tool_name": "plan_generator"}, "study_planner")
    memory_manager.add_event(EventType.TOOL_RESULT, "✅ plan_generator: 生成了学习计划", {"tool_name": "plan_generator", "success": True}, "study_planner")
    memory_manager.add_event(EventType.ASSISTANT_MESSAGE, "这是你的英语学习计划：\n1. 每天背单词\n2. 练习听力\n3. 口语练习", agent_name="study_planner")
    
    # Test 4: Show all events (single session) with their IDs
    print("\n📋 Test 4: Show all events (single session) with their auto-generated IDs")
    
    all_events = memory_manager.get_events()
    print(f"Total events (single session): {len(all_events)}")
    
    for i, event in enumerate(all_events):
        print(f"{i+1}. [{event.id}] [{event.type.value}] {event.content[:50]}...")
        print(f"   Session: {event.session_id}")
        print(f"   Task: {event.task_id}")
        print()
    
    # Test 5: Get events by task (single session)
    print("\n🎯 Test 5: Get events by task (single session)")
    
    task1_events = memory_manager.get_events(task_id=task_id1)
    task2_events = memory_manager.get_events(task_id=task_id2)
    task3_events = memory_manager.get_events(task_id=task_id3)
    
    print(f"Task 1 (write_sorting_function) events: {len(task1_events)}")
    print(f"Task 2 (optimize_sorting_function) events: {len(task2_events)}")
    print(f"Task 3 (create_study_plan) events: {len(task3_events)}")
    
    # Test 6: Format full history (single session)
    print("\n📋 Test 6: Format full history (single session)")
    
    full_history = memory_manager.format_full_history()
    print("Full History:")
    print(full_history)
    
    print("\n✅ LLM-controlled memory design test completed!")


if __name__ == "__main__":
    asyncio.run(test_llm_controlled_memory())