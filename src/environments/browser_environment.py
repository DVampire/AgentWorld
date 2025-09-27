"""Browser Environment for AgentWorld - 直接使用 browser-use 工具源码."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union
from inspect import cleandoc
import logging

# Browser-use 核心组件
from browser_use import BrowserSession
from browser_use.tools.service import Tools as BrowserUseTools
from browser_use.agent.views import ActionModel, ActionResult
from browser_use.tools.views import (
    GoToUrlAction,
    ClickElementAction,
    InputTextAction,
    ScrollAction,
    SendKeysAction,
    CloseTabAction,
    SwitchTabAction,
    SearchGoogleAction,
    GetDropdownOptionsAction,
    SelectDropdownOptionAction,
    DoneAction,
    NoParamsAction,
    UploadFileAction
)
from browser_use.browser.views import BrowserError
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.base import BaseChatModel

# 项目依赖
try:
    from src.logger import logger
    from src.registry import ENVIRONMENTS
    from src.utils import assemble_project_path
except ImportError:
    # Fallback for direct testing
    class SimpleLogger:
        def info(self, msg): print(msg)
        def error(self, msg): print(f"ERROR: {msg}")

    logger = SimpleLogger()

    class SimpleRegistry:
        @staticmethod
        def register_module(force=True):
            def decorator(cls):
                return cls
            return decorator

    ENVIRONMENTS = SimpleRegistry()

    def assemble_project_path(path):
        """Simple path assembly."""
        return Path(path).resolve()


@ENVIRONMENTS.register_module(force=True)
class BrowserEnvironment:
    """Browser Environment that uses real browser-use tools."""
    
    def __init__(
        self,
        headless: bool = False,
        window_size: Tuple[int, int] = (1920, 1080),
        downloads_dir: str = "downloads",
        chrome_args: Optional[list[str]] = None,
        max_pages: int = 10,
        **kwargs
    ):
        """
        Initialize the browser environment using browser-use.
        
        Args:
            headless: Whether to run browser in headless mode
            window_size: Browser window size
            downloads_dir: Directory for downloads
            chrome_args: Additional Chrome arguments
            max_pages: Maximum number of pages to allow
        """
        self.headless = headless
        self.window_size = window_size
        self.downloads_dir = Path(assemble_project_path(downloads_dir))
        self.chrome_args = chrome_args or []
        self.max_pages = max_pages
        
        # Browser-use components - 使用原生组件
        self.browser_session: Optional[BrowserSession] = None
        self.tools: Optional[BrowserUseTools] = None
        self.file_system: Optional[FileSystem] = None
        self.page_extraction_llm: Optional[BaseChatModel] = None
        
        # Environment state
        self.state = {}
        self.info = {}
        self.done = False
        
        # Create downloads directory
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"| 🌐 Browser Environment initialized")
        logger.info(f"| ├── Headless: {self.headless}")
        logger.info(f"| ├── Window Size: {self.window_size}")
        logger.info(f"| └── Downloads Dir: {self.downloads_dir}")

    async def _init_browser_session(self):
        """Initialize browser session using browser-use."""
        try:
            logger.info("| ✅ 正在初始化 browser-use 原生组件")
            
            # 创建 browser session - 使用原生配置
            self.browser_session = BrowserSession(
                headless=self.headless,
                downloads_path=str(self.downloads_dir),
                args=self.chrome_args,
            )
            
            # 启动 browser session
            await self.browser_session.start()
            
            # 先初始化页面提取LLM
            page_extraction_llm = None
            exclude_actions = []
            
            try:
                from src.models import model_manager
                # 尝试获取一个browser-use兼容的模型作为页面提取LLM
                available_models = model_manager.list_models()
                if available_models:
                    # 优先使用browser-use兼容的模型（bs-前缀）
                    browser_models = [m for m in available_models if m.startswith('bs-')]
                    if browser_models:
                        # 在browser模型中优先选择gpt-4
                        preferred_models = [m for m in browser_models if 'gpt-4' in m.lower()]
                        if not preferred_models:
                            preferred_models = browser_models
                        
                        model_name = preferred_models[0]
                        page_extraction_llm = model_manager.get_model(model_name)
                        logger.info(f"| ✅ 使用browser-use兼容模型 {model_name} 作为页面提取LLM")
                    else:
                        logger.warning("| ⚠️ 没有browser-use兼容的LLM模型（bs-前缀），将排除需要LLM的动作")
                        exclude_actions.append('extract_structured_data')
                else:
                    logger.warning("| ⚠️ 没有可用的LLM模型，将排除需要LLM的动作")
                    exclude_actions.append('extract_structured_data')
            except Exception as llm_e:
                logger.warning(f"| ⚠️ 页面提取LLM初始化失败: {llm_e}，将排除需要LLM的动作")
                exclude_actions.append('extract_structured_data')
            
            self.page_extraction_llm = page_extraction_llm
            
            # 初始化 browser-use 原生工具
            self.tools = BrowserUseTools(
                exclude_actions=exclude_actions,  # 根据LLM可用性排除动作
                output_model=None,   # 不使用结构化输出
                display_files_in_done_text=True
            )
            
            # 初始化文件系统（可选）
            try:
                self.file_system = FileSystem(base_dir=self.downloads_dir)
            except Exception as fs_e:
                logger.warning(f"| ⚠️ 文件系统初始化失败，将使用基础功能: {fs_e}")
                self.file_system = None
            
            logger.info("| ✅ Browser session 初始化成功")
            
        except Exception as e:
            logger.error(f"| ❌ 初始化 browser session 失败: {e}")
            raise e

    async def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境到初始状态。
        
        Returns:
            Tuple of (state, info)
        """
        # 初始化 browser session（如果未初始化）
        if self.browser_session is None:
            await self._init_browser_session()
        
        # 导航到空白页面
        try:
            # 使用 browser-use 原生方法导航
            nav_action = ActionModel(go_to_url=GoToUrlAction(url="about:blank", new_tab=False))
            await self.tools.act(
                action=nav_action,
                browser_session=self.browser_session,
                file_system=self.file_system
            )
        except Exception as e:
            logger.warning(f"| ⚠️ 导航到空白页面失败: {e}")
        
        # 获取当前状态
        state = await self._get_current_state()
        info = dict(
            done=False,
            tabs=await self._get_tabs_info(),
            current_url=await self._get_current_url(),
            browser_ready=bool(self.browser_session and self.tools)
        )
        
        self.state = state
        self.info = info
        self.done = False
        
        logger.info("| 🔄 Browser Environment 重置到初始状态")
        return state, info
    
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the browser environment using browser-use tools.
        
        Args:
            action: Action dictionary containing operation type and parameters
            
        Returns:
            Tuple of (state, reward, terminated, truncated, info)
        """
        try:
            # 创建 ActionModel 并执行
            action_model = await self._create_action_model(action)
            
            # 使用 browser-use 原生工具执行动作
            action_result = await self.tools.act(
                action=action_model,
                browser_session=self.browser_session,
                page_extraction_llm=self.page_extraction_llm,
                file_system=self.file_system,
                available_file_paths=[],
                sensitive_data=None
            )
            
            # 更新状态
            new_state = await self._get_current_state()
            
            # 计算奖励
            reward = 1.0 if not action_result.error else -1.0
            
            # 检查终止条件
            terminated = action_result.is_done or self.done
            truncated = False
            
            # 更新信息
            info = dict(
                action_result=self._format_action_result(action_result),
                tabs=await self._get_tabs_info(),
                current_url=await self._get_current_url(),
                done=terminated,
                success=action_result.success if hasattr(action_result, 'success') else not bool(action_result.error)
            )
            
            self.state = new_state
            self.info = info
            self.done = terminated
            
            logger.info(f"| 🔄 Browser action executed: {action.get('type', 'unknown')}")
            
            return new_state, reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"| ❌ Browser action failed: {e}")
            error_state = {"error": str(e)}
            error_info = {"error": str(e), "done": False, "success": False}
            return error_state, -1.0, False, False, error_info

    async def _create_action_model(self, action: Dict[str, Any]) -> ActionModel:
        """根据 action 字典创建 browser-use ActionModel。"""
        action_type = action.get("type", action.get("action_type", ""))
        operation = action.get("operation", "")
        params = action.get("params", {})
        
        # 合并全部参数
        all_params = {**action, **params}
        
        try:
            # 首先获取 ActionModel 类型（动态创建）
            ActionModelClass = self.tools.registry.create_action_model()
            
            # 根据 action_type 和 operation 创建对应的参数
            action_data = self._get_action_data(action_type, operation, all_params)
            
            # 使用动态ActionModel类创建实例
            return ActionModelClass(**action_data)
            
        except Exception as e:
            logger.error(f"| ❌ 创建ActionModel失败: {e}")
            # 返回一个安全的默认动作
            ActionModelClass = self.tools.registry.create_action_model()
            return ActionModelClass(wait=3)

    def _get_action_data(self, action_type: str, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """根据 action_type 和 operation 获取动作数据。"""
        if action_type == "navigation":
            return self._get_navigation_action_data(operation, params)
        elif action_type == "interaction":
            return self._get_interaction_action_data(operation, params)
        elif action_type == "data_extraction":
            return self._get_data_extraction_action_data(operation, params)
        elif action_type == "tab_management":
            return self._get_tab_management_action_data(operation, params)
        elif action_type == "search":
            return self._get_search_action_data(operation, params)
        elif action_type == "done":
            return self._get_done_action_data(params)
        else:
            # 默认到 wait 动作
            return {"wait": {"seconds": max(1, min(params.get("seconds", 3), 30))}}
    
    def _get_navigation_action_data(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取导航动作数据。"""
        if operation == "goto":
            return {
                "go_to_url": GoToUrlAction(
                    url=params.get("url", ""),
                    new_tab=params.get("new_tab", False)
                )
            }
        elif operation == "back":
            return {"go_back": NoParamsAction()}
        elif operation == "refresh":
            # 使用发送键盘快捷键来刷新
            return {"send_keys": SendKeysAction(keys="F5")}
        else:
            raise ValueError(f"不支持的导航操作: {operation}")

    def _get_interaction_action_data(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取交互动作数据。"""
        if operation == "click":
            # click需要有效的元素索引，且不能为0
            index = params.get("index")
            if index is None:
                raise ValueError("click operation requires 'index' parameter - must specify which element to click")
            if index == 0:
                raise ValueError("click operation cannot use index=0 - browser-use prohibits clicking index 0")
            
            return {
                "click_element_by_index": ClickElementAction(
                    index=index,
                    while_holding_ctrl=params.get("while_holding_ctrl", False)
                )
            }
        elif operation == "type":
            # input_text需要有效的元素索引，不能随意默认
            index = params.get("index")
            if index is None:
                raise ValueError("type operation requires 'index' parameter - must specify which element to type into")
            
            return {
                "input_text": InputTextAction(
                    index=index,
                    text=params.get("text", ""),
                    clear_existing=params.get("clear_existing", True)
                )
            }
        elif operation == "scroll":
            direction = params.get("direction", "down")
            return {
                "scroll": ScrollAction(
                    down=direction.lower() == "down",
                    num_pages=params.get("num_pages", 1.0),
                    frame_element_index=params.get("frame_element_index")
                )
            }
        elif operation == "send_keys":
            return {
                "send_keys": SendKeysAction(
                    keys=params.get("keys", "")
                )
            }
        elif operation == "scroll_to_text":
            return {"scroll_to_text": params.get("text", "")}
        elif operation == "upload_file":
            return {
                "upload_file_to_element": UploadFileAction(
                    index=params.get("index", 0),
                    path=params.get("path", "")
                )
            }
        else:
            raise ValueError(f"不支持的交互操作: {operation}")

    def _get_data_extraction_action_data(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取数据提取动作数据。"""
        if operation == "extract_structured_data":
            # extract_structured_data 是特殊的，它直接接收kwargs
            return {
                "extract_structured_data": {
                    "query": params.get("query", ""),
                    "extract_links": params.get("extract_links", False),
                    "start_from_char": params.get("start_from_char", 0)
                }
            }
        elif operation == "get_dropdown_options":
            return {
                "get_dropdown_options": GetDropdownOptionsAction(
                    index=params.get("index", 1)
                )
            }
        elif operation == "select_dropdown_option":
            return {
                "select_dropdown_option": SelectDropdownOptionAction(
                    index=params.get("index", 1),
                    text=params.get("text", "")
                )
            }
        elif operation == "wait":
            # wait 动作需要一个包含seconds字段的参数对象
            seconds = max(1, min(params.get("seconds", 3), 30))
            return {"wait": {"seconds": seconds}}
        else:
            # 如果是未知的数据提取操作，默认使用 wait
            return {"wait": {"seconds": 3}}

    def _get_tab_management_action_data(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取标签管理动作数据。"""
        if operation == "open_tab":
            # 使用 go_to_url 动作打开新标签
            return {
                "go_to_url": GoToUrlAction(
                    url=params.get("url", "about:blank"),
                    new_tab=True
                )
            }
        elif operation == "close_tab":
            return {
                "close_tab": CloseTabAction(
                    tab_id=params.get("tab_id", "0000")  # 默认tab_id
                )
            }
        elif operation == "switch_tab":
            return {
                "switch_tab": SwitchTabAction(
                    tab_id=params.get("tab_id", "0000")  # 默认tab_id
                )
            }
        elif operation == "list_tabs":
            # 等待一下让标签加载完成
            return {"wait": {"seconds": 1}}
        else:
            raise ValueError(f"不支持的标签管理操作: {operation}")

    def _get_search_action_data(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取搜索动作数据。"""
        if operation == "google":
            return {
                "search_google": SearchGoogleAction(
                    query=params.get("query", params.get("search_query", ""))
                )
            }
        else:
            raise ValueError(f"不支持的搜索操作: {operation}")
    
    def _get_done_action_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取完成动作数据。"""
        return {
            "done": DoneAction(
                text=params.get("text", "任务完成"),
                success=params.get("success", True),
                files_to_display=params.get("files_to_display", [])
            )
        }
    
    def _format_action_result(self, action_result: ActionResult) -> Dict[str, Any]:
        """格式化 ActionResult 为字典。"""
        return {
            "success": not bool(action_result.error),
            "content": action_result.extracted_content or "",
            "error": action_result.error,
            "is_done": action_result.is_done,
            "long_term_memory": action_result.long_term_memory,
            "metadata": action_result.metadata,
            "attachments": getattr(action_result, 'attachments', [])
        }
    
    async def _get_current_state(self) -> Dict[str, Any]:
        """获取当前浏览器状态，使用browser-use官方方法。"""
        try:
            if self.browser_session:
                current_url = await self._get_current_url()
                tabs = await self._get_tabs_info()
                
                # 使用browser-use官方的get_browser_state_summary方法
                browser_state_summary = {}
                try:
                    if hasattr(self.browser_session, 'get_browser_state_summary'):
                        summary = await self.browser_session.get_browser_state_summary(
                            cache_clickable_elements_hashes=True,
                            include_screenshot=False,  # 不包含截图以提高性能
                            cached=False,
                            include_recent_events=False
                        )
                        # 提取核心状态信息
                        browser_state_summary = {
                            "url": summary.url,
                            "title": summary.title,
                            "tabs": [{"target_id": tab.target_id, "url": tab.url, "title": tab.title} for tab in summary.tabs],
                            "dom_elements_count": len(summary.dom_state.selector_map) if summary.dom_state and summary.dom_state.selector_map else 0,
                            "interactive_elements": len([k for k, v in summary.dom_state.selector_map.items() if k > 0]) if summary.dom_state and summary.dom_state.selector_map else 0,
                            "dom_representation": summary.dom_state.llm_representation() if summary.dom_state else ""
                        }
                except Exception as e:
                    logger.warning(f"| ⚠️ 获取browser-use状态失败: {e}")
                    pass
                
                return {
                    "current_url": current_url,
                    "tabs": tabs,
                    "ready": True,
                    "browser_state_summary": browser_state_summary
                }
            else:
                return {"ready": False, "error": "浏览器未初始化"}
        except Exception as e:
            return {"ready": False, "error": str(e)}

    async def _get_current_url(self) -> str:
        """获取当前 URL。"""
        try:
            if self.browser_session:
                return await self.browser_session.get_current_page_url()
            return "about:blank"
        except Exception:
            return "about:blank"

    async def _get_tabs_info(self) -> list:
        """获取所有打开标签的信息。"""
        try:
            if self.browser_session:
                tabs = await self.browser_session.get_tabs()
                current_tab = None
                try:
                    current_tab = await self.browser_session.get_current_tab()
                except Exception:
                    pass
                
                return [
                    {
                        "target_id": tab.target_id if hasattr(tab, 'target_id') else tab.id,
                        "url": tab.url or "about:blank",
                        "title": tab.title or "无标题",
                        "active": current_tab and tab.id == current_tab.id if current_tab else False
                    }
                    for tab in tabs
                ]
            return []
        except Exception as e:
            logger.warning(f"| ⚠️ 获取标签信息失败: {e}")
            return []

    async def get_browser_state_info(self) -> Dict[str, Any]:
        """获取详细的浏览器状态信息，使用browser-use官方方法。"""
        try:
            if not self.browser_session:
                return {"error": "浏览器未初始化"}
            
            # 使用browser-use官方的完整状态获取方法
            try:
                if hasattr(self.browser_session, 'get_browser_state_summary'):
                    summary = await self.browser_session.get_browser_state_summary(
                        cache_clickable_elements_hashes=True,
                        include_screenshot=True,  # 包含截图用于调试
                        cached=False,
                        include_recent_events=True
                    )
                    
                    # 提取完整的状态信息
                    state_info = {
                        "url": summary.url,
                        "title": summary.title,
                        "tabs": [{"target_id": tab.target_id, "url": tab.url, "title": tab.title} for tab in summary.tabs],
                        "ready": True,
                        "has_screenshot": summary.screenshot is not None,
                        "recent_events": summary.recent_events,
                        "browser_errors": summary.browser_errors or [],
                        "is_pdf_viewer": summary.is_pdf_viewer,
                        "pixels_above": summary.pixels_above,
                        "pixels_below": summary.pixels_below
                    }
                    
                    # DOM状态信息
                    if summary.dom_state:
                        state_info.update({
                            "dom_elements_count": len(summary.dom_state.selector_map),
                            "interactive_elements": len([k for k, v in summary.dom_state.selector_map.items() if k > 0]),
                            "dom_representation": summary.dom_state.llm_representation(),
                            "selector_map_keys": list(summary.dom_state.selector_map.keys())[:20]  # 前20个索引，避免输出过长
                        })
                    
                    return state_info
                else:
                    # 回退到基础方法
                    elements_info = {}
                    try:
                        selector_map = await self.browser_session.get_selector_map()
                        elements_info = {
                            "total_elements": len(selector_map),
                            "interactive_elements": len([k for k, v in selector_map.items() if k > 0])
                        }
                    except Exception:
                        pass
                    
                    return {
                        "url": await self._get_current_url(),
                        "tabs": await self._get_tabs_info(),
                        "elements": elements_info,
                        "ready": True
                    }
            except Exception as e:
                logger.error(f"| ❌ 获取browser-use完整状态失败: {e}")
                return {"error": str(e), "ready": False}
            
        except Exception as e:
            return {"error": str(e), "ready": False}
    
    async def get_page_content_official(self, extract_links: bool = False) -> Dict[str, Any]:
        """使用browser-use官方方法提取页面内容。"""
        try:
            if not self.browser_session or not self.tools:
                return {"error": "浏览器或工具未初始化"}
            
            # 使用browser-use官方的extract_clean_markdown方法
            if hasattr(self.tools, 'extract_clean_markdown'):
                content, stats = await self.tools.extract_clean_markdown(
                    self.browser_session, 
                    extract_links=extract_links
                )
                return {
                    "content": content,
                    "stats": stats,
                    "success": True
                }
            else:
                return {"error": "extract_clean_markdown方法不可用"}
        except Exception as e:
            logger.error(f"| ❌ 官方页面内容提取失败: {e}")
            return {"error": str(e)}

    async def take_screenshot(self, file_path: Optional[str] = None) -> str:
        """拍摄当前页面的截图。"""
        try:
            if not self.browser_session:
                raise Exception("浏览器未初始化")
            
            if not file_path:
                import time
                file_path = str(self.downloads_dir / f"screenshot_{int(time.time())}.png")
            
            await self.browser_session.take_screenshot(file_path)
            return file_path
        except Exception as e:
            logger.error(f"| ❌ 截图失败: {e}")
            raise e
    
    async def close(self):
        """清理资源。"""
        try:
            if self.browser_session:
                await self.browser_session.stop()
                self.browser_session = None
                self.tools = None
                logger.info("| 🔄 Browser session 已关闭")
        except Exception as e:
            logger.error(f"| ⚠️ 关闭 browser session 错误: {e}")
