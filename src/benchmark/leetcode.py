import os
import time
import json
import re
import inflection
import asyncio
from typing import Optional, Dict, Any, Set, List, ClassVar
from pydantic import PrivateAttr, Field, ConfigDict
from playwright.async_api import async_playwright
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.config import config
from src.benchmark.types import Benchmark, Task, Stats , Result
from src.registry import BENCHMARK
from src.utils import file_lock

# 创建提交锁，确保浏览器操作串行化
class SubmitLock:
    """用于序列化浏览器提交操作的锁"""
    _instance = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._lock = asyncio.Lock()
        return cls._instance
    
    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def __aenter__(self):
        await self.lock.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        self.lock.release()

submit_lock = SubmitLock()

SYSTEM_PROMPT = """
You are a helpful assistant that solves LeetCode coding problems. Please think step by step and provide your solution code.

Output format:
The output should be a JSON object with the following fields:
{
    "reasoning": "Your step-by-step reasoning process",
    "result": "Your solution code".
}

Example:
Task ID: 1
Problem Name: Two Sum
Problem: Given an array of integers, return the two numbers such that they add up to a specific target.
Language: python3
Template:
```python
#
# @lc app=leetcode id=1 lang=python3
#
# [1] Two Sum
#

# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
# @lc code=end
```

Output:
{
    "reasoning": "Step 1: I need to find two numbers that sum to the target. I can use a hashmap to store each number and its index as I iterate through the array.\\n\\nStep 2: For each number, I calculate the complement (target - current number). If the complement exists in the hashmap, I found the pair. Otherwise, I add the current number to the hashmap.\\n\\nStep 3: This approach has O(n) time complexity and O(n) space complexity.",
    "result": "#\\n# @lc app=leetcode id=1 lang=python3\\n#\\n# [1] Two Sum\\n#\\n\\n# @lc code=start\\nclass Solution:\\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\\n        hashmap = {}\\n        for i, num in enumerate(nums):\\n            complement = target - num\\n            if complement in hashmap:\\n                return [hashmap[complement], i]\\n            hashmap[num] = i\\n        return []\\n# @lc code=end"
}

Please write your solution code base on your language template.
"""

class CodeSubmitter:
    def __init__(self, headless: bool = False):
        """
        Initialization: Automatically load configuration from environment variables
        Uses git workflow: clone -> write file -> commit -> push -> codespace pull -> open -> submit
        """
        self.username = os.getenv("GITHUB_USERNAME")
        self.password = os.getenv("GITHUB_PASSWORD")
        self.project_url = os.getenv("GITHUB_PROJECT_URL")
        self.codespace_url = os.getenv("GITHUB_CODESPACE_URL")
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE") or None

        self.playwright = None
        self.context = None
        self.page = None
        self.repo_path = None
        self.headless = headless

    async def initialize(self):
        logger.info("| 🚀 Initializing LeitCode Benchmark Submitter...")
        
        self.repo_slug = self.project_url.rstrip('/').replace('https://github.com/', '').replace('http://github.com/', '')
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE") or None
        
        # 1. 动态查找项目根目录 "AgentWorld"
        current_file_path = os.path.abspath(__file__)
        project_root_name = "AgentWorld"
        
        if project_root_name in current_file_path:
            # 截取路径直到 AgentWorld
            # 例如: /Users/jfw/.../AgentWorld/src/bench.py -> /Users/jfw/.../AgentWorld
            root_path = current_file_path.split(project_root_name)[0] + project_root_name
            
            # 2. 拼接目标子路径
            self.output_dir = os.path.join(
                root_path, 
                "workdir", "tool_calling_agent", "benchmark", "leetcode"
            )
        else:
            # 如果路径里没找到 AgentWorld，回退到当前目录
            logger.warning(f"| ⚠️ Could not find '{project_root_name}' in path. Saving to local ./results")
            self.output_dir = os.path.join(os.getcwd(), "results")

        # 3. 确保目录存在
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"| 📂 Output directory ready: {self.output_dir}")
        except Exception as e:
            logger.error(f"| ❌ Failed to create directory: {e}")
            self.output_dir = "." # 最后的保底

        self.output_file = os.path.join(self.output_dir, "results.jsonl")
        # ================= [修改结束] =================
        
        logger.info(f"| 📝 Results will be saved to: {self.output_file}")
        # 保持原始引用
        self.base_dir = os.path.join(config.workdir, "benchmark", "leetcode")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 1. Setup git repository
        await self._setup_git_repo()
        
        # 2. Setup browser
        self.playwright = await async_playwright().start()
        user_data_dir = os.path.join(self.base_dir, "playwright_user_data")
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.headless,
            viewport={'width': 1280, 'height': 800},
            args=["--disable-blink-features=AutomationControlled"],
            permissions=["clipboard-read", "clipboard-write"]
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()

        # 3. Setup LeetCode cookie
        await self._setup_leetcode_cookie()
        
        # 4. Login GitHub and enter Codespace
        await self._login_and_navigate()
        
        # 5. Verify LeetCode login status
        await self._ensure_leetcode_login()
        
        logger.info("| ✅ Environment all ready, can start evaluation")

    

    async def save_result(self, result: Result) -> None:
        """
        保存结果，并过滤掉 prompt, answer (代码), extra 等冗余字段
        """
        try:
            # [修改点] 使用 exclude 参数过滤不需要的字段
            # 1. 'prompt': 题目描述太长，且不需要保存
            # 2. 'answer': 你提到代码不保存在这里
            # 3. 'ground_truth': 通常为 null 或不重要
            # 4. 'extra': 这里面包含了重复的 result 信息，直接去掉以减少冗余
            json_line = result.model_dump_json(
                exclude={
                    'prompt', 
                    'answer', 
                    'ground_truth', 
                    'extra' 
                }
            )
            
            # 使用 'a' (append) 模式写入
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json_line + "\n")
                
            logger.info(f"| 💾 Saved result for Task {result.task_id} (Score: {result.score})")
            
        except Exception as e:
            logger.error(f"| ❌ Failed to save result to file: {e}")


    async def _setup_git_repo(self):
        """Create temp directory and clone the project"""
        logger.info("| 🔍 Cloning git repository...")
        
        repo_name = self.repo_slug.split('/')[-1]
        self.repo_path = os.path.join(self.base_dir, repo_name)
        
        # Convert to SSH format: git@github.com:username/repo.git
        ssh_url = f"git@github.com:{self.repo_slug}.git"
        
        try:
            if os.path.exists(self.repo_path):
                logger.info(f"| 🔍 Repository already exists, skipping clone")
                return
            await self._run_git_command(['clone', ssh_url, repo_name], self.base_dir)
            logger.info(f"| 🔍 Successfully cloned repository to {self.repo_path}")
        except Exception as e:
            raise Exception(f"| ❌ Failed to clone repository: {str(e)}")

    async def _run_git_command(self, args: List[str], cwd: str) -> str:
        """Run git command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            'git', *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode().strip() or stdout.decode().strip()
            raise Exception(f"Git command failed (exit {process.returncode}): {error_msg}")
        return stdout.decode().strip()
    
    async def _setup_leetcode_cookie(self):
        """Use Playwright to login and extract cookies to bypass Cloudflare 403"""
        logger.info("| 🔍 Logging in to LeetCode via browser to fetch session cookies...")
        
        try:
            if self.leetcode_cookie:
                logger.info("| 🔍 LeetCode cookie already exists, skipping login")
                return
            
            # 1. Navigate to login page
            await self.page.goto("https://leetcode.com/accounts/login/", wait_until="networkidle")
            await asyncio.sleep(2)

            # 2. Check if already logged in (due to persistent context)
            if "login" not in self.page.url:
                logger.info("| 🔍 Already logged in to LeetCode.")
            else:
                logger.info("| 🔍 Entering LeetCode credentials...")
                # LeetCode uses specific IDs for login inputs
                await self.page.fill('input[name="login"]', os.getenv("LEETCODE_USERNAME"))
                await self.page.fill('input[name="password"]', os.getenv("LEETCODE_PASSWORD"))
                
                # Click sign in and wait for navigation
                await self.page.click('button[type="submit"], #signin_btn')
                
                # Wait for navigation back to home or dashboard
                try:
                    await self.page.wait_for_url("https://leetcode.com/", timeout=20000)
                    logger.info("| ✅ LeetCode web login successful.")
                except:
                    logger.warning("| ⚠️ Login navigation timed out, checking cookies anyway...")

            # 3. Extract cookies from context
            cookies = await self.context.cookies()
            cookie_str = "; ".join([f"{c['name']}={c['value']}" for c in cookies if c['domain'].endswith("leetcode.com")])
            
            if "LEETCODE_SESSION" in cookie_str:
                self.leetcode_cookie = cookie_str
                logger.info(f"| ✅ Successfully extracted LeetCode cookies (length: {len(cookie_str)})")
            else:
                logger.error("| ❌ Failed to find LEETCODE_SESSION in cookies.")
                
        except Exception as e:
            logger.error(f"| ❌ Error during LeetCode cookie setup: {str(e)}")
            raise e

    async def _login_and_navigate(self):
        """Login GitHub and enter specified Codespace"""
        try:
            await self.page.goto("https://github.com/login")
            await asyncio.sleep(1)
            if await self.page.locator('input[name="login"]').count() > 0:
                logger.info("| 🔍 Logging in to GitHub...")
                await self.page.fill('input[name="login"]', self.username)
                await self.page.fill('input[name="password"]', self.password)
                await self.page.click('input[type="submit"][value="Sign in"]')
                await self.page.wait_for_url("https://github.com/", timeout=15000)
            
            logger.info(f"| 🔍 Visiting project: {self.project_url}")
            await self.page.goto(self.project_url)
            await asyncio.sleep(2)
            
            logger.info(f"| 🔍 Going to Codespace: {self.codespace_url} ...")
            await self.page.goto(self.codespace_url)
            await asyncio.sleep(10)
            
        except Exception as e:
            raise Exception(f"| ❌ Failed to enter Codespace: {str(e)}")

    async def _ensure_leetcode_login(self):
        """Check LeetCode plugin status, if not logged in then use Cookie to log in"""
        logger.info("| 🔍 Verifying LeetCode login status...")
        try:
            leetcode_icon = self.page.locator('a[aria-label="LeetCode"], li[aria-label="LeetCode"], .codicon-leetcode').first
            if await leetcode_icon.is_visible():
                await leetcode_icon.click()
                await asyncio.sleep(10)
            
            needs_login = await self.page.get_by_text("Sign in to LeetCode", exact=False).is_visible()
            
            if needs_login:
                logger.info("| 🔍 Detected not logged in, using Cookie to log in...")
                await self._perform_leetcode_login()
            else:
                logger.info("| 🔍 Looks like already logged in (or no login prompt detected).")
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.warning(f"| ⚠️ Non-fatal error occurred during LeetCode login verification: {e}")

    async def _perform_leetcode_login(self):
        await self.page.keyboard.press("F1")
        await asyncio.sleep(1)
        await self.page.keyboard.type("LeetCode: Sign In")
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(1)
        await self.page.keyboard.press("Enter") 
        await asyncio.sleep(1)
        await self.page.keyboard.type("Cookie")
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(1)
        
        logger.info("| 🔍 Entering Cookie...")
        json_cookie = json.dumps(self.leetcode_cookie)
        await self.page.evaluate(f"navigator.clipboard.writeText({json_cookie})")
        await asyncio.sleep(0.2)
        await self.page.keyboard.press("Control+V") 
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Enter")
        
        logger.info("| 🔍 Waiting for login to take effect...")
        await asyncio.sleep(30)

    async def _sync_codespace_repo(self):
        """[新增] 在 Codespace 执行 Git Pull 确保代码同步"""
        logger.info("| 🔄 Syncing Codespace with remote repository (Git Pull)...")
        
        # 此时可能没有打开的文件编辑器，点击 body 确保获取焦点以便唤起命令面板
        try:
            await self.page.click("body", timeout=1000)
        except:
            pass

        # 唤起命令面板
        await self.page.keyboard.press("Control+Shift+P")
        await asyncio.sleep(1)
        
        # 输入并执行 Git Pull
        await self.page.keyboard.type("Git: Pull")
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Enter")
        
        # 等待同步完成（根据网络情况可调整）
        logger.info("| ⏳ Waiting 5 seconds for git pull to complete...")
        await asyncio.sleep(5)

    async def _open_editor_file(self, filename):
        """Open explorer and open code file"""
        logger.info(f"| 🔍 Opening work file: {filename}")
        
        await self.page.keyboard.press("Control+Shift+P")

# 2. 输入 "Close All Editors" 并回车
        await self.page.keyboard.type("Close All Editors")
        await self.page.keyboard.press("Enter")

        # Switch back to explorer
        explorer_icon = self.page.locator('li[aria-label="Explorer"], li[aria-label="资源管理器"], .codicon-files').first
        if await explorer_icon.is_visible():
            await explorer_icon.click()
            await asyncio.sleep(0.5)

        # Summon file search
        # 尝试点击编辑器区域，如果不存在（例如还没打开文件），则点击空白处
        try:
            await self.page.click('.monaco-editor', timeout=1000)
        except:
            await self.page.mouse.click(500, 500)
            
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Control+P")
        await asyncio.sleep(0.5)
        await self.page.keyboard.type(filename)
        await asyncio.sleep(1)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(2)

    async def submit_code(self, code_content: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Submit code using git workflow: write -> commit -> push -> PULL in codespace -> open -> submit"""

        target_file = filename or "code.py"
        logger.info(f"| 🔍 Preparing evaluation for {target_file}...")
        
        # 1. Write code to file in repo (Local)
        file_path = os.path.join(self.repo_path, target_file)
        
        logger.info(f"| 🔍 Writing code to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # 2. Git add, commit, and push (Local)
        logger.info("| 🔍 Committing and pushing to GitHub...")
        try:
            # Configure git user (required for commit)
            await self._run_git_command(['config', 'user.name', self.username], self.repo_path)
            await self._run_git_command(['config', 'user.email', f'{self.username}@users.noreply.github.com'], self.repo_path)
            
            # Git add
            await self._run_git_command(['add', target_file], self.repo_path)
            
            # Check if there are changes to commit
            status_out = await self._run_git_command(['status', '--porcelain'], self.repo_path)
            
            if status_out.strip():
                # Git commit
                await self._run_git_command(['commit', '-m', f'Add solution: {target_file}'], self.repo_path)
                
                # Git push
                await self._run_git_command(['push'], self.repo_path)
                logger.info("| 🔍 Successfully pushed to GitHub")
            else:
                logger.info("| 🔍 No changes to commit (file unchanged)")
        except Exception as e:
            logger.warning(f"| ⚠️ Git operation failed: {str(e)}")
            # Continue anyway, file might already be in repo or push might not be needed
        
        # 3. Wait a bit for GitHub to sync
        await asyncio.sleep(3)

        # 4. [修改] Pull latest code inside Codespace FIRST
        # 只有先 Pull 把文件拉下来，后面的 Open 才能找到文件
        await self._sync_codespace_repo()
        
        # 5. [修改] Open file in Codespace
        await self._open_editor_file(target_file)
        await asyncio.sleep(2)
        
        # 6. Trigger LeetCode submission
        logger.info("| 🔍 Triggering LeetCode submission...")
        await self.page.keyboard.press("Control+Shift+P")
        await asyncio.sleep(1)
        await self.page.keyboard.type("LeetCode: Submit to LeetCode")
        await asyncio.sleep(1)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(5)
        
        # 7. Wait for result
        result = await self._wait_for_result()
        # 1. 打开命令面板
        await self.page.keyboard.press("Control+Shift+P")

# 2. 输入 "Close All Editors" 并回车
        await self.page.keyboard.type("Close All Editors")
        await self.page.keyboard.press("Enter")

        return result

    async def _wait_for_result(self) -> Dict[str, Any]:
        """
        Wait for LeetCode result and parse detailed metrics from the UI text.
        Returns a dictionary containing raw status and parsed metrics.
        """
        # 初始化默认数据结构
        metrics = {
            "status": "Timeout",
            "total_cases": 0,
            "passed_cases": 0,
            "runtime": 0.0,            # ms
            "memory_usage": 0.0,       # MB
            "runtime_beats": 0.0,      # percentage
            "memory_beats": 0.0,       # percentage
        }
        
        end_time = time.time() + 60
        # 结果关键词
        keywords = ["Accepted", "Wrong Answer", "Time Limit Exceeded", "Runtime Error", "Memory Limit Exceeded", "Compile Error"]
        found = False
        
        logger.info("| 🔍 Waiting for detailed evaluation results...")
        
        while time.time() < end_time and not found:
            for frame in self.page.frames:
                try:
                    content_locator = frame.locator("body")
                    # 获取页面全部文本
                    page_text = await content_locator.inner_text()
                    
                    # 1. 优先检测状态关键词
                    detected_status = next((kw for kw in keywords if kw in page_text), None)
                    
                    if detected_status:
                        metrics["status"] = detected_status
                        metrics["details_text"] = page_text[:500] 

                        # --- 通用解析：对 Accepted, Wrong Answer, TLE 等都尝试抓取用例数 ---
                        # 兼容 "68/68 cases passed" 和 "34/68 test cases passed"
                        cases_match = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:test\s*)?cases\s*passed", page_text, re.IGNORECASE)
                        if cases_match:
                            metrics["passed_cases"] = int(cases_match.group(1))
                            metrics["total_cases"] = int(cases_match.group(2))
                        
                        # --- 特有解析：只有 Accepted 才会有运行时间和击败比例 ---
                        if detected_status == "Accepted":
                            # 解析运行时: 抓取 "27 ms" 或 "Runtime: 27 ms"
                            runtime_match = re.search(r"(\d+(?:\.\d+)?)\s*ms", page_text, re.IGNORECASE)
                            if runtime_match:
                                metrics["runtime"] = float(runtime_match.group(1))

                            # 解析内存: 抓取 "21.4 MB" 或 "Memory: 21.4 MB"
                            memory_match = re.search(r"(\d+(?:\.\d+)?)\s*MB", page_text, re.IGNORECASE)
                            if memory_match:
                                metrics["memory_usage"] = float(memory_match.group(1))
                                
                            # 解析击败率: 抓取 "beats 27.14 %"
                            beats_matches = re.findall(r"beats\s*([\d\.]+)\s*%", page_text, re.IGNORECASE)
                            if len(beats_matches) >= 1:
                                metrics["runtime_beats"] = float(beats_matches[0])
                            if len(beats_matches) >= 2:
                                metrics["memory_beats"] = float(beats_matches[1])

                        # --- 特有解析：Wrong Answer 有时会有输入/输出详情，如果有需要可以后续添加 ---
                        # 目前只要用例数抓对了，Result 里的 extra 就会正确显示

                        found = True
                        break
                except Exception:
                    continue
            
            if found: break
            await asyncio.sleep(1)
            
        logger.info(f"| 🔍 Result Parsed: {metrics['status']} (Passed: {metrics['passed_cases']}/{metrics['total_cases']})")
        return metrics
    async def close(self):
        """Cleanup browser and temp directory"""
        if self.context: 
            await self.context.close()
        if self.playwright: 
            await self.playwright.stop()

@BENCHMARK.register_module(force=True)
class LeetCodeBenchmark(Benchmark):
    """
    LeetCode Benchmark with Resume Capability.
    Automatically filters out tasks present in 'tmp/answer.jsonl'.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="leetcode", description="The name of the benchmark")
    path: str = Field(default="datasets/leetcode", description="The path to the benchmark dataset")
    language: str = Field(default="python3", description="Programming language for LeetCode (e.g., python3, cpp, java)")
    
    system_prompt: Optional[str] = Field(default=SYSTEM_PROMPT, description="The system prompt for the benchmark")
    
    _id_to_record_map: Dict[str, Dict] = PrivateAttr(default_factory=dict)
    _submitter: Any = PrivateAttr(default=None)
    _submitter_started: bool = PrivateAttr(default=False)
    
    _data_records: List[Dict] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tasks: List[Task] = PrivateAttr(default_factory=list)

    # Configuration for different languages
    LANGUAGE_CONFIG: ClassVar[Dict[str, Dict[str, str]]] = {
        "python3": {"ext": "py", "lang_tag": "python3", "comment": "#"},
        "python": {"ext": "py", "lang_tag": "python", "comment": "#"},
        "cpp": {"ext": "cpp", "lang_tag": "cpp", "comment": "//"},
        "c++": {"ext": "cpp", "lang_tag": "cpp", "comment": "//"},
        "java": {"ext": "java", "lang_tag": "java", "comment": "//"},
        "javascript": {"ext": "js", "lang_tag": "javascript", "comment": "//"},
        "typescript": {"ext": "ts", "lang_tag": "typescript", "comment": "//"},
        "c": {"ext": "c", "lang_tag": "c", "comment": "//"},
        "csharp": {"ext": "cs", "lang_tag": "csharp", "comment": "//"},
        "c#": {"ext": "cs", "lang_tag": "csharp", "comment": "//"},
        "go": {"ext": "go", "lang_tag": "golang", "comment": "//"},
        "ruby": {"ext": "rb", "lang_tag": "ruby", "comment": "#"},
        "swift": {"ext": "swift", "lang_tag": "swift", "comment": "//"},
        "rust": {"ext": "rs", "lang_tag": "rust", "comment": "//"},
        "scala": {"ext": "scala", "lang_tag": "scala", "comment": "//"},
        "kotlin": {"ext": "kt", "lang_tag": "kotlin", "comment": "//"},
        "php": {"ext": "php", "lang_tag": "php", "comment": "//"},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submitter = CodeSubmitter(headless=False)
        self._submitter_started = False

    async def initialize(self):
        """Initialize the benchmark by loading dataset, filtering finished tasks, and starting browser."""
        try:
            from src.data.leetcode import LeetCodeDataset
            # 1. 加载数据集
            dataset = LeetCodeDataset(
                path=self.path,
                split=self.split,
                name=self.subset if self.subset else None
            )
            self._id_to_record_map = {}
            
            # 获取原始数据记录
            if hasattr(dataset, 'data'):
                self._data_records = dataset.data.to_dict(orient="records")
            else:
                self._data_records = []

            # ================= [新增逻辑：断点续跑过滤] =================
            
            # A. 动态定位结果文件路径 (逻辑需与 CodeSubmitter 保持完全一致)
            current_file_path = os.path.abspath(__file__)
            project_root_name = "AgentWorld"
            
            if project_root_name in current_file_path:
                root_path = current_file_path.split(project_root_name)[0] + project_root_name
                output_dir = os.path.join(
                    root_path, 
                    "workdir", "tool_calling_agent", "benchmark", "leetcode"
                )
            else:
                output_dir = os.path.join(os.getcwd(), "results")
            
            # 固定文件名，不再带时间戳
            result_file = os.path.join(output_dir, "results.jsonl")

            # B. 读取已完成的任务 ID
            finished_ids = set()
            if os.path.exists(result_file):
                logger.info(f"| 🔄 Found existing result file: {result_file}, checking for finished tasks...")
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            try:
                                record = json.loads(line)
                                if "task_id" in record:
                                    finished_ids.add(str(record["task_id"]))
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.warning(f"| ⚠️ Error reading result file for filtering: {e}")
            
            # C. 过滤 self._data_records
            original_count = len(self._data_records)
            self._data_records = [
                r for r in self._data_records 
                if str(r.get("id") or r.get("task_id", "0")) not in finished_ids
            ]
            filtered_count = len(self._data_records)
            
            if len(finished_ids) > 0:
                logger.info(f"| ⏭️ Skipped {original_count - filtered_count} tasks (already in results). Remaining: {filtered_count} tasks.")
            else:
                logger.info(f"| 🆕 No finished tasks found. Starting fresh with all {filtered_count} tasks.")

            # ================= [逻辑结束] =================

            # 2. 基于过滤后的数据建立索引映射
            for record in self._data_records:
                tid = str(record.get("id") or record.get("task_id", "0"))
                self._id_to_record_map[tid] = record
                
            logger.info(f"[{self.name}] Index built: {len(self._id_to_record_map)} records mapped.")
            
            # ================= [新增：提前初始化浏览器] =================
            # 在 benchmark 初始化阶段就启动浏览器，而不是等到第一次 eval
            # 这样多个并发任务可以共享同一个浏览器实例
            if not self._submitter_started and self._data_records:
                logger.info(f"| 🚀 Pre-initializing browser for concurrent evaluation...")
                try:
                    await self._submitter.initialize()
                    self._submitter_started = True
                    logger.info(f"| ✅ Browser initialized successfully, ready for concurrent tasks")
                except Exception as e:
                    logger.error(f"| ❌ Failed to initialize browser during benchmark init: {e}")
                    # 不抛出异常，允许后续尝试
            # ================= [逻辑结束] =================
            
        except ImportError:
            logger.error(f"[{self.name}] Failed to import LeetCodeDataset")
            
    async def reset(self) -> Optional[Task]:
        self._index = 0
        self._tasks = []
        return await self.step()

    async def step(self) -> Optional[Task]:
        if self._index >= len(self._data_records):
            return None
        
        record = self._data_records[self._index]
        self._index += 1
        
        task_id = str(record.get("id") or record.get("task_id", ""))
        task_name = record.get("name") or record.get("problem_name") or "Unknown"
        
        # templates are in record
        templates = record.get("code_template", {})
        # Map our language to potential keys in the template dict
        lang_key = self.language.lower()
        template = templates.get(lang_key)
        
        input_text = f"""
TASK ID: {task_id}
Problem Name: {task_name}
Problem: {record.get("question") or record.get("prompt") or "Unknown"}
Template:
```{lang_key}
#
# @lc app=leetcode id={task_id} lang={lang_key}
#
# [{task_id}] {task_name}
#
# @lc code=start
{template}
# @lc code=end
```
"""
        file_ext = self.LANGUAGE_CONFIG[self.language]["ext"]
        file_name = f"{task_id}.{inflection.parameterize(task_name)}"

        return Task(
            task_id=task_id,
            input=input_text,
            system_prompt=self.system_prompt,
            ground_truth=record.get("true_answer") or record.get("answer"),
            extra={
                "file_name": file_name,
                "file_ext": file_ext,
                "task_start_time": time.time()  # <--- 新增：记录任务开始时间
            }
        )

    async def eval(self, task: Task) -> Optional[Task]:
        """
        评测任务。使用 submit_lock 确保浏览器操作串行化，
        多个并发任务会在这里排队等待提交。
        
        时间记录：
        - inference_time: 推理耗时（由调用方设置）
        - submit_time: 浏览器提交耗时（本方法内计算）
        - spend_time: 总处理时间 = inference_time + submit_time
        """
        task_id = task.task_id
        code_content = task.result

        # ================= [使用 submit_lock 串行化浏览器操作] =================
        # 获取提交锁，确保同一时间只有一个任务在操作浏览器
        async with submit_lock:
            # ✅ 在获取锁后立即记录提交开始时间
            submit_start_time = time.time()
            task.extra["submit_start_time"] = submit_start_time
            
            logger.info(f"| 🔒 [Task {task_id}] Acquired submit lock, starting evaluation...")
            
            # 2. 确保 submitter 已初始化（理论上在 benchmark.initialize 已经完成）
            if not self._submitter_started:
                logger.info("| 🔍 Starting CodeSubmitter browser (delayed init)...")
                try:
                    await self._submitter.initialize()
                    self._submitter_started = True
                except Exception as e:
                    logger.error(f"| ❌ Failed to start submitter: {e}")
                    task.score = 0.0
                    self._tasks.append(task)
                    return task

            # 3. 处理无代码情况
            if not code_content:
                logger.error(f"| ❌ No code provided for Task {task_id}.")
                
                submit_end_time = time.time()
                submit_time = submit_end_time - submit_start_time
                inference_time = task.extra.get("inference_time", 0.0)
                spend_time = inference_time + submit_time
                
                # 更新 task.extra 中的时间信息
                task.extra["submit_time"] = submit_time
                task.extra["spend_time"] = spend_time

                task.score = 0.0
                self._tasks.append(task)
                
                # 构造错误状态的 Result 对象
                error_result = Result(
                    task_id=task_id,
                    prompt=task.input,
                    prediction="response_error",
                    answer="None",
                    score=0.0,
                    metrics={
                        "inference_time": inference_time,
                        "submit_time": submit_time
                    },
                    extra=None,
                    start_time=task.extra.get("inference_start_time", submit_start_time),
                    end_time=submit_end_time,
                    spend_time=spend_time
                )
                
                if self._submitter:
                    await self._submitter.save_result(error_result)
                    
                return task

            # 4. 正常评测流程
            try:
                file_name = f"{task.extra['file_name']}.{task.extra['file_ext']}"
                
                # 执行评测（浏览器操作）
                result_dict = await self._submitter.submit_code(code_content, file_name)
                
                # ✅ 计算提交耗时
                submit_end_time = time.time()
                submit_time = submit_end_time - submit_start_time
                inference_time = task.extra.get("inference_time", 0.0)
                spend_time = inference_time + submit_time
                
                # 更新 task.extra 中的时间信息
                task.extra["submit_time"] = submit_time
                task.extra["spend_time"] = spend_time
                task.extra["result"] = result_dict
                
                task.score = self._parse_result_score(result_dict)
                self._tasks.append(task)
                
                status_str = result_dict.get("status", "Unknown")
                
                # 将时间信息也加入 metrics
                result_dict["inference_time"] = inference_time
                result_dict["submit_time"] = submit_time
                
                # 构造正常流程的 Result 对象
                evaluation_result = Result(
                    task_id=task_id,
                    prompt=task.input,
                    prediction=status_str,
                    answer="None",
                    score=task.score,
                    metrics=result_dict,
                    extra=task.extra,
                    start_time=task.extra.get("inference_start_time", submit_start_time),
                    end_time=submit_end_time,
                    spend_time=spend_time
                )
                
                # 保存结果
                await self._submitter.save_result(evaluation_result)
                
                logger.info(f"| 🔓 [Task {task_id}] Evaluation complete (inference: {inference_time:.2f}s, submit: {submit_time:.2f}s)")
                return task

            except Exception as e:
                logger.error(f"| ❌ Submission error: {e}")
                import traceback
                traceback.print_exc()
                
                submit_end_time = time.time()
                submit_time = submit_end_time - submit_start_time
                inference_time = task.extra.get("inference_time", 0.0)
                spend_time = inference_time + submit_time
                
                error_result = Result(
                    task_id=task_id,
                    prompt=task.input,
                    prediction="system_error",
                    answer="None",
                    score=0.0,
                    metrics={
                        "error": str(e),
                        "inference_time": inference_time,
                        "submit_time": submit_time
                    },
                    extra=None,
                    start_time=task.extra.get("inference_start_time", submit_start_time),
                    end_time=submit_end_time,
                    spend_time=spend_time
                )
                await self._submitter.save_result(error_result)

                task.score = 0.0
                self._tasks.append(task)
                return task
        # ================= [锁作用域结束] =================
        
    async def stats(self) -> Optional[Stats]:
        total = len(self._data_records)
        attempted = len(self._tasks)
        correct = sum(1 for r in self._tasks if r.score and r.score >= 1.0)
        
        task_times = {r.task_id: r.time for r in self._tasks if r.time is not None}
        avg_time = sum(task_times.values()) / len(task_times) if task_times else 0.0
        
        return Stats(
            accuracy=correct / attempted if attempted > 0 else 0.0,
            total=total,
            correct=correct,
            wrong=attempted - correct,
            times=task_times,
            average_time=avg_time
        )

    def _parse_result_score(self, result: Dict[str, Any]) -> float:
        """
        Calculates score based on passed cases / total cases.
        """
        # 1. 直接从 result 获取我们在 browser 中解析好的整数
        # 使用 .get(key, 0) 防止键不存在报错
        passed = result.get("passed_cases", 0)
        total = result.get("total_cases", 0)

        # 2. 只要有总用例数，就按照比例计算分数 (涵盖了 Accepted, Wrong Answer, TLE 等)
        if total > 0:
            return float(passed) / float(total)

        # 3. 如果没有用例数据 (比如 total 为 0)
        status = result.get("status", "")
        
        # 如果是 Accepted 但没抓取到用例数，保底给 1.0
        if status == "Accepted":
            return 1.0
            
        # Compile Error 或其他无法解析的情况，给 0.0
        return 0.0
    
    async def cleanup(self):
        """Cleanup benchmark resources (close browser)."""
        if self._submitter_started and self._submitter:
            try:
                await self._submitter.close()
                self._submitter_started = False
                logger.info(f"| [{self.name}] 🚪 Browser closed successfully")
            except Exception as e:
                logger.warning(f"| [{self.name}] ⚠️ Error during browser cleanup: {e}")

    def __del__(self):
        if hasattr(self, '_submitter') and self._submitter_started:
            try:
                if hasattr(self._submitter, 'close'):
                    self._submitter.close()
            except Exception:
                pass