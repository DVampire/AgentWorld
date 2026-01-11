import os
import time
import json
import re
import inflection
import asyncio
from typing import Optional, Dict, Any, Set, List, ClassVar
from pydantic import PrivateAttr, Field, ConfigDict
from playwright.async_api import async_playwright


from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.config import config
from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK


SYSTEM_PROMPT = """
You are a helpful assistant that solves LeetCode coding problems. Please think step by step and provide your solution code.

Output format:
The output should be a JSON object with the following fields, DO NOT add any other text like "```json" or "```" or anything else:
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
        Uses git workflow: clone -> write file -> commit -> push -> open in codespace -> submit
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
            
            # According to the provided HTML, the button is an 'a' tag with aria-label="LeetCode"
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
        await self.page.keyboard.press("Meta+V") 
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Enter")
        
        logger.info("| 🔍 Waiting for login to take effect...")
        await asyncio.sleep(30)

    async def _open_editor_file(self, filename):
        """Open explorer and open code file"""
        logger.info(f"| 🔍 Opening work file: {filename}")
        
        # Switch back to explorer
        explorer_icon = self.page.locator('li[aria-label="Explorer"], li[aria-label="资源管理器"], .codicon-files').first
        if await explorer_icon.is_visible():
            await explorer_icon.click()
            await asyncio.sleep(0.5)

        # Summon file search
        try:
            await self.page.click('.monaco-editor', timeout=1000)
        except:
            await self.page.mouse.click(500, 500)
            
        await asyncio.sleep(0.5)
        await self.page.keyboard.press("Meta+P")
        await asyncio.sleep(0.5)
        await self.page.keyboard.type(filename)
        await asyncio.sleep(1)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(2)

    async def submit_code(self, code_content: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Submit code using git workflow: write -> commit -> push -> open in codespace -> submit"""

        target_file = filename or "code.py"
        logger.info(f"| 🔍 Preparing evaluation for {target_file}...")
        
        # 1. Write code to file in repo
        file_path = os.path.join(self.repo_path, target_file)
        
        logger.info(f"| 🔍 Writing code to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # 2. Git add, commit, and push
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
        
        # 4. Open file
        await self._open_editor_file(target_file)
        await asyncio.sleep(2)
        
        # 5. Trigger LeetCode submission
        logger.info("| 🔍 Triggering LeetCode submission...")
        await self.page.keyboard.press("Meta+Shift+P")
        await asyncio.sleep(1)
        await self.page.keyboard.type("LeetCode: Submit to LeetCode")
        await asyncio.sleep(1)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(1)
        
        # 6. Wait for result
        result = await self._wait_for_result()
        
        return result

    async def _wait_for_result(self) -> Dict[str, Any]:
        result_data = {"status": "Timeout", "details": []}
        end_time = time.time() + 60
        keywords = ["Accepted", "Wrong Answer", "Time Limit Exceeded", "Runtime Error","Memory Limit Exceeded", "Compile Error"]
        found = False
        
        logger.info("| 🔍 Waiting for evaluation results...")
        while time.time() < end_time and not found:
            for frame in self.page.frames:
                try:
                    for kw in keywords:
                        # Increase hierarchy judgment to prevent misjudgment of text in code
                        if await frame.locator(f"h2:has-text('{kw}')").is_visible():
                            result_data["status"] = kw
                            # Try to get details
                            details = await frame.locator("ul > li").all_inner_texts()
                            if not details: # Fallback: sometimes it's plain text
                                details = [await frame.locator("body").inner_text()]
                                if details[0]:
                                    details[0] = details[0][:200]
                            result_data["details"] = [d.strip() for d in details]
                            found = True
                            break
                except: continue
            if found: break
            await asyncio.sleep(1)
        
        logger.info(f"| 🔍 Result: {result_data['status']}")
        return result_data

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
        """Initialize the benchmark by loading dataset and filtering finished tasks."""
        try:
            from src.data.leetcode import LeetCodeDataset
            dataset = LeetCodeDataset(
                path=self.path,
                split=self.split,
                name=self.subset if self.subset else None
            )
            self._id_to_record_map = {}
            if hasattr(dataset, 'data'):
                self._data_records = dataset.data.to_dict(orient="records")
                for record in self._data_records:
                    tid = str(record.get("id") or record.get("task_id", "0"))
                    self._id_to_record_map[tid] = record
            logger.info(f"[{self.name}] Index built: {len(self._id_to_record_map)} records mapped.")
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
            }
        )

    async def eval(self, task: Task) -> Optional[Task]:
        task_id = task.task_id
        code_content = task.result

        if not code_content:
            logger.error(f"| ❌ No code provided.")
            task.score = 0.0
            self._tasks.append(task)
            return task

        if not self._submitter_started:
            logger.info("| 🔍 Starting CodeSubmitter browser...")
            try:
                await self._submitter.initialize()
                self._submitter_started = True
            except Exception as e:
                logger.error(f"| ❌ Failed to start submitter: {e}")
                task.score = 0.0
                self._tasks.append(task)
                return task

        try:
            file_name = f"{task.extra['file_name']}.{task.extra['file_ext']}"
            result = await self._submitter.submit_code(code_content, file_name)
            task.extra["result"] = result
            task.score = self._parse_result_score(result)
            self._tasks.append(task)
            return task

        except Exception as e:
            logger.error(f"| ❌ Submission error: {e}")
            import traceback
            traceback.print_exc()
            task.score = 0.0
            self._tasks.append(task)
            return task

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
        status = result.get("status", "")
        details = result.get("details", [])

        if status == "Accepted":
            return 1.0
        if status == "Compile Error":
            return 0.0
        
        for line in details:
            if isinstance(line, str):
                match = re.search(r"(\d+)/(\d+)\s+cases\s+passed", line)
                if match:
                    passed = int(match.group(1))
                    total = int(match.group(2))
                    return passed / total if total > 0 else 0.0
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