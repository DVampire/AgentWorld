import os
import time
import json
import re
import textwrap
import asyncio
import subprocess
import tempfile
import shutil
from typing import Optional, Dict, Any, Set, List, ClassVar
from pydantic import PrivateAttr, Field, ConfigDict
from playwright.sync_api import sync_playwright


from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK


SYSTEM_PROMPT = """
You are a helpful assistant that solves LeetCode coding problems. Please think step by step and provide your solution code.

Output format:
The output should be a JSON object with the following fields, DO NOT add any other text like "```json" or "```" or anything else:
{
    "reasoning": "Your step-by-step reasoning process",
    "code": "You solution code".
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
    "code": "#\\n# @lc app=leetcode id=1 lang=python3\\n#\\n# [1] Two Sum\\n#\\n\\n# @lc code=start\\nclass Solution:\\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\\n        hashmap = {}\\n        for i, num in enumerate(nums):\\n            complement = target - num\\n            if complement in hashmap:\\n                return [hashmap[complement], i]\\n            hashmap[num] = i\\n        return []\\n# @lc code=end"
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
        self.project_url = os.getenv("PROJECT_URL")
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE", '''gr_user_id=be5a137e-7528-4f14-b61c-4aa42990b6b2; 87b5a3c3f1a55520_gr_last_sent_cs1=9G1HZU0qIi; __gads=ID=de292e9c1cb6c106:T=1765938808:RT=1766543736:S=ALNI_MYlIoxT871eSPAyHWK_ZzIDpf4UWw; __gpi=UID=000011cca31639aa:T=1765938808:RT=1766543736:S=ALNI_MZ_wIJDgu0ssJV7kMXiHQNFnRFu_A; __eoi=ID=cf11d4e870725a42:T=1765938808:RT=1766543736:S=AA-AfjZe2WYyRLd9KO_CrSZ9-Cbn; __stripe_mid=bb795905-81bf-4fbe-ab32-6f2eab2953979d719d; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22e01e1a3e-d1ff-484a-921a-c935b7acb5d2%5C%22%2C%5B1765938808%2C764000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol8XJWFWLRqQoxn4V5NGNacqrf2axRoZbnSzjt9FDCgio7VGlMUmfAHmuJR3eCKQOGZhbvfkzI0MQ47cmZMD0vhDI74gCne94rKDpAW4vA4lX9gCRaKAhoODVKYoyq2RLlU103h4WsiMaw49tXS_4v66HRVsww%3D%3D%22%5D%5D; _gid=GA1.2.499499743.1767949572; _gat=1; ip_check=(false, "203.149.208.10"); 87b5a3c3f1a55520_gr_last_sent_cs1=cb3c8033-d028-4fc0-b728-d34c0ded5ba8; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=cb3c8033-d028-4fc0-b728-d34c0ded5ba8; 87b5a3c3f1a55520_gr_session_id_sent_vst=cb3c8033-d028-4fc0-b728-d34c0ded5ba8; cf_clearance=4XpweYq7AjSJxTC1LJNWUO6AxcSrBNR87kYg3r3Zmh8-1767949576-1.2.1.1-icsz6m3WwzdCzBs5MwrPyYbejFzfCAtVGEmcFgPA3ccivpDzY0EMN0oIxSBOpnYjxNvwWeujkL1yQBx_C2d_naSM56jK.1ZH654OB8ChEvSWA9Kyr1rnwWlkgy1AtCsATSyiTF3rWHGmajWygjg.8ybeCJRmLiGNIR6bcM1LpE46OavQGl4pzSvFa4kuIHZ4iE83emD2Yn_H2o2W9N14x0.OrZzcR576XeHZ9D4o0ks; csrftoken=TWzsxpzzuMjm9DfbmF2qGcdhzD6on7KN; messages=W1siX19qc29uX21lc3NhZ2UiLDAsMjUsIlN1Y2Nlc3NmdWxseSBzaWduZWQgaW4gYXMgWldULiIsIiJdXQ:1ve8Rq:SnFlIE7KwInIQOS-WtCeuJpqGhtT7Ejruq68HLahlBQ; _dd_s=rum=0&expire=1767950481494; INGRESSCOOKIE=6627697b7c459f78a36d8c57ae61cdf7|8e0876c7c1464cc0ac96bc2edceabd27; LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjAyNzQ5NDAiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI0OGQ3NDJkOTQ2MWY0NDQ5Mjg4ODE5ZWIyM2RkZWNmMGE3NTEzMGExZWU2M2NmZDNlZDY5MTQzMmNhNTY0MzBmIiwic2Vzc2lvbl91dWlkIjoiZmVmMThlMzgiLCJpZCI6MjAyNzQ5NDAsImVtYWlsIjoiemhhbmd3ZW50OTYzQGdtYWlsLmNvbSIsInVzZXJuYW1lIjoiOUcxSFpVMHFJaSIsInVzZXJfc2x1ZyI6IjlHMUhaVTBxSWkiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvOUcxSFpVMHFJaS9hdmF0YXJfMTc2NTkzODgwNS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3Njc5NDk1ODIsImlwIjoiMjAzLjE0OS4yMDguMTAiLCJpZGVudGl0eSI6IjY4NGZhYzNkOGU1OTU4NDU2NDBlNTA3YTkxMjJiZDU1IiwiZGV2aWNlX3dpdGhfaXAiOlsiYTlmY2E2MjVjYWJmZTFjZjNhNWQ4MDFmNGViODYwMzgiLCIyMDMuMTQ5LjIwOC4xMCJdLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9.BjMgGvZySenDbq9FFpq5r10NhEuphwYJ40OcnSxHFYk; 87b5a3c3f1a55520_gr_cs1=9G1HZU0qIi; _ga=GA1.1.1585026129.1765938626; _ga_CDRWKZTDEX=GS2.1.s1767949571$o11$g1$t1767949588$j43$l0$h0''')
        
        self.codespace_name = os.getenv("CODESPACE_NAME", "vigilant space spork")
        self.headless = headless

        if not all([self.username, self.password, self.project_url, self.leetcode_cookie]):
            raise ValueError("❌ Error: .env must contain: GITHUB_USERNAME, GITHUB_PASSWORD, PROJECT_URL, LEETCODE_COOKIE")

        self._playwright = None
        self._context = None
        self.page = None
        self._is_ready = False
        self._work_dir = None
        self._repo_path = None

    def start(self):
        """Setup git repo and browser -> Login GitHub -> Enter Codespace -> Verify LeetCode"""
        logger.info("[System] Starting evaluation environment...")
        
        # 1. Setup git repository
        self._setup_git_repo()
        
        # 2. Setup browser
        self._playwright = sync_playwright().start()
        user_data_dir = os.path.join(os.getcwd(), "playwright_user_data")
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=self.headless,
            viewport={'width': 1280, 'height': 800},
            args=["--disable-blink-features=AutomationControlled"],
            permissions=["clipboard-read", "clipboard-write"]
        )
        self.page = self._context.pages[0] if self._context.pages else self._context.new_page()
        
        # 3. Login GitHub and enter Codespace
        self._login_and_navigate()
        
        # 4. Verify LeetCode login status
        self._ensure_leetcode_login()
        
        self._is_ready = True
        logger.info("[System] ✅ Environment all ready, can start evaluation")

    def _setup_git_repo(self):
        """Create temp directory and clone the project"""
        logger.info(">> Setting up git repository...")
        
        # Create temporary directory
        self._work_dir = tempfile.mkdtemp(prefix="leetcode_submit_")
        logger.info(f">> Created temp directory: {self._work_dir}")
        
        # Extract repo name and owner from project_url
        # e.g., https://github.com/username/repo -> username/repo
        url_parts = self.project_url.rstrip('/').replace('https://github.com/', '').replace('http://github.com/', '')
        repo_name = url_parts.split('/')[-1]
        self._repo_path = os.path.join(self._work_dir, repo_name)
        
        # Convert to SSH format: git@github.com:username/repo.git
        ssh_url = f"git@github.com:{url_parts}.git"
        
        try:
            subprocess.run(
                ['git', 'clone', ssh_url, repo_name],
                cwd=self._work_dir,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f">> Successfully cloned repository to {self._repo_path}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout if e.stdout else "Unknown error"
            raise Exception(f"Failed to clone repository: {error_msg}")

    def _login_and_navigate(self):
        """Login GitHub and enter specified Codespace"""
        try:
            self.page.goto("https://github.com/login")
            time.sleep(1)
            if self.page.locator('input[name="login"]').count() > 0:
                logger.info(">> Logging in to GitHub...")
                self.page.fill('input[name="login"]', self.username)
                self.page.fill('input[name="password"]', self.password)
                self.page.click('input[type="submit"][value="Sign in"]')
                self.page.wait_for_url("https://github.com/", timeout=15000)
            
            logger.info(f">> Visiting project: {self.project_url}")
            self.page.goto(self.project_url)
            time.sleep(2)
            
            code_btn = self.page.locator('summary:has-text("Code"), button:has-text("Code")').first
            if code_btn.is_visible():
                code_btn.click()
            time.sleep(1)
            
            codespaces_tab = self.page.locator('[role="tab"]:has-text("Codespaces")')
            if codespaces_tab.is_visible():
                codespaces_tab.click()
            time.sleep(1)
            
            logger.info(f">> Searching for Codespace: {self.codespace_name} ...")
            target = self.page.get_by_text(self.codespace_name, exact=False).first
            target.wait_for(state="visible", timeout=10000)
            target.click()
            
            logger.info(">> Waiting for VS Code Web to load...")
            time.sleep(25)
            self.page.wait_for_selector('.monaco-workbench', timeout=120000)
            logger.info(">> VS Code load complete")
            
        except Exception as e:
            raise Exception(f"Failed to enter Codespace: {str(e)}")

    def _ensure_leetcode_login(self):
        """Check LeetCode plugin status, if not logged in then use Cookie to log in"""
        logger.info(">> Verifying LeetCode login status...")
        try:
            leetcode_icon = self.page.locator('li[aria-label="LeetCode"], .codicon-leetcode, a[title="LeetCode"]').first
            if leetcode_icon.is_visible():
                leetcode_icon.click()
                time.sleep(10)
            
            needs_login = self.page.get_by_text("Sign in to LeetCode", exact=False).is_visible()
            
            if needs_login:
                logger.info(">> Detected not logged in, using Cookie to log in...")
                self._perform_leetcode_login()
            else:
                logger.info(">> Looks like already logged in (or no login prompt detected).")
                time.sleep(5)
                
        except Exception as e:
            logger.warning(f"⚠️ Non-fatal error occurred during LeetCode login verification: {e}")

    def _perform_leetcode_login(self):
        self.page.keyboard.press("F1")
        time.sleep(1)
        self.page.keyboard.type("LeetCode: Sign In")
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        time.sleep(1)
        self.page.keyboard.press("Enter") 
        time.sleep(1)
        self.page.keyboard.type("Cookie")
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        time.sleep(1)
        
        logger.info(">> Entering Cookie...")
        json_cookie = json.dumps(self.leetcode_cookie)
        self.page.evaluate(f"navigator.clipboard.writeText({json_cookie})")
        time.sleep(0.2)
        self.page.keyboard.press("Meta+V") 
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        
        logger.info(">> Waiting for login to take effect...")
        time.sleep(5)

    def _close_all_editors(self):
        """Close all currently opened editor pages"""
        logger.info(">> Closing all opened editors...")
        # Press ESC first to prevent popups or menus from blocking
        self.page.keyboard.press("Escape")
        time.sleep(0.5)
        
        # Focus on page center to ensure keyboard events are captured
        self.page.mouse.click(600, 400)
        
        # Use Command Palette to execute Close All Editors
        self.page.keyboard.press("F1")
        time.sleep(1)
        self.page.keyboard.type("View: Close All Editors")
        time.sleep(0.8) # Wait for filtering
        self.page.keyboard.press("Enter")
        time.sleep(1.5) # Wait for animation to close

    def _open_editor_file(self, filename):
        """Open explorer and open code file"""
        logger.info(f">> Opening work file: {filename}")
        
        # Switch back to explorer
        explorer_icon = self.page.locator('li[aria-label="Explorer"], li[aria-label="资源管理器"], .codicon-files').first
        if explorer_icon.is_visible():
            explorer_icon.click()
            time.sleep(0.5)

        # Summon file search
        try:
            self.page.click('.monaco-editor', timeout=1000)
        except:
            self.page.mouse.click(500, 500)
            
        time.sleep(0.5)
        self.page.keyboard.press("Meta+P")
        time.sleep(0.5)
        self.page.keyboard.type(filename)
        time.sleep(1)
        self.page.keyboard.press("Enter")
        time.sleep(2)

    def submit_code(self, code_content: str, filename: Optional[str] = None) -> Dict[str, Any]:
        """Submit code using git workflow: write -> commit -> push -> open in codespace -> submit"""
        if not self._is_ready:
            raise RuntimeError("Please call .start() first")

        target_file = filename or "code_submit.py"
        logger.info(f"\n[Action] Preparing evaluation for {target_file}...")
        
        # 1. Write code to file in repo
        file_path = os.path.join(self._repo_path, target_file)
        clean_code = self._clean_code(code_content)
        
        logger.info(f">> Writing code to {file_path}...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_code)
        
        # 2. Git add, commit, and push
        logger.info(">> Committing and pushing to GitHub...")
        try:
            # Configure git user (required for commit)
            subprocess.run(
                ['git', 'config', 'user.name', self.username],
                cwd=self._repo_path,
                check=True,
                capture_output=True
            )
            subprocess.run(
                ['git', 'config', 'user.email', f'{self.username}@users.noreply.github.com'],
                cwd=self._repo_path,
                check=True,
                capture_output=True
            )
            
            # Git add
            result = subprocess.run(
                ['git', 'add', target_file],
                cwd=self._repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Check if there are changes to commit
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self._repo_path,
                check=True,
                capture_output=True,
                text=True
            )
            
            if status_result.stdout.strip():
                # Git commit
                subprocess.run(
                    ['git', 'commit', '-m', f'Add solution: {target_file}'],
                    cwd=self._repo_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Git push
                subprocess.run(
                    ['git', 'push'],
                    cwd=self._repo_path,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info(">> Successfully pushed to GitHub")
            else:
                logger.info(">> No changes to commit (file unchanged)")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
            logger.warning(f">> Git operation failed: {error_msg}")
            # Continue anyway, file might already be in repo or push might not be needed
        
        # 3. Wait a bit for GitHub to sync
        time.sleep(3)
        
        # 4. Refresh codespace and open file
        logger.info(">> Refreshing codespace and opening file...")
        self.page.reload()
        time.sleep(5)
        self._open_editor_file(target_file)
        time.sleep(2)
        
        # 5. Trigger LeetCode submission
        logger.info(">> Triggering LeetCode submission...")
        self.page.keyboard.press("Control+Shift+P")
        time.sleep(1)
        self.page.keyboard.type("LeetCode: Submit")
        time.sleep(1)
        self.page.keyboard.press("Enter")
        time.sleep(1)
        
        # 6. Wait for result
        result = self._wait_for_result()
        
        return result

    def _clean_code(self, code: str) -> str:
        code = code.strip()
        if "```" in code:
            match = re.search(r"```(?:python|python3)?(.*?)```", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
        return textwrap.dedent(code)

    def _wait_for_result(self) -> Dict[str, Any]:
        result_data = {"status": "Timeout", "details": []}
        end_time = time.time() + 60
        keywords = ["Accepted", "Wrong Answer", "Time Limit Exceeded", "Runtime Error","Memory Limit Exceeded", "Compile Error"]
        found = False
        
        logger.info(">> Waiting for evaluation results...")
        while time.time() < end_time and not found:
            for frame in self.page.frames:
                try:
                    for kw in keywords:
                        # Increase hierarchy judgment to prevent misjudgment of text in code
                        if frame.locator(f"h2:has-text('{kw}')").is_visible():
                            result_data["status"] = kw
                            # Try to get details
                            details = frame.locator("ul > li").all_inner_texts()
                            if not details: # Fallback: sometimes it's plain text
                                details = [frame.locator("body").inner_text()[:200]]
                            result_data["details"] = [d.strip() for d in details]
                            found = True
                            break
                except: continue
            if found: break
            time.sleep(1)
        
        logger.info(f"[Result] {result_data['status']}")
        return result_data

    def close(self):
        """Cleanup browser and temp directory"""
        if self._context: 
            self._context.close()
        if self._playwright: 
            self._playwright.stop()
        self._is_ready = False
        
        # Cleanup temp directory
        if self._work_dir and os.path.exists(self._work_dir):
            try:
                shutil.rmtree(self._work_dir)
                logger.info(f">> Cleaned up temp directory: {self._work_dir}")
            except Exception as e:
                logger.warning(f">> Failed to cleanup temp directory: {e}")

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
        "python3": {"ext": ".py", "lang_tag": "python3", "comment": "#"},
        "python": {"ext": ".py", "lang_tag": "python", "comment": "#"},
        "cpp": {"ext": ".cpp", "lang_tag": "cpp", "comment": "//"},
        "c++": {"ext": ".cpp", "lang_tag": "cpp", "comment": "//"},
        "java": {"ext": ".java", "lang_tag": "java", "comment": "//"},
        "javascript": {"ext": ".js", "lang_tag": "javascript", "comment": "//"},
        "typescript": {"ext": ".ts", "lang_tag": "typescript", "comment": "//"},
        "c": {"ext": ".c", "lang_tag": "c", "comment": "//"},
        "csharp": {"ext": ".cs", "lang_tag": "csharp", "comment": "//"},
        "c#": {"ext": ".cs", "lang_tag": "csharp", "comment": "//"},
        "go": {"ext": ".go", "lang_tag": "golang", "comment": "//"},
        "ruby": {"ext": ".rb", "lang_tag": "ruby", "comment": "#"},
        "swift": {"ext": ".swift", "lang_tag": "swift", "comment": "//"},
        "rust": {"ext": ".rs", "lang_tag": "rust", "comment": "//"},
        "scala": {"ext": ".scala", "lang_tag": "scala", "comment": "//"},
        "kotlin": {"ext": ".kt", "lang_tag": "kotlin", "comment": "//"},
        "php": {"ext": ".php", "lang_tag": "php", "comment": "//"},
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submitter = CodeSubmitter(headless=False)
        self._submitter_started = False
        os.makedirs("tmp", exist_ok=True)

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
            
        # 自动过滤掉已经跑完的任务
        self._filter_finished_tasks()
        await self.reset()

    def _filter_finished_tasks(self):
        """Filter out tasks that have already been completed (present in tmp/answer.jsonl)."""
        jsonl_path = os.path.join("tmp", "answer.jsonl")
        if not os.path.exists(jsonl_path):
            return
        
        finished_task_ids = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        task_id = str(entry.get("task_id", ""))
                        if task_id:
                            finished_task_ids.add(task_id)
                    except json.JSONDecodeError:
                        continue
            
            if finished_task_ids:
                original_count = len(self._data_records)
                self._data_records = [
                    record for record in self._data_records
                    if str(record.get("id") or record.get("task_id", "")) not in finished_task_ids
                ]
                filtered_count = len(self._data_records)
                logger.info(
                    f"[{self.name}] Filtered {original_count - filtered_count} finished tasks. "
                    f"Remaining: {filtered_count}"
                )
        except Exception as e:
            logger.warning(f"[{self.name}] ⚠️ Failed to filter finished tasks: {e}")

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
        
        # templates are in record
        templates = record.get("code_template", {})
        # Map our language to potential keys in the template dict
        lang_key = self.language.lower()
        template = templates.get(lang_key)
        
        input_text = f"""
TASK ID: {task_id}
Problem Name: {record.get("name") or record.get("problem_name") or "Unknown"}
Problem: {record.get("question") or record.get("prompt") or "Unknown"}
Template:
```{lang_key}
#
# @lc app=leetcode id={task_id} lang={lang_key}
#
# [{task_id}] {record.get("name") or record.get("problem_name") or "Unknown"}
#
# @lc code=start
{template}
# @lc code=end
```
"""
            
        return Task(
            task_id=task_id,
            input=input_text,
            system_prompt=self.system_prompt,
            ground_truth=record.get("true_answer") or record.get("answer"),
            extra={k: v for k, v in record.items() if k not in ["true_answer", "answer", "task_id", "id", "question", "prompt", "code_template"]}
        )

    async def eval(self, task: Task) -> Optional[Task]:
        prediction = str(task.prediction) if task.prediction is not None else ""
        task_id = task.task_id
        
        # Try to parse JSON format with reasoning and code fields
        reasoning = None
        code_content = None
        
        try:
            # Try to extract JSON from the prediction (handle cases with markdown code blocks)
            cleaned_prediction = prediction.strip()
            # Remove markdown code blocks if present
            cleaned_prediction = re.sub(r'```json\s*', '', cleaned_prediction)
            cleaned_prediction = re.sub(r'```\s*$', '', cleaned_prediction)
            
            # Try to find JSON object
            json_match = re.search(r'\{.*?"reasoning".*?"code".*?\}', cleaned_prediction, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                reasoning = parsed.get("reasoning")
                code_content = parsed.get("code")
        except (json.JSONDecodeError, AttributeError, KeyError):
            pass
        
        # If JSON parsing failed, try to extract code directly
        if not code_content:
            code_body = self._extract_inner_code(prediction)
            if code_body:
                code_content = code_body
        else:
            # Extract code from the code field (which should contain @lc markers)
            code_body = self._extract_inner_code(code_content)
            if code_body:
                code_content = code_body
        
        if not code_content:
            logger.error(f"[{task_id}] ❌ No code extracted.")
            task.score = 0.0
            task.reasoning = reasoning
            self._tasks.append(task)
            return task
        
        # Store reasoning if available
        if reasoning:
            task.reasoning = reasoning

        lang_cfg = self.LANGUAGE_CONFIG.get(self.language.lower(), self.LANGUAGE_CONFIG["python3"])
        comment = lang_cfg["comment"]
        ext = lang_cfg["ext"]
        lang_tag = lang_cfg["lang_tag"]

        name = "Unknown_Problem"
        if task_id in self._id_to_record_map:
            raw_name = self._id_to_record_map[task_id].get("name") or "Unknown"
            name = re.sub(r'[\\/*?:"<>| ]', '_', str(raw_name))

        # Construct customized filename: "task_id.name_slug.ext"
        name_slug = name.lower().replace("_", "-")
        custom_filename = f"{task_id}.{name_slug}{ext}"

        full_content = (
            f"{comment}\n"
            f"{comment} @lc app=leetcode id={task_id} lang={lang_tag}\n"
            f"{comment}\n"
            f"{comment} [{task_id}] {name}\n"
            f"{comment}\n\n"
            f"{comment} @lc code=start\n"
            f"{code_content}\n"
            f"{comment} @lc code=end\n"
        )

        if not self._submitter_started:
            logger.info("[LeetCodeBenchmark] Starting CodeSubmitter browser...")
            try:
                await asyncio.to_thread(self._submitter.start)
                self._submitter_started = True
            except Exception as e:
                logger.error(f"[{task_id}] ❌ Failed to start submitter: {e}")
                task.score = 0.0
                self._tasks.append(task)
                return task

        try:
            logger.info(f"[{task_id}] 🚀 Submitting code as {custom_filename}...")
            result = await asyncio.to_thread(self._submitter.submit_code, full_content, custom_filename)
            
            self._save_artifacts(task_id, name, full_content, result, custom_filename)

            score = self._parse_result_score(result)
            logger.info(f"[{task_id}] Result: {result.get('status')} | Score: {score:.2f}")
            task.score = score
            self._tasks.append(task)
            return task

        except Exception as e:
            logger.error(f"[{task_id}] ❌ Submission error: {e}")
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

    def _save_artifacts(self, task_id: str, task_name: str, code_content: str, result: Dict[str, Any], filename: Optional[str] = None):
        try:
            if filename:
                code_filename = os.path.join("tmp", filename)
            else:
                code_filename = os.path.join("tmp", f"{task_id}_{task_name}.py")
                
            with open(code_filename, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            status = result.get("status", "Unknown")
            details = result.get("details", [])
            # ... rest of the function remains same ...
            
            passed_cases = 0
            total_cases = 0
            runtime = -1.0
            memory_usage = -1.0
            runtime_beats = -1.0
            memory_beats = -1.0
            
            for line in details:
                if not isinstance(line, str): continue
                
                case_match = re.search(r"(\d+)/(\d+)\s+cases\s+passed", line)
                if case_match:
                    passed_cases = int(case_match.group(1))
                    total_cases = int(case_match.group(2))
                    
                    rt_match = re.search(r"\((\d+(?:\.\d+)?)\s*ms\)", line)
                    if rt_match:
                        runtime = float(rt_match.group(1))

                mem_match = re.search(r"\(([\d\.]+)\s*MB\)", line)
                if mem_match:
                    memory_usage = float(mem_match.group(1))
                    
                rt_beats_match = re.search(r"runtime beats\s+(\d+(?:\.\d+)?)\s*%", line)
                if rt_beats_match:
                    runtime_beats = float(rt_beats_match.group(1))
                
                mem_beats_match = re.search(r"memory usage beats\s+(\d+(?:\.\d+)?)\s*%", line)
                if mem_beats_match:
                    memory_beats = float(mem_beats_match.group(1))
            
            log_entry = {
                "task_id": task_id,
                "status": status,
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "runtime": runtime,
                "memory_usage": memory_usage,
                "runtime_beats": runtime_beats,
                "memory_beats": memory_beats
            }
            
            jsonl_path = os.path.join("tmp", "answer.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
            logger.info(f"[{task_id}] 💾 Stats saved: Runtime={runtime}, Mem={memory_usage}")

        except Exception as e:
            logger.warning(f"[{task_id}] ⚠️ Failed to save artifacts: {e}")

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

    def _extract_inner_code(self, text: str) -> Optional[str]:
        if not text: return None
        # Support both # and // comment styles for markers
        pattern = r"(?:#|//)\s*@lc code=start\s+(.*?)(?:#|//)\s*@lc code=end"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    async def cleanup(self):
        """Cleanup benchmark resources (close browser)."""
        if self._submitter_started and self._submitter:
            try:
                # CodeSubmitter.close is sync
                await asyncio.to_thread(self._submitter.close)
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