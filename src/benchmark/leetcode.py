import os
import time
import json
import re
import textwrap
import asyncio
from typing import Optional, Dict, Any, Set, List, ClassVar
from pydantic import PrivateAttr, Field
from playwright.sync_api import sync_playwright


from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK

class CodeSubmitter:
    def __init__(self, headless: bool = False):
        """
        Initialization: Automatically load configuration from environment variables
        """
        self.username = os.getenv("GITHUB_USERNAME")
        self.password = os.getenv("GITHUB_PASSWORD")
        self.project_url = os.getenv("PROJECT_URL")
        # Note: Keeping your original Cookie variable here, it's recommended to put it in .env for actual use
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE", '''gr_user_id=f5bc3375-7445-4b43-9dba-c6924bbcfbab; 87b5a3c3f1a55520_gr_last_sent_cs1=wingsjackf; __gads=ID=3314383dc9dbfd63:T=1765959508:RT=1765960725:S=ALNI_MZKM_0ftmbjBxPFmNH_MZ2VCbWhyw; __gpi=UID=000011cce2fa9e5c:T=1765959508:RT=1765960725:S=ALNI_MbcfJ91PkyeMjJ6PQSD5P84TL1iuQ; __eoi=ID=ca213cabdcdde30d:T=1765959508:RT=1765960725:S=AA-AfjZvByvqUBZM2BFwXNjTpLxd; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%223a4905a5-e4a0-4763-a6f1-5edb178b6c7e%5C%22%2C%5B1765959509%2C600000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol_nsKDRYW3BKFir-WrjYIGQzhRueop_DLCdXVdbKZkJlKLpqDk2dohSEl2k7xcVCcX_sD6WaoCw0at9T8TK80gwB9aB6cOadaFu6Co06Q3cJV4W5DqeaY3kTNhirm2PIkLSaA2Ct4WiTdfv2feHDzSnedsZyA%3D%3D%22%5D%5D; csrftoken=IT5nHJNNAfalbpaEayqDMreg9hA2nF1J; INGRESSCOOKIE=c913d62aa38b9134d1c42f708eb6ac49|8e0876c7c1464cc0ac96bc2edceabd27; ip_check=(false, "103.158.75.119"); LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjAyNzQ5NDAiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI0OGQ3NDJkOTQ2MWY0NDQ5Mjg4ODE5ZWIyM2RkZWNmMGE3NTEzMGExZWU2M2NmZDNlZDY5MTQzMmNhNTY0MzBmIiwic2Vzc2lvbl91dWlkIjoiYzE4N2FmNjciLCJpZCI6MjAyNzQ5NDAsImVtYWlsIjoiemhhbmd3ZW50OTYzQGdtYWlsLmNvbSIsInVzZXJuYW1lIjoiOUcxSFpVMHFJaSIsInVzZXJfc2x1ZyI6IjlHMUhaVTBxSWkiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvOUcxSFpVMHFJaS9hdmF0YXJfMTc2NTkzODgwNS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3Njc0MzIzNDAsImlwIjoiMTAzLjE1OC43NS4xMTkiLCJpZGVudGl0eSI6IjY4NGZhYzNkOGU1OTU4NDU2NDBlNTA3YTkxMjJiZDU1IiwiZGV2aWNlX3dpdGhfaXAiOlsiMDdjMGViOWFlZWVlNDExOTg2YzRlMGVkNzMxZDQ4ZDMiLCIxMDMuMTU4Ljc1LjExOSJdLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9._4L208MdmfadNb-Q7ahtAyC43KT68Aib46170PUcgZ0; 87b5a3c3f1a55520_gr_session_id=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; 87b5a3c3f1a55520_gr_session_id_sent_vst=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; _gid=GA1.2.292073357.1767432342; _gat=1; cf_clearance=j5hHTIUqMV7e6HwQcwdTbrbv0UIM1dcHIdhp00lmWFI-1767432349-1.2.1.1-rjaw_.2y8WCm.1oJ.mYBlnUUAaiXI_Ek4G1PbJfDFDQ5YgvBMsfftn98Rl.AukvwMJvEK2.YrRXt1EFDZJU4Crq9ENjAMe8ixUz5xhUjKQCFJcRJWtO47ocGv_8z9twEFRYnU7HsAqr2Le9CJkrxy_ltWgnZUDrWiXW39grOEBXCh0rCjPEsQEgE5CUsM1f.OjhTuYgInziAX6SzRDW7251QrzWglEHVuRbnfGKRn20; 87b5a3c3f1a55520_gr_cs1=wingsjackf; _ga_CDRWKZTDEX=GS2.1.s1767432342$o6$g1$t1767432377$j25$l0$h0; _ga=GA1.1.350590212.1765439998''')
        
        self.codespace_name = os.getenv("CODESPACE_NAME", "fuzzy memory") 
        self.headless = headless

        if not all([self.username, self.password, self.project_url, self.leetcode_cookie]):
            raise ValueError("❌ Error: .env must contain: GITHUB_USERNAME, GITHUB_PASSWORD, PROJECT_URL, LEETCODE_COOKIE")

        self._playwright = None
        self._context = None
        self.page = None
        self._is_ready = False
        self.target_filename = "code_submit.py" # Define target filename

    def start(self):
        """Start browser -> Login GitHub -> Enter Codespace -> Verify LeetCode -> Prepare environment"""
        logger.info("[System] Starting evaluation environment...")
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
        
        # 1. Login GitHub and enter Codespace
        self._login_and_navigate()
        
        # 2. Verify LeetCode login status
        self._ensure_leetcode_login()

        # 3. Don't force open file during initialization, leave it to submit_code or open once to confirm environment
        self._open_editor_file(self.target_filename)
        
        self._is_ready = True
        logger.info("[System] ✅ Environment all ready, can start evaluation")

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
        """Submit code and return result (includes pre and post cleanup logic)"""
        if not self._is_ready:
            raise RuntimeError("Please call .start() first")

        target_file = filename or self.target_filename
        logger.info(f"\n[Action] Preparing evaluation for {target_file}...")
        
        # 1. Before submission: Close all open Tabs
        self._close_all_editors()

        # 2. Re-open target file
        self._open_editor_file(target_file)
        
        # 3. Focus editor
        logger.info(">> Writing code...")
        try:
            self.page.click('.monaco-editor', timeout=2000)
        except:
            self.page.mouse.click(600, 400)

        # 4. Clean and paste code
        clean_code = self._clean_code(code_content)

        self.page.keyboard.press("Escape")
        time.sleep(0.1)
        self.page.keyboard.press("Meta+A")
        time.sleep(0.1)
        self.page.keyboard.press("Backspace")
        
        json_str = json.dumps(clean_code)
        self.page.evaluate(f"navigator.clipboard.writeText({json_str})")
        time.sleep(0.2)
        self.page.keyboard.press("Meta+V")
        time.sleep(0.5)
        self.page.keyboard.press("Meta+S") # Save
        
        # 5. Trigger submission
        logger.info(">> Triggering LeetCode submission...")
        time.sleep(0.5)
        self.page.keyboard.press("Control+Shift+P")
        time.sleep(1)
        
        # 6. Wait for result
        result = self._wait_for_result()

        # 7. After submission: After getting result, close all Tabs (cleanup scene)
        self._close_all_editors()
        
        return result

        # 7. After submission: After getting result, close all Tabs (cleanup scene)
        self._close_all_editors()
        
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
        if self._context: self._context.close()
        if self._playwright: self._playwright.stop()
        self._is_ready = False

@BENCHMARK.register_module(force=True)
class LeetCodeBenchmark(Benchmark):
    """
    LeetCode Benchmark with Resume Capability.
    Automatically filters out tasks present in 'tmp/answer.jsonl'.
    """
    name: str = "leetcode"
    path: str = "datasets/leetcode"
    language: str = Field(default="python3", description="Programming language for LeetCode (e.g., python3, cpp, java)")
    
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

    async def reset(self) -> Optional[Task]:
        self._index = 0
        self._tasks = []
        logger.info(f"| [{self.name}] ✅ Progress reset. Ready to start.")
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
        
        # Fallback for common aliases
        if not template:
            if lang_key == "python3": template = templates.get("python")
            elif lang_key == "python": template = templates.get("python3")
            elif lang_key in ["cpp", "c++"]: template = templates.get("cpp") or templates.get("c++")
        
        input_text = record.get("question") or record.get("prompt") or ""
        if template:
            input_text = (
                f"{input_text}\n\n"
                f"Please write the solution based on this template:\n"
                f"```{lang_key}\n{template}\n```"
            )
            
        return Task(
            task_id=task_id,
            input=input_text,
            system_prompt=self.get_task_description(),
            ground_truth=record.get("true_answer") or record.get("answer"),
            extra={k: v for k, v in record.items() if k not in ["true_answer", "answer", "task_id", "id", "question", "prompt", "code_template"]}
        )

    async def eval(self, task: Task) -> Optional[Task]:
        prediction = str(task.prediction) if task.prediction is not None else ""
        task_id = task.task_id
        
        code_body = self._extract_inner_code(prediction)
        if not code_body:
            logger.error(f"[{task_id}] ❌ No code extracted.")
            task.score = 0.0
            self._tasks.append(task)
            return task

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
            f"{code_body}\n"
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