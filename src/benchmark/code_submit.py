"""
File Name: github_solver.py
Function: Encapsulates GitHub Codespace evaluation tool (includes LeetCode Cookie login verification + automatic page cleanup logic)
"""

import os
import time
import json
import re
import textwrap
from typing import Dict, Any
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

class CodeSubmitter:
    def __init__(self, headless: bool = False):
        """
        Initialization: Automatically load configuration from environment variables
        """
        load_dotenv(verbose=True)
        
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
        print("[System] Starting evaluation environment...")
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
        print("[System] ✅ Environment all ready, can start evaluation")

    def _login_and_navigate(self):
        """Login GitHub and enter specified Codespace"""
        try:
            self.page.goto("https://github.com/login")
            time.sleep(1)
            if self.page.locator('input[name="login"]').count() > 0:
                print(">> Logging in to GitHub...")
                self.page.fill('input[name="login"]', self.username)
                self.page.fill('input[name="password"]', self.password)
                self.page.click('input[type="submit"][value="Sign in"]')
                self.page.wait_for_url("https://github.com/", timeout=15000)
            
            print(f">> Visiting project: {self.project_url}")
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
            
            print(f">> Searching for Codespace: {self.codespace_name} ...")
            target = self.page.get_by_text(self.codespace_name, exact=False).first
            target.wait_for(state="visible", timeout=10000)
            target.click()
            
            print(">> Waiting for VS Code Web to load...")
            time.sleep(25)
            self.page.wait_for_selector('.monaco-workbench', timeout=120000)
            print(">> VS Code load complete")
            
        except Exception as e:
            raise Exception(f"Failed to enter Codespace: {str(e)}")

    def _ensure_leetcode_login(self):
        """Check LeetCode plugin status, if not logged in then use Cookie to log in"""
        print(">> Verifying LeetCode login status...")
        try:
            leetcode_icon = self.page.locator('li[aria-label="LeetCode"], .codicon-leetcode, a[title="LeetCode"]').first
            if leetcode_icon.is_visible():
                leetcode_icon.click()
                time.sleep(10)
            
            needs_login = self.page.get_by_text("Sign in to LeetCode", exact=False).is_visible()
            
            if needs_login:
                print(">> Detected not logged in, using Cookie to log in...")
                self._perform_leetcode_login()
            else:
                print(">> Looks like already logged in (or no login prompt detected).")
                time.sleep(5)
                
        except Exception as e:
            print(f"⚠️ Non-fatal error occurred during LeetCode login verification: {e}")

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
        
        print(">> Entering Cookie...")
        json_cookie = json.dumps(self.leetcode_cookie)
        self.page.evaluate(f"navigator.clipboard.writeText({json_cookie})")
        time.sleep(0.2)
        self.page.keyboard.press("Meta+V") 
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        
        print(">> Waiting for login to take effect...")
        time.sleep(5) 

    def _close_all_editors(self):
        """Close all currently opened editor pages"""
        print(">> Closing all opened editors...")
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
        print(f">> Opening work file: {filename}")
        
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

    def submit_code(self, code_content: str) -> Dict[str, Any]:
        """Submit code and return result (includes pre and post cleanup logic)"""
        if not self._is_ready:
            raise RuntimeError("Please call .start() first")

        print("\n[Action] Preparing evaluation...")
        
        # 1. Before submission: Close all open Tabs
        self._close_all_editors()

        # 2. Re-open target file
        self._open_editor_file(self.target_filename)
        
        # 3. Focus editor
        print(">> Writing code...")
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
        print(">> Triggering LeetCode submission...")
        time.sleep(0.5)
        self.page.keyboard.press("Control+Shift+P")
        time.sleep(1)
        
        # 6. Wait for result
        result = self._wait_for_result()

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
        
        print(">> Waiting for evaluation results...")
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
        
        print(f"[Result] {result_data['status']}")
        return result_data

    def close(self):
        if self._context: self._context.close()
        if self._playwright: self._playwright.stop()
        self._is_ready = False
