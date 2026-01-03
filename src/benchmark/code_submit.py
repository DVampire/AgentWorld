"""
文件名: github_solver.py
功能: 封装 GitHub Codespace 评测工具 (包含 LeetCode Cookie 登录校验 + 页面自动清理逻辑)
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
        初始化：自动从环境变量加载配置
        """
        load_dotenv(verbose=True)
        
        self.username = os.getenv("GITHUB_USERNAME")
        self.password = os.getenv("GITHUB_PASSWORD")
        self.project_url = os.getenv("PROJECT_URL")
        # 注意：这里保留了你的原始 Cookie 变量，实际使用建议放入 .env
        self.leetcode_cookie = os.getenv("LEETCODE_COOKIE", '''gr_user_id=f5bc3375-7445-4b43-9dba-c6924bbcfbab; 87b5a3c3f1a55520_gr_last_sent_cs1=wingsjackf; __gads=ID=3314383dc9dbfd63:T=1765959508:RT=1765960725:S=ALNI_MZKM_0ftmbjBxPFmNH_MZ2VCbWhyw; __gpi=UID=000011cce2fa9e5c:T=1765959508:RT=1765960725:S=ALNI_MbcfJ91PkyeMjJ6PQSD5P84TL1iuQ; __eoi=ID=ca213cabdcdde30d:T=1765959508:RT=1765960725:S=AA-AfjZvByvqUBZM2BFwXNjTpLxd; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%223a4905a5-e4a0-4763-a6f1-5edb178b6c7e%5C%22%2C%5B1765959509%2C600000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol_nsKDRYW3BKFir-WrjYIGQzhRueop_DLCdXVdbKZkJlKLpqDk2dohSEl2k7xcVCcX_sD6WaoCw0at9T8TK80gwB9aB6cOadaFu6Co06Q3cJV4W5DqeaY3kTNhirm2PIkLSaA2Ct4WiTdfv2feHDzSnedsZyA%3D%3D%22%5D%5D; csrftoken=IT5nHJNNAfalbpaEayqDMreg9hA2nF1J; INGRESSCOOKIE=c913d62aa38b9134d1c42f708eb6ac49|8e0876c7c1464cc0ac96bc2edceabd27; ip_check=(false, "103.158.75.119"); LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjAyNzQ5NDAiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI0OGQ3NDJkOTQ2MWY0NDQ5Mjg4ODE5ZWIyM2RkZWNmMGE3NTEzMGExZWU2M2NmZDNlZDY5MTQzMmNhNTY0MzBmIiwic2Vzc2lvbl91dWlkIjoiYzE4N2FmNjciLCJpZCI6MjAyNzQ5NDAsImVtYWlsIjoiemhhbmd3ZW50OTYzQGdtYWlsLmNvbSIsInVzZXJuYW1lIjoiOUcxSFpVMHFJaSIsInVzZXJfc2x1ZyI6IjlHMUhaVTBxSWkiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvOUcxSFpVMHFJaS9hdmF0YXJfMTc2NTkzODgwNS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3Njc0MzIzNDAsImlwIjoiMTAzLjE1OC43NS4xMTkiLCJpZGVudGl0eSI6IjY4NGZhYzNkOGU1OTU4NDU2NDBlNTA3YTkxMjJiZDU1IiwiZGV2aWNlX3dpdGhfaXAiOlsiMDdjMGViOWFlZWVlNDExOTg2YzRlMGVkNzMxZDQ4ZDMiLCIxMDMuMTU4Ljc1LjExOSJdLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9._4L208MdmfadNb-Q7ahtAyC43KT68Aib46170PUcgZ0; 87b5a3c3f1a55520_gr_session_id=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; 87b5a3c3f1a55520_gr_session_id_sent_vst=f3d4efff-188b-4d79-b5fa-d6fe979a5c29; _gid=GA1.2.292073357.1767432342; _gat=1; cf_clearance=j5hHTIUqMV7e6HwQcwdTbrbv0UIM1dcHIdhp00lmWFI-1767432349-1.2.1.1-rjaw_.2y8WCm.1oJ.mYBlnUUAaiXI_Ek4G1PbJfDFDQ5YgvBMsfftn98Rl.AukvwMJvEK2.YrRXt1EFDZJU4Crq9ENjAMe8ixUz5xhUjKQCFJcRJWtO47ocGv_8z9twEFRYnU7HsAqr2Le9CJkrxy_ltWgnZUDrWiXW39grOEBXCh0rCjPEsQEgE5CUsM1f.OjhTuYgInziAX6SzRDW7251QrzWglEHVuRbnfGKRn20; 87b5a3c3f1a55520_gr_cs1=wingsjackf; _ga_CDRWKZTDEX=GS2.1.s1767432342$o6$g1$t1767432377$j25$l0$h0; _ga=GA1.1.350590212.1765439998''')
    
        self.codespace_name = os.getenv("CODESPACE_NAME", "fuzzy memory") 
        self.headless = headless

        if not all([self.username, self.password, self.project_url, self.leetcode_cookie]):
            raise ValueError("❌ 错误: .env 必须包含: GITHUB_USERNAME, GITHUB_PASSWORD, PROJECT_URL, LEETCODE_COOKIE")

        self._playwright = None
        self._context = None
        self.page = None
        self._is_ready = False
        self.target_filename = "code_submit.py" # 定义目标文件名

    def start(self):
        """启动浏览器 -> 登录 GitHub -> 进入 Codespace -> 校验 LeetCode -> 准备环境"""
        print("[System] 正在启动评测环境...")
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
        
        # 1. 登录 GitHub 并进入 Codespace
        self._login_and_navigate()
        
        # 2. 校验 LeetCode 登录状态
        self._ensure_leetcode_login()

        # 3. 初始化时不强制打开文件，留给 submit_code 处理，或者打开一次确认环境
        self._open_editor_file(self.target_filename)
        
        self._is_ready = True
        print("[System] ✅ 环境全部就绪，可以开始评测")

    def _login_and_navigate(self):
        """登录 GitHub 并进入指定的 Codespace"""
        try:
            self.page.goto("https://github.com/login")
            time.sleep(1)
            if self.page.locator('input[name="login"]').count() > 0:
                print(">> 正在登录 GitHub...")
                self.page.fill('input[name="login"]', self.username)
                self.page.fill('input[name="password"]', self.password)
                self.page.click('input[type="submit"][value="Sign in"]')
                self.page.wait_for_url("https://github.com/", timeout=15000)
            
            print(f">> 正在访问项目: {self.project_url}")
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
            
            print(f">> 正在查找 Codespace: {self.codespace_name} ...")
            target = self.page.get_by_text(self.codespace_name, exact=False).first
            target.wait_for(state="visible", timeout=10000)
            target.click()
            
            print(">> 等待 VS Code Web 加载...")
            time.sleep(25)
            self.page.wait_for_selector('.monaco-workbench', timeout=120000)
            print(">> VS Code 加载完成")
            
        except Exception as e:
            raise Exception(f"进入 Codespace 失败: {str(e)}")

    def _ensure_leetcode_login(self):
        """检查 LeetCode 插件状态，如果未登录则使用 Cookie 登录"""
        print(">> 正在校验 LeetCode 登录状态...")
        try:
            leetcode_icon = self.page.locator('li[aria-label="LeetCode"], .codicon-leetcode, a[title="LeetCode"]').first
            if leetcode_icon.is_visible():
                leetcode_icon.click()
                time.sleep(10)
            
            needs_login = self.page.get_by_text("Sign in to LeetCode", exact=False).is_visible()
            
            if needs_login:
                print(">> 检测到未登录，正在使用 Cookie 登录...")
                self._perform_leetcode_login()
            else:
                print(">> 看起来已登录 (或未检测到登录提示)。")
                time.sleep(5)
                
        except Exception as e:
            print(f"⚠️ LeetCode 登录校验过程出现非致命错误: {e}")

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
        
        print(">> 正在输入 Cookie...")
        json_cookie = json.dumps(self.leetcode_cookie)
        self.page.evaluate(f"navigator.clipboard.writeText({json_cookie})")
        time.sleep(0.2)
        self.page.keyboard.press("Meta+V") 
        time.sleep(0.5)
        self.page.keyboard.press("Enter")
        
        print(">> 等待登录生效...")
        time.sleep(5) 

    def _close_all_editors(self):
        """[新增] 关闭当前所有打开的编辑器页面"""
        print(">> 正在关闭所有已打开的编辑器...")
        # 先按一下 ESC，防止有弹窗或菜单遮挡
        self.page.keyboard.press("Escape")
        time.sleep(0.5)
        
        # 聚焦到页面中心确保键盘事件被捕获
        self.page.mouse.click(600, 400)
        
        # 使用 Command Palette 执行 Close All Editors
        self.page.keyboard.press("F1")
        time.sleep(1)
        self.page.keyboard.type("View: Close All Editors")
        time.sleep(0.8) # 等待筛选
        self.page.keyboard.press("Enter")
        time.sleep(1.5) # 等待动画关闭

    def _open_editor_file(self, filename):
        """打开资源管理器并打开代码文件"""
        print(f">> 正在打开工作文件: {filename}")
        
        # 切换回资源管理器
        explorer_icon = self.page.locator('li[aria-label="Explorer"], li[aria-label="资源管理器"], .codicon-files').first
        if explorer_icon.is_visible():
            explorer_icon.click()
            time.sleep(0.5)

        # 呼出文件搜索
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
        """提交代码并返回结果 (包含前后清理逻辑)"""
        if not self._is_ready:
            raise RuntimeError("请先调用 .start()")

        print("\n[Action] 准备评测...")
        
        # 1. [新增] 提交前：关闭所有打开的 Tab
        self._close_all_editors()

        # 2. [修改] 重新打开目标文件
        self._open_editor_file(self.target_filename)
        
        # 3. 聚焦编辑器
        print(">> 正在写入代码...")
        try:
            self.page.click('.monaco-editor', timeout=2000)
        except:
            self.page.mouse.click(600, 400)

        # 4. 清洗与粘贴代码
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
        self.page.keyboard.press("Meta+S") # 保存
        
        # 5. 触发提交
        print(">> 触发 LeetCode 提交...")
        time.sleep(0.5)
        self.page.keyboard.press("Control+Shift+P")
        time.sleep(1)
        
        # 6. 等待结果
        result = self._wait_for_result()

        # 7. [新增] 提交后：拿到结果后，关闭所有 Tab (清理现场)
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
        
        print(">> 等待评测结果...")
        while time.time() < end_time and not found:
            for frame in self.page.frames:
                try:
                    for kw in keywords:
                        # 增加层级判断，防止误判代码文本中的文字
                        if frame.locator(f"h2:has-text('{kw}')").is_visible():
                            result_data["status"] = kw
                            # 尝试获取详情
                            details = frame.locator("ul > li").all_inner_texts()
                            if not details: # 备用：有时候是纯文本
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