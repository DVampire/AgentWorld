"""
GitHub自动化脚本 - 使用Playwright
功能：自动登录GitHub，进入Codespaces，登录LeetCode并爬取指定范围的题目保存为jsonl
"""
from markdownify import markdownify as md
from playwright.sync_api import Playwright, sync_playwright, expect
import time
import asyncio
import json # 新增：用于处理JSON数据
import re   # 新增：用于正则提取ID和名称
from dotenv import load_dotenv
load_dotenv(verbose=True)
import os

class GitHubAutomation:
    def __init__(self, username: str, password: str, project_url: str, leetcode_cookie: str = ""):
        self.username = username
        self.password = password
        self.project_url = project_url
        self.leetcode_cookie = leetcode_cookie
        self.browser = None
        self.page = None
        self.context = None
        self.leetcode_logged_in_via_cookie = False
    
    def start_browser(self, playwright: Playwright, headless: bool = False):
        """启动浏览器并持久化配置"""
        print("正在启动浏览器...")
        user_data_dir = os.path.join(os.getcwd(), "playwright_user_data")
        self.context = playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            viewport={'width': 1280, 'height': 800},
            args=["--disable-blink-features=AutomationControlled"]
        )
        if len(self.context.pages) > 0:
            self.page = self.context.pages[0]
        else:
            self.page = self.context.new_page()
        print(f"浏览器启动成功！数据保存在: {user_data_dir}")
    
    # ... [之前的 navigate_to_github, click_sign_in, login, navigate_to_project, click_code_button, enter_codespaces, open_leetcode_plugin, login_leetcode 保持不变] ...
    # 为了节省篇幅，这里省略重复的登录和导航代码，请保留你原有的这部分代码
    
    def navigate_to_github(self):
        print("正在访问 GitHub...")
        self.page.goto("https://github.com")
        time.sleep(2)

    def click_sign_in(self):
        # ... (保持原有代码)
        try:
            sign_in_button = self.page.locator('a:text("Sign in")').first
            sign_in_button.click(timeout=5000)
        except:
            self.page.goto("https://github.com/login")

    def login(self):
        # ... (保持原有代码)
        print("正在登录...")
        try:
            if self.page.locator('input[name="login"]').count() == 0:
                return
            self.page.locator('input[name="login"]').fill(self.username)
            self.page.locator('input[name="password"]').fill(self.password)
            self.page.locator('input[type="submit"][value="Sign in"]').click()
            time.sleep(5)
        except Exception as e:
            print(f"登录异常: {e}")

    def navigate_to_project(self):
        self.page.goto(self.project_url)
        time.sleep(3)

    def click_code_button(self):
        # ... (保持原有代码)
        try:
            code_button = self.page.locator('button[data-variant="primary"]:has-text("Code")').first
            code_button.click()
            time.sleep(2)
        except:
            pass # 省略备用方案代码以聚焦核心逻辑

    def enter_codespaces(self, codespace_name="fuzzy memory"):
        # ... (保持原有代码)
        # 这里建议直接使用最稳妥的等待逻辑
        print("进入Codespaces逻辑...")
        try:
            # 简化版逻辑，请使用你原有的详细逻辑
            self.page.locator('[role="tab"]:has-text("Codespaces")').click()
            time.sleep(2)
            self.page.get_by_text(codespace_name, exact=False).first.click()
            print("等待Codespace启动（25秒）...")
            time.sleep(25)
        except Exception as e:
            print(f"进入Codespaces错误: {e}")

    def open_leetcode_plugin(self):
        # ... (保持原有代码)
        print("打开插件...")
        try:
            self.page.locator('a[aria-label="LeetCode"]').first.click()
            time.sleep(3)
        except:
            pass

    def login_leetcode(self):
        # ... (保持原有代码，逻辑不变)
        print("登录 LeetCode...")
        # 假设这里包含了你原本完整的登录逻辑
        pass 

    def click_all_problems(self):
        """点击"All"按钮显示所有题目"""
        print("正在点击All按钮以展开题目列表...")
        try:
            # 尝试点击 All 节点，如果已经展开可能不需要点，但为了保险点击一下
            # VS Code Tree view 中的 All
            all_btn = self.page.locator('div[role="treeitem"] >> text="All"').first
            if all_btn.is_visible():
                all_btn.click()
            else:
                # 备用选择器
                self.page.locator('span:has-text("All")').first.click()
            
            print("已点击All按钮")
            time.sleep(3)
        except Exception as e:
            print(f"点击All按钮时遇到小问题 (可能已展开): {e}")

    def _sanitize_filename(self, text: str) -> str:
        """
        处理文件名：
        1. 转小写
        2. 将空格替换为下划线
        3. 去除文件名中的非法字符 (如 / \ : * ? " < > |)
        Example: "Two Sum" -> "two_sum"
        """
        # 转小写
        text = text.lower()
        # 替换空格为下划线
        text = text.replace(" ", "_")
        # 使用正则去除非法字符，只保留字母、数字、下划线、点、中文
        text = re.sub(r'[^\w\u4e00-\u9fa5\._-]', '', text)
        return text

    # 替换原本的 crawl_leetcode_problems 方法
    def close_all_editors(self):
        """
        强制关闭所有编辑器，防止标签页堆积
        """
        print("正在执行: 关闭所有编辑器...")
        try:
            # 聚焦到页面任意位置
            self.page.mouse.click(100, 100)
            
            # 打开命令面板
            self.page.keyboard.press("F1")
            time.sleep(0.5)
            self.page.keyboard.type("View: Close All Editors")
            time.sleep(0.5)
            self.page.keyboard.press("Enter")
            
            # 给一点时间让 UI 反应
            time.sleep(1)
        except Exception as e:
            print(f"关闭编辑器命令执行失败: {e}")

    def _get_max_visible_id(self):
        """
        [辅助方法] 获取当前屏幕侧边栏中可见的最大的题目 ID
        """
        try:
            # 获取所有可见的 treeitem 文本
            # 我们只获取当前 viewport 内可见的元素
            visible_items = self.page.locator('[role="treeitem"]').all_text_contents()
            
            max_id = -1
            # 正则匹配 [数字]
            pattern = re.compile(r'\[(\d+)\]')
            
            for item_text in visible_items:
                match = pattern.search(item_text)
                if match:
                    current_id = int(match.group(1))
                    if current_id > max_id:
                        max_id = current_id
            
            return max_id
        except Exception as e:
            print(f"获取最大ID失败: {e}")
            return -1

    def crawl_leetcode_problems(self, start_id: int = 1, end_id: int = 10, output_jsonl: str = "leetcode_index.jsonl"):
        """
        [智能跳过版] 解决ID不连续导致错过题目、过度滚动的问题
        """
        print(f"\n开始爬取题目 [{start_id}] 到 [{end_id}] ...")
        
        base_dir = "question"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        # 1. 读取已存在 ID
        existing_ids = set()
        if os.path.exists(output_jsonl):
            with open(output_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        if "id" in data:
                            existing_ids.add(int(data["id"]))
                    except: continue
        else:
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                pass

        success_count = 0 
            
        for i in range(start_id, end_id + 1):
            try:
                if i in existing_ids:
                    print(f"⏩ 第 {i} 题已存在，跳过。")
                    continue

                print(f"\n--- 正在处理第 {i} 题 ---")
                self.close_all_editors()
                
                # ========================================================
                # 2. [核心修改] 智能查找与跳过逻辑
                # ========================================================
                problem_text_pattern = re.compile(f"\\[{i}\\]\\s")
                problem_item = None
                found = False
                
                # 最多尝试滚动查找 10 次
                max_scroll_attempts = 20
                
                for attempt in range(max_scroll_attempts):
                    # A. 先尝试直接定位
                    locator = self.page.locator(f'[role="treeitem"]').filter(has_text=problem_text_pattern).first
                    
                    if locator.count() > 0 and locator.is_visible():
                        problem_item = locator
                        found = True
                        print(f"✅ 找到题目 [{i}]")
                        break
                    
                    # B. 如果没找到，判断是否需要滚动
                    # 获取当前屏幕上能看到的最大 ID
                    current_max_id = self._get_max_visible_id()
                    print(f"当前屏幕最大ID: {current_max_id}, 目标ID: {i}")
                    
                    if current_max_id > i:
                        # [关键]：如果屏幕上已经出现了比目标大的 ID (比如显示了 160)，
                        # 而我们还在找 159 且没找到，说明 159 根本不存在（不连续）。
                        print(f"⚠️ 题目 [{i}] 不存在或已被跳过 (当前已至 [{current_max_id}])。停止搜索。")
                        found = False
                        break # 跳出滚动循环，直接去处理下一个 i (比如 160)
                    
                    # C. 只有当 current_max_id < i 时，才说明还没滚到位，需要继续滚
                    print(f"向下滚动查找 (尝试 {attempt+1})...")
                    tree_container = self.page.locator('div[role="tree"]').first
                    if tree_container.is_visible():
                        tree_container.hover()
                        self.page.mouse.wheel(0, 600)
                        time.sleep(1.0) # 等待加载
                    else:
                        break

                if not found:
                    print(f"⏭️ 未找到题目 [{i}]，跳过处理下一题。")
                    continue # 这里 continue 是去执行 for i 循环的下一次迭代
                
                # ========================================================
                # 下面的逻辑保持不变
                # ========================================================
                
                # 确保可见再点击
                if problem_item:
                    problem_item.scroll_into_view_if_needed()
                    full_text = problem_item.text_content()
                    print(f"目标题目: {full_text}")
                    
                    problem_name_raw = "unknown"
                    if f"[{i}]" in full_text:
                        parts = full_text.split(f"[{i}]")
                        if len(parts) > 1:
                            # 记得加上之前的去图标逻辑
                            problem_name_raw = parts[1].strip().replace("🔓", "").replace("🔒", "").strip()
                    
                    problem_item.click()
                
                # 循环校验内容
                print("等待 Webview 更新...")
                html_content = ""
                deadline = time.time() + 10
                
                while time.time() < deadline:
                    for frame in self.page.frames:
                        try:
                            # 1. 获取 frame 文本
                            body_text = frame.locator('body').inner_text()
                            
                            if "Subscribe to unlock" in body_text:
                                html_content = "PREMIUM_LOCKED"
                                break

                            # 2. 关键词初筛
                            if frame.locator('text="Example 1:"').count() > 0 or \
                               frame.locator('text="Description"').count() > 0 or \
                               frame.locator('h1').count() > 0:
                                
                                # ================================================================
                                # [核心修复] 终极标准化：去除所有格式差异 (大小写、符号、多余空格)
                                # ================================================================
                                
                                # 定义一个简单的清理函数（内部 lambda）
                                # 作用：转小写 -> 统一引号 -> 将所有连续空白(空格/换行/tab)替换为单空格
                                normalize = lambda s: " ".join(s.lower().replace("’", "'").replace("‘", "'").split())
                                
                                target_name_clean = normalize(problem_name_raw)
                                page_content_clean = normalize(body_text)
                                
                                # 3. 进行模糊匹配
                                if target_name_clean in page_content_clean:
                                    html_content = frame.locator('body').inner_html()
                                    print(f"✅ 内容匹配成功")
                                    break
                                else:
                                    # 调试用：如果没匹配上，看看清理后长什么样
                                    # print(f"DEBUG: Expect '{target_name_clean}' not in content")
                                    pass

                        except: continue
                    
                    if html_content: break
                    print(".", end="", flush=True)
                    time.sleep(1)
                
                print("") 
                
                if not html_content or html_content == "PREMIUM_LOCKED":
                    print(f"❌ 跳过内容 (付费或获取失败)")
                    self.page.keyboard.press("Control+W")
                    continue
                
                # 保存
                markdown_content = md(html_content, heading_style="ATX")
                markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
                header = f"# {i}. {problem_name_raw}\n\n"
                final_md = header + markdown_content
                
                safe_name = self._sanitize_filename(problem_name_raw)
                file_name = f"{i}.{safe_name}.md"
                file_path = os.path.join(base_dir, file_name)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_md)
                
                relative_path = f"./{base_dir}/{file_name}"
                index_data = {"id": i, "name": problem_name_raw, "file": relative_path}
                
                with open(output_jsonl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(index_data, ensure_ascii=False) + "\n")
                
                print(f"💾 第 {i} 题已保存")
                success_count += 1
                
            except Exception as e:
                print(f"❌ 异常: {e}")
                try: self.page.keyboard.press("Control+W")
                except: pass
        
        print(f"\n任务结束！本次新增保存 {success_count} 个文件。")


    def close_browser(self):
        if self.browser:
            self.browser.close()
        elif self.context:
            self.context.close()
        print("浏览器已关闭")

    def run(self, playwright: Playwright, headless: bool = False):
        try:
            # 1. 启动浏览器
            self.start_browser(playwright, headless)
            
            # 2. 访问 GitHub 并登录
            self.navigate_to_github()
            self.click_sign_in() 
            self.login()
            
            # 3. 访问项目并进入 Codespaces
            self.navigate_to_project()
            self.click_code_button()
            self.enter_codespaces()
            
            # 4. 打开 LeetCode 插件
            self.open_leetcode_plugin()
            self.login_leetcode()
            
            # 5. 确保题目列表可见 (点击 All)
            self.click_all_problems()
            
            # 6. 【核心任务】爬取题目 [1] 到 [10]
            # --- 修改点在这里：将 output_file 改为 output_jsonl ---
            self.crawl_leetcode_problems(
                start_id=1, 
                end_id=3787, 
                output_jsonl="leetcode_index.jsonl"
            )
            
            # 7. 截图留念
            #self.page.screenshot(path="final_status.png")
            
        except Exception as e:
            print(f"\n执行过程中出错: {e}")
            self.page.screenshot(path="error_global.png")
        finally:
            time.sleep(5)
            self.close_browser()

def main():
    # 配置区域
    USERNAME = os.getenv("GITHUB_USERNAME")
    PASSWORD = os.getenv("GITHUB_PASSWORD")
    PROJECT_URL = os.getenv("PROJECT_URL")
    LEETCODE_COOKIE = '''gr_user_id=f5bc3375-7445-4b43-9dba-c6924bbcfbab; 87b5a3c3f1a55520_gr_last_sent_cs1=wingsjackf; __gads=ID=3314383dc9dbfd63:T=1765959508:RT=1765960725:S=ALNI_MZKM_0ftmbjBxPFmNH_MZ2VCbWhyw; __gpi=UID=000011cce2fa9e5c:T=1765959508:RT=1765960725:S=ALNI_MbcfJ91PkyeMjJ6PQSD5P84TL1iuQ; __eoi=ID=ca213cabdcdde30d:T=1765959508:RT=1765960725:S=AA-AfjZvByvqUBZM2BFwXNjTpLxd; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%223a4905a5-e4a0-4763-a6f1-5edb178b6c7e%5C%22%2C%5B1765959509%2C600000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol_nsKDRYW3BKFir-WrjYIGQzhRueop_DLCdXVdbKZkJlKLpqDk2dohSEl2k7xcVCcX_sD6WaoCw0at9T8TK80gwB9aB6cOadaFu6Co06Q3cJV4W5DqeaY3kTNhirm2PIkLSaA2Ct4WiTdfv2feHDzSnedsZyA%3D%3D%22%5D%5D; 87b5a3c3f1a55520_gr_session_id=7bc4c39d-dc90-44e7-8d68-128068f77c4b; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=7bc4c39d-dc90-44e7-8d68-128068f77c4b; 87b5a3c3f1a55520_gr_session_id_sent_vst=7bc4c39d-dc90-44e7-8d68-128068f77c4b; INGRESSCOOKIE=b08110f5697829b692af77fc7e655968|8e0876c7c1464cc0ac96bc2edceabd27; _gid=GA1.2.326707289.1766546661; ip_check=(false, "103.158.75.119"); cf_clearance=QQQdfOA13ryP5XJM6_p6Mr7fNndlTfxZBvZX_KSYCK0-1766546679-1.2.1.1-Q7JRTBHOGNcfLOxO12t4KAR0o1OOYcOYozKWxAnKUWzt8u4XZy1iXJykQGmhRfOtWrEeA3ipoBRQo_wAQxFPsFPR0Ht1I88L1bomLz7dPg73qqb6V404TRJreO0aZmNWFk2zbNyhFtPbFjyKmTj.GleHNqOcB.Zmln4nd_OkR3DcBYotJjdWahbUT1FG147K0TIkF3fTXMjHCAlSBEZPjevAnhx5VREmUYHGgZJ8OjE; csrftoken=IT5nHJNNAfalbpaEayqDMreg9hA2nF1J; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFT0lFSitXBpSO4NDkZKJJWmpNTCdOVmaeQWKxQnpmXXpyVmJydRsAIaloaFR4C0RsLAL2BSrc:1vYFV3:4lfGSK2-Vrt_Skd_g5zB8nhg4Mwh-MEHJJ5H9tgwwCg; LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjAyNzQ5NDAiLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI0OGQ3NDJkOTQ2MWY0NDQ5Mjg4ODE5ZWIyM2RkZWNmMGE3NTEzMGExZWU2M2NmZDNlZDY5MTQzMmNhNTY0MzBmIiwic2Vzc2lvbl91dWlkIjoiYzE4N2FmNjciLCJpZCI6MjAyNzQ5NDAsImVtYWlsIjoiemhhbmd3ZW50OTYzQGdtYWlsLmNvbSIsInVzZXJuYW1lIjoiOUcxSFpVMHFJaSIsInVzZXJfc2x1ZyI6IjlHMUhaVTBxSWkiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvOUcxSFpVMHFJaS9hdmF0YXJfMTc2NTkzODgwNS5wbmciLCJyZWZyZXNoZWRfYXQiOjE3NjY1NDY3MjEsImlwIjoiMTAzLjE1OC43NS4xMTkiLCJpZGVudGl0eSI6IjY4NGZhYzNkOGU1OTU4NDU2NDBlNTA3YTkxMjJiZDU1IiwiZGV2aWNlX3dpdGhfaXAiOlsiMDdjMGViOWFlZWVlNDExOTg2YzRlMGVkNzMxZDQ4ZDMiLCIxMDMuMTU4Ljc1LjExOSJdLCJfc2Vzc2lvbl9leHBpcnkiOjEyMDk2MDB9.bS2c4rcWHuqBjJcAQpvgiNQmTtHqQrMu9jm_dj-gPYg; _dd_s=rum=0&expire=1766547619940; 87b5a3c3f1a55520_gr_cs1=wingsjackf; _gat=1; _ga=GA1.1.350590212.1765439998; _ga_CDRWKZTDEX=GS2.1.s1766546651$o3$g1$t1766546726$j59$l0$h0''' # 建议放入环境变量
    
    # 你的 LeetCode Cookie
    # LEETCODE_COOKIE = "你的cookie字符串"

    print("="*50)
    print("LeetCode 题目爬虫脚本 (Codespaces版)")
    print("="*50)
    
    automation = GitHubAutomation(USERNAME, PASSWORD, PROJECT_URL, LEETCODE_COOKIE)
    
    with sync_playwright() as playwright:
        automation.run(playwright, headless=False) # 建议先用 False 观察运行情况

if __name__ == "__main__":
    main()