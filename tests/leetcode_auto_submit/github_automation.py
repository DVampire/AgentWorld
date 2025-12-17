"""
GitHub自动化脚本 - 使用Playwright
功能：自动登录GitHub并导航到指定项目的codebase
"""

from playwright.sync_api import Playwright, sync_playwright, expect
import time
import asyncio
from dotenv import load_dotenv
load_dotenv(verbose=True)
import os

class GitHubAutomation:
    def __init__(self, username: str, password: str, project_url: str, leetcode_cookie: str = ""):
        """
        初始化GitHub自动化类
        
        Args:
            username: GitHub用户名或邮箱
            password: GitHub密码
            project_url: 目标项目的URL (例如: https://github.com/username/repo)
            leetcode_cookie: LeetCode的cookie用于登录
        """
        self.username = username
        self.password = password
        self.project_url = project_url
        self.leetcode_cookie = leetcode_cookie
        self.browser = None
        self.page = None
        # 标记本次是否是通过 Cookie 完成了 LeetCode 登录
        self.leetcode_logged_in_via_cookie = False
    
    def start_browser(self, playwright: Playwright, headless: bool = False):
        """
        启动浏览器并持久化配置（保存快捷键、登录状态等）
        """
        print("正在启动浏览器...")
        user_data_dir = os.path.join(os.getcwd(), "playwright_user_data")
        self.context = playwright.chromium.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=headless,
        viewport={'width': 1280, 'height': 800},
        # 建议添加以下参数以提高兼容性，模拟真实用户环境
        args=[
            "--disable-blink-features=AutomationControlled", # 减少被检测为机器人的概率
        ]
    )
    
    # 3. 持久化模式下，通常会自动创建一个页面，或者手动创建
        if len(self.context.pages) > 0:
            self.page = self.context.pages[0]
        else:
            self.page = self.context.new_page()
        
        print(f"浏览器启动成功！数据保存在: {user_data_dir}")
    
    def navigate_to_github(self):
        """导航到GitHub主页"""
        print("正在访问 GitHub...")
        self.page.goto("https://github.com")
        time.sleep(2)
        print("已到达 GitHub 主页")
    
    def click_sign_in(self):
        """点击登录按钮或直接导航到登录页面"""
        print("正在查找登录按钮...")
        try:
            # 方法1: 尝试点击右上角的Sign in按钮
            sign_in_button = self.page.locator('a:text("Sign in")').first
            # 设置较短的超时时间，如果按钮不可见就用备用方案
            sign_in_button.click(timeout=5000)
            print("已点击 Sign in 按钮")
            time.sleep(2)
        except Exception as e:
            print(f"点击按钮失败，直接导航到登录页面: {e}")
            # 方法2: 直接导航到登录页面（更可靠）
            self.page.goto("https://github.com/login")
            print("已直接导航到登录页面")
            time.sleep(2)
    
    def login(self):
        """
        执行登录操作
        如果当前页面上已经没有登录表单，则认为已经登录，直接跳过
        """
        print("正在登录...")
        try:
            # 如果页面上没有登录表单，说明很可能已经登录，直接跳过
            login_inputs = self.page.locator('input[name="login"]')
            password_inputs = self.page.locator('input[name="password"]')
            if login_inputs.count() == 0 or password_inputs.count() == 0:
                print("未发现 GitHub 登录表单，推断已登录，跳过登录步骤")
                return

            # 输入用户名/邮箱
            username_input = login_inputs.first
            username_input.fill(self.username)
            print(f"已输入用户名: {self.username}")
            
            # 输入密码
            password_input = password_inputs.first
            password_input.fill(self.password)
            print("已输入密码")
            
            # 点击登录按钮
            login_button = self.page.locator('input[type="submit"][value="Sign in"]').first
            login_button.click()
            print("已点击登录提交按钮")
            
            # 等待登录完成（可能需要处理2FA）
            time.sleep(5)
            
            # 检查是否登录成功
            if "login" not in self.page.url:
                print("登录成功！")
            else:
                print("警告: 可能需要处理双因素认证(2FA)或其他验证")
                # 等待用户手动处理2FA
                print("如果需要2FA验证，请在浏览器中手动完成...")
                time.sleep(20)
                
        except Exception as e:
            print(f"登录失败: {e}")
            raise
    
    def navigate_to_project(self):
        """导航到指定的项目"""
        print(f"正在访问项目: {self.project_url}")
        self.page.goto(self.project_url)
        time.sleep(3)
        print("已到达项目页面")
    
    def click_code_button(self):
        """点击绿色的Code按钮"""
        print("正在查找并点击Code按钮...")
        try:
            # GitHub项目页面上绿色的Code按钮
            # 使用 data-variant="primary" 来识别绿色按钮
            code_button = self.page.locator('button[data-variant="primary"]:has-text("Code")').first
            code_button.wait_for(state="visible", timeout=10000)
            code_button.click()
            print("已点击Code按钮")
            time.sleep(2)
        except Exception as e:
            print(f"方法1失败: {e}")
            # 备用方案：使用 exact=True 精确匹配
            try:
                code_button = self.page.get_by_role("button", name="Code", exact=True)
                code_button.click()
                print("使用备用方式点击了Code按钮")
                time.sleep(2)
            except Exception as e2:
                print(f"方法2失败: {e2}")
                # 第三种方案：直接用CSS选择器匹配绿色按钮
                try:
                    code_button = self.page.locator('.prc-Button-ButtonBase-9n-Xk[data-variant="primary"]').first
                    code_button.click()
                    print("使用方法3点击了Code按钮")
                    time.sleep(2)
                except Exception as e3:
                    print(f"方法3也失败: {e3}")
    
    def enter_codespaces(self, codespace_name: str = "fuzzy memory"):
        """
        进入Codespaces
        点击Code按钮后，选择Codespaces标签，然后点击指定的codespace
        
        Args:
            codespace_name: 要进入的codespace名称，默认为"fuzzy memory"
        """
        print("正在进入Codespaces...")
        try:
            # 等待弹窗出现
            time.sleep(1)
            
            # 点击Codespaces标签
            codespaces_tab = self.page.locator('[role="tab"]:has-text("Codespaces"), button:has-text("Codespaces")').first
            codespaces_tab.wait_for(state="visible", timeout=10000)
            codespaces_tab.click()
            print("已点击Codespaces标签")
            time.sleep(2)
            
            # 截图查看当前状态
            self.page.screenshot(path="codespaces_tab_clicked.png")
            print("已截图: codespaces_tab_clicked.png")
            
            # 查找并点击名为 "fuzzy memory" 的现有codespace
            print(f"正在查找Codespace: {codespace_name}")
            try:
                # 尝试多种方式查找codespace
                # 方式1: 通过文本查找
                codespace_link = self.page.locator(f'a:has-text("{codespace_name}"), button:has-text("{codespace_name}"), [title*="{codespace_name}"]').first
                codespace_link.wait_for(state="visible", timeout=10000)
                codespace_link.click()
                print(f"已点击Codespace: {codespace_name}")
            except Exception as e1:
                print(f"方式1失败: {e1}")
                try:
                    # 方式2: 查找codespace列表中的链接
                    codespace_item = self.page.locator(f'li:has-text("{codespace_name}") a, div:has-text("{codespace_name}") a').first
                    codespace_item.click()
                    print(f"已通过方式2点击Codespace: {codespace_name}")
                except Exception as e2:
                    print(f"方式2失败: {e2}")
                    try:
                        # 方式3: 直接查找包含codespace名称的任何可点击元素
                        self.page.get_by_text(codespace_name, exact=False).first.click()
                        print(f"已通过方式3点击Codespace: {codespace_name}")
                    except Exception as e3:
                        print(f"方式3失败: {e3}")
                        # 截图保存当前状态以便调试
                        self.page.screenshot(path="codespace_not_found.png")
                        print("已保存截图: codespace_not_found.png")
            
            # 等待Codespace加载（30秒）
            print("等待Codespace启动（30秒）...")
            time.sleep(30)
            
            current_url = self.page.url
            print(f"当前页面URL: {current_url}")
            print("Codespaces访问完成")
            
        except Exception as e:
            print(f"进入Codespaces失败: {e}")
            self.page.screenshot(path="codespaces_error.png")
            print("已保存错误截图: codespaces_error.png")
    
    def access_codebase(self):
        """
        访问项目的codebase（浏览文件）
        """
        print("正在访问代码库...")
        try:
            current_url = self.page.url
            print(f"当前页面URL: {current_url}")
            time.sleep(3)
            print("代码库访问完成")
        except Exception as e:
            print(f"访问代码库失败: {e}")
    
    def open_leetcode_plugin(self):
        """
        点击左侧侧边栏的LeetCode插件按钮
        """
        print("正在打开LeetCode插件...")
        try:
            # 尝试多种方式查找LeetCode按钮
            # 方式1: 通过aria-label查找
            leetcode_btn = self.page.locator('a[aria-label="LeetCode"], [aria-label="LeetCode"]').first
            leetcode_btn.wait_for(state="visible", timeout=15000)
            leetcode_btn.click()
            print("已点击LeetCode插件按钮")
            time.sleep(3)
        except Exception as e1:
            print(f"方式1失败: {e1}")
            try:
                # 方式2: 通过class包含leetcode查找
                leetcode_btn = self.page.locator('[class*="leetcode"], [class*="LeetCode"]').first
                leetcode_btn.click()
                print("已通过方式2点击LeetCode按钮")
                time.sleep(3)
            except Exception as e2:
                print(f"方式2失败: {e2}")
                try:
                    # 方式3: 通过activity-bar中的按钮查找
                    leetcode_btn = self.page.locator('.action-label[aria-label="LeetCode"]').first
                    leetcode_btn.click()
                    print("已通过方式3点击LeetCode按钮")
                    time.sleep(3)
                except Exception as e3:
                    print(f"方式3也失败: {e3}")
                    self.page.screenshot(path="leetcode_btn_not_found.png")
                    print("已保存截图: leetcode_btn_not_found.png")
    
    def login_leetcode(self):
        """
        登录LeetCode
        1. 点击 "Sign in to LeetCode"
        2. 选择使用cookie登录
        3. 输入cookie
        4. 等待登录成功
        """
        print("正在登录LeetCode...")
        try:
            # 0. 先判断是否已经登录：如果已经能看到题目列表或 All 节点，就直接跳过
            try:
                time.sleep(2)
                all_node = self.page.locator(
                    'span:has-text("All"), .monaco-highlighted-label:has-text("All"), [role="treeitem"]:has-text("All")'
                ).first
                if all_node and all_node.is_visible():
                    print("检测到 LeetCode 题目列表 / All 按钮已可见，推断已登录 LeetCode，跳过 Cookie 登录步骤")
                    # 已是登录状态，但不是本次通过 Cookie 登录
                    self.leetcode_logged_in_via_cookie = False
                    return
            except Exception:
                # 忽略检测错误，继续正常登录流程
                pass

            # 步骤1: 点击 "Sign in to LeetCode"
            print("正在查找Sign in to LeetCode按钮...")
            try:
                sign_in_btn = self.page.locator('a:has-text("Sign in to LeetCode"), span:has-text("Sign in to LeetCode"), [title*="Sign in"]').first
                sign_in_btn.wait_for(state="visible", timeout=10000)
                sign_in_btn.click()
                print("已点击Sign in to LeetCode")
                time.sleep(2)
            except Exception as e1:
                print(f"方式1失败: {e1}")
                try:
                    # 备用方式
                    self.page.get_by_text("Sign in to LeetCode", exact=False).first.click()
                    print("已通过备用方式点击Sign in")
                    time.sleep(2)
                except Exception as e2:
                    print(f"备用方式也失败: {e2}")
            
            # 步骤2: 选择使用cookie登录
            print("正在选择Cookie登录方式...")
            time.sleep(1)
            try:
                # 查找cookie选项（通常显示为 "LeetCode Cookie" 或类似文本）
                cookie_option = self.page.locator(':has-text("Cookie"), :has-text("cookie")').first
                cookie_option.wait_for(state="visible", timeout=10000)
                cookie_option.click()
                print("已选择Cookie登录方式")
                time.sleep(2)
            except Exception as e:
                print(f"选择cookie方式失败: {e}")
                # 可能会出现快速选择菜单，尝试按键盘
                try:
                    self.page.keyboard.type("cookie")
                    time.sleep(1)
                    self.page.keyboard.press("Enter")
                    print("已通过键盘选择Cookie方式")
                    time.sleep(2)
                except:
                    pass
            
            # 步骤3: 输入cookie
            print("正在输入Cookie...")
            time.sleep(1)
            try:
                # 查找输入框并输入cookie
                # VS Code的输入框通常是 input.input 或 .quick-input-box input
                cookie_input = self.page.locator('input.input, .quick-input-box input, input[type="text"]').first
                cookie_input.wait_for(state="visible", timeout=10000)
                cookie_input.fill(self.leetcode_cookie)
                print("已输入Cookie")
                time.sleep(1)
                
                # 按回车确认
                self.page.keyboard.press("Enter")
                print("已按回车确认")
            except Exception as e:
                print(f"输入cookie失败: {e}")
                # 尝试直接在页面上按键输入
                try:
                    self.page.keyboard.type(self.leetcode_cookie)
                    time.sleep(1)
                    self.page.keyboard.press("Enter")
                    print("已通过键盘直接输入Cookie并确认")
                except Exception as e2:
                    print(f"键盘输入也失败: {e2}")
            
            # 步骤4: 等待登录成功
            print("等待LeetCode登录成功（10秒）...")
            time.sleep(10)
            
            self.page.screenshot(path="leetcode_login_completed.png")
            print("LeetCode登录流程完成")
            print("已保存截图: leetcode_login_completed.png")
            # 标记：本次是通过 Cookie 完成的登录
            self.leetcode_logged_in_via_cookie = True
            
        except Exception as e:
            print(f"LeetCode登录失败: {e}")
            self.page.screenshot(path="leetcode_login_error.png")
            print("已保存错误截图: leetcode_login_error.png")
    
    def click_all_problems(self):
        """
        点击"All"按钮显示所有题目
        """
        print("正在点击All按钮...")
        try:
            # 方式1: 通过文本查找All按钮
            all_btn = self.page.locator('span:has-text("All"), .monaco-highlighted-label:has-text("All")').first
            all_btn.wait_for(state="visible", timeout=10000)
            all_btn.click()
            print("已点击All按钮")
            time.sleep(3)
        except Exception as e1:
            print(f"方式1失败: {e1}")
            try:
                # 方式2: 通过get_by_text查找
                all_btn = self.page.get_by_text("All", exact=True).first
                all_btn.click()
                print("已通过方式2点击All按钮")
                time.sleep(3)
            except Exception as e2:
                print(f"方式2失败: {e2}")
                try:
                    # 方式3: 查找树形视图中的All节点
                    all_btn = self.page.locator('[role="treeitem"]:has-text("All"), .tree-item:has-text("All")').first
                    all_btn.click()
                    print("已通过方式3点击All按钮")
                    time.sleep(3)
                except Exception as e3:
                    print(f"方式3也失败: {e3}")
                    self.page.screenshot(path="all_btn_not_found.png")
                    print("已保存截图: all_btn_not_found.png")
    
    def click_first_problem(self):
        """
        点击第一道题目
        题目格式: [1] Two Sum
        """
        print("正在点击第一道题目...")
        try:
            # 等待题目列表加载
            time.sleep(3)
            
            # 方式1: 查找题目列表中的 "[1] Two Sum"
            # 题目格式是 [序号] 题目名
            first_problem = self.page.locator('[role="treeitem"]:has-text("[1]"), .monaco-list-row:has-text("[1]")').first
            first_problem.wait_for(state="visible", timeout=10000)
            first_problem.click()
            print("已点击第一道题目 [1] Two Sum")
            time.sleep(3)
        except Exception as e1:
            print(f"方式1失败: {e1}")
            try:
                # 方式2: 直接通过文本 "[1] Two Sum" 查找
                first_problem = self.page.get_by_text("[1] Two Sum", exact=False).first
                first_problem.click()
                print("已通过方式2点击第一道题目")
                time.sleep(3)
            except Exception as e2:
                print(f"方式2失败: {e2}")
                try:
                    # 方式3: 查找左侧面板中包含 "[1]" 的元素
                    first_problem = self.page.locator('.pane-body :has-text("[1]"), .sidebar :has-text("[1]")').first
                    first_problem.click()
                    print("已通过方式3点击第一道题目")
                    time.sleep(3)
                except Exception as e3:
                    print(f"方式3失败: {e3}")
                    try:
                        # 方式4: 双击第一个列表项
                        first_problem = self.page.locator('.monaco-list-row').first
                        first_problem.dblclick()
                        print("已通过方式4双击第一道题目")
                        time.sleep(3)
                    except Exception as e4:
                        print(f"方式4也失败: {e4}")
                        self.page.screenshot(path="first_problem_not_found.png")
                        print("已保存截图: first_problem_not_found.png")
    
    def click_explorer_icon(self):
        """
        点击左上角的文件图标（资源管理器）查看项目文件
        """
        print("正在点击文件图标...")
        try:
            # 方式1: 通过aria-label查找资源管理器图标
            explorer_btn = self.page.locator('[aria-label="资源管理器"], [aria-label="Explorer"], .codicon-files').first
            explorer_btn.wait_for(state="visible", timeout=10000)
            explorer_btn.click()
            print("已点击资源管理器图标")
            time.sleep(2)
        except Exception as e1:
            print(f"方式1失败: {e1}")
            try:
                # 方式2: 查找侧边栏第一个图标
                explorer_btn = self.page.locator('.activitybar .action-item').first
                explorer_btn.click()
                print("已通过方式2点击资源管理器")
                time.sleep(2)
            except Exception as e2:
                print(f"方式2失败: {e2}")
                try:
                    # 方式3: 使用键盘快捷键 Ctrl+Shift+E
                    self.page.keyboard.down("Control")
                    self.page.keyboard.down("Shift")
                    self.page.keyboard.press("E")
                    self.page.keyboard.up("Shift")
                    self.page.keyboard.up("Control")
                    print("已通过快捷键Ctrl+Shift+E打开资源管理器")
                    time.sleep(2)
                except Exception as e3:
                    print(f"方式3也失败: {e3}")
    
    def open_code_file(self, problem_name: str = "two_sum"):
        """
        打开与题目名称匹配的代码文件
        
        Args:
            problem_name: 题目名称（用于匹配文件名），默认为"two_sum"
        """
        print(f"正在打开代码文件: {problem_name}...")
        try:
            # 等待文件列表加载
            time.sleep(2)
            
            # 方式1: 查找包含题目名称的文件
            code_file = self.page.locator(f'[role="treeitem"]:has-text("{problem_name}"), .monaco-list-row:has-text("{problem_name}")').first
            code_file.wait_for(state="visible", timeout=10000)
            code_file.click()
            print(f"已点击代码文件: {problem_name}")
            time.sleep(2)
        except Exception as e1:
            print(f"方式1失败: {e1}")
            try:
                # 方式2: 通过get_by_text查找
                code_file = self.page.get_by_text(problem_name, exact=False).first
                code_file.click()
                print(f"已通过方式2点击代码文件")
                time.sleep(2)
            except Exception as e2:
                print(f"方式2失败: {e2}")
                try:
                    # 方式3: 使用Ctrl+P快速打开文件
                    self.page.keyboard.down("Control")
                    self.page.keyboard.press("P")
                    self.page.keyboard.up("Control")
                    time.sleep(1)
                    self.page.keyboard.type(problem_name)
                    time.sleep(1)
                    self.page.keyboard.press("Enter")
                    print(f"已通过Ctrl+P打开文件")
                    time.sleep(2)
                except Exception as e3:
                    print(f"方式3也失败: {e3}")
                    self.page.screenshot(path="code_file_not_found.png")
                    print("已保存截图: code_file_not_found.png")
    
    def submit_code(self):
        """
        使用 Ctrl+Shift+P 触发 LeetCode 提交，并解析 webview 中的结果
        """
        print("正在提交代码...")
        try:
            # 1. 点击编辑器确保焦点
            self.page.locator('.monaco-editor').first.click()

            # 2. 触发提交快捷键
            self.page.keyboard.press("Control+Shift+P")
            print("已触发命令面板...")

            # 如需显式输入命令，可放开下面两行
            # self.page.keyboard.type("LeetCode: Submit")
            # self.page.keyboard.press("Enter")

            # 3. 智能等待结果出现
            print("等待提交结果...")

            keywords = [
                "Accepted",
                "Wrong Answer",
                "Time Limit Exceeded",
                "Compile Error",
                "Runtime Error",
                "Memory Limit Exceeded",
            ]

            result_data = {"header": None, "details": []}

            # 4. 遍历 frame 检索结果，设置总超时 30 秒
            end_time = time.time() + 30
            found = False

            while time.time() < end_time and not found:
                for frame in self.page.frames:
                    try:
                        for kw in keywords:
                            h2_locator = frame.locator(f"h2:has-text('{kw}')").first

                            if h2_locator.count() > 0 and h2_locator.is_visible():
                                result_data["header"] = h2_locator.inner_text().strip()

                                # 提取 ul > li 的详细信息
                                li_elements = frame.locator("ul > li")
                                count = li_elements.count()
                                result_data["details"] = [
                                    li_elements.nth(i).inner_text().strip()
                                    for i in range(count)
                                ]

                                found = True
                                break
                    except Exception:
                        # 忽略跨域或无效 frame
                        continue

                if not found:
                    time.sleep(1)  # 每秒轮询一次

            # 5. 输出结果
            if found:
                print("\n" + "=" * 15 + " LeetCode 提交结果 " + "=" * 15)
                print(f"状态: {result_data['header']}")
                for line in result_data["details"]:
                    print(f"详情: {line}")
                print("=" * 48)

                # 截图保存结果
                self.page.screenshot(path="submit_result.png")
                print("已保存提交结果截图: submit_result.png")

                return result_data
            else:
                print("❌ 未能在超时时间内解析到结果，请手动检查页面。")
                self.page.screenshot(path="submit_error.png")
                print("已保存错误截图: submit_error.png")

        except Exception as e:
            print(f"提交过程中发生异常: {e}")
            self.page.screenshot(path="submit_error.png")
            print("已保存错误截图: submit_error.png")
    
    def take_screenshot(self, filename: str = "github_screenshot.png"):
        """截图保存当前页面"""
        self.page.screenshot(path=filename)
        print(f"截图已保存: {filename}")
    
    def close_browser(self):
        """关闭浏览器"""
        if self.browser:
            self.browser.close()
            print("浏览器已关闭")
    
    def run(self, playwright: Playwright, headless: bool = False):
        """
        运行完整的自动化流程
        
        Args:
            playwright: Playwright实例
            headless: 是否以无头模式运行
        """
        try:
            # 启动浏览器
            self.start_browser(playwright, headless)
            
            # 访问GitHub
            self.navigate_to_github()
            
            # 判断是否已经登录 GitHub：
            # 简单策略：如果页面上看不到 "Sign in" 按钮，就认为已经是登录状态
            try:
                sign_in_locator = self.page.locator('a:text("Sign in")')
                if sign_in_locator.count() == 0:
                    print("检测到页面上不存在 Sign in 按钮，推断已登录 GitHub，跳过登录步骤")
                else:
                    print("检测到 Sign in 按钮，开始执行登录流程")
                    # 点击登录
                    self.click_sign_in()
                    # 执行登录
                    self.login()
            except Exception as e:
                # 检测过程中出错，不影响后续流程，按原逻辑尝试登录
                print(f"检测登录状态时出错，按正常流程尝试登录: {e}")
                self.click_sign_in()
                self.login()
            
            # 访问指定项目
            self.navigate_to_project()
            
            # 点击Code按钮
            self.click_code_button()
            
            # 进入Codespaces
            self.enter_codespaces()
            
            # 截图Codespaces加载完成状态
            self.take_screenshot("codespaces_loaded.png")
            
            # 点击LeetCode插件
            self.open_leetcode_plugin()
            
            # 登录LeetCode（可能会通过 Cookie 登录，也可能检测到已登录而跳过）
            self.login_leetcode()
            
            # 如果本次是通过 Cookie 登录成功，则认为会停留在当前视图，
            # 根据你的需求：直接去文件中点击代码并提交，跳过点击 All 和选择题目
            if self.leetcode_logged_in_via_cookie:
                print("本次通过 Cookie 登录 LeetCode，按照配置跳过点击 All 和选择题目步骤")
            else:
                # 非本次 Cookie 登录（可能是本来就已登录，或者登录失败但仍可用），按原逻辑操作 LeetCode 面板
                # 点击All显示所有题目
                self.click_all_problems()
                
                # 点击第一道题目
                self.click_first_problem()
            
            # 点击资源管理器图标查看项目文件
            self.click_explorer_icon()
            
            # 打开与题目匹配的代码文件
            self.open_code_file("two_sum")
            
            # 提交代码
            self.submit_code()
            
            # 最终截图
            self.take_screenshot()
            
            print("\n所有操作完成！")
            
        except Exception as e:
            print(f"\n执行过程中出错: {e}")
            self.take_screenshot("error_screenshot.png")
        finally:
            # 等待一段时间以便观察结果
            time.sleep(5)
            self.close_browser()


def main():
    """
    主函数 - 配置参数并运行自动化脚本
    """
    # ===== 配置区域 - 请修改以下参数 =====
    USERNAME = os.getenv("GITHUB_USERNAME")  # 你的GitHub用户名或邮箱
    PASSWORD = os.getenv("GITHUB_PASSWORD")  # 你的GitHub密码
    PROJECT_URL = os.getenv("PROJECT_URL")  # 目标项目URL
    HEADLESS = False  # 是否使用无头模式（True=不显示浏览器窗口）
    
    # LeetCode Cookie - 请从浏览器中复制你的LeetCode cookie
    # 登录leetcode.com后，从浏览器开发者工具中获取cookie
    LEETCODE_COOKIE = '''gr_user_id=f5bc3375-7445-4b43-9dba-c6924bbcfbab; ip_check=(false, "103.158.75.119"); _gid=GA1.2.1756888602.1765959475; 87b5a3c3f1a55520_gr_session_id=acd3e6ec-18a0-4b02-b9b8-1ce1cb55886e; 87b5a3c3f1a55520_gr_session_id_sent_vst=acd3e6ec-18a0-4b02-b9b8-1ce1cb55886e; INGRESSCOOKIE=d5903d7f4968dda3a8b32b5c1059eecc|8e0876c7c1464cc0ac96bc2edceabd27; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=acd3e6ec-18a0-4b02-b9b8-1ce1cb55886e; 87b5a3c3f1a55520_gr_last_sent_cs1=wingsjackf; __gads=ID=3314383dc9dbfd63:T=1765959508:RT=1765960725:S=ALNI_MZKM_0ftmbjBxPFmNH_MZ2VCbWhyw; __gpi=UID=000011cce2fa9e5c:T=1765959508:RT=1765960725:S=ALNI_MbcfJ91PkyeMjJ6PQSD5P84TL1iuQ; __eoi=ID=ca213cabdcdde30d:T=1765959508:RT=1765960725:S=AA-AfjZvByvqUBZM2BFwXNjTpLxd; cf_clearance=ISGXzvp4MMkDWifeyAMB5Uq4azoJS6Ft1AtUQpad7js-1765960756-1.2.1.1-wXJHQQDwnwy4Pn7lLP5YoO9FeWeiA2NeoxtXO3QgyDJQz2qiQ2AQNXF5hUioMaV6Wwwm73MNRBZRN2Di1mtmSgTa1LLCQzNjBgEEYNxAtP04ZtOZsS0oNoHszszMHwHl2Z0kVF9hCwpraWUF3YFydGVzhs5vResJ52aUZ4mUC75kryWBi8TYL49ARV0Sw6kpbfY4RY7PR7EH1t3g3ujZnzgm0dAvlrvZjFNZiL4AB1Y; csrftoken=0UmLpAPIHKUwflRMLoFj1VibDATnrrg3; messages=.eJyLjlaKj88qzs-Lz00tLk5MT1XSMdAxMtVRiswvVchILEtVKM5Mz0tNUcgvLdFT0lFSitXBpSO4NDkZKJJWmpNTCdOVmaeQWKxQnpmXXpyVmJydBjEiFgAb_Scz:1vVn49:pfBODKRt-VHt01GyzInbGeTbBO-fDhvado-7I8Z0WuM; _dd_s=rum=0&expire=1765961655836; LEETCODE_SESSION=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfYXV0aF91c2VyX2lkIjoiMjAyNzgxMTciLCJfYXV0aF91c2VyX2JhY2tlbmQiOiJhbGxhdXRoLmFjY291bnQuYXV0aF9iYWNrZW5kcy5BdXRoZW50aWNhdGlvbkJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI3N2I4ZjlmYzhiNTIxNGNhMDIyNTE3NWMxYjY2YWRiMmFlOWU2YmNlODQwNzJhNzViNGE0ZWUwZjQ1NjUxNDNlIiwic2Vzc2lvbl91dWlkIjoiZTBlY2Y4MjAiLCJpZCI6MjAyNzgxMTcsImVtYWlsIjoiMTMyNzg5OTg5MEBxcS5jb20iLCJ1c2VybmFtZSI6IndpbmdzamFja2YiLCJ1c2VyX3NsdWciOiJ3aW5nc2phY2tmIiwiYXZhdGFyIjoiaHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VzZXJzL3dpbmdzamFja2YvYXZhdGFyXzE3NjU5NTk0OTkucG5nIiwicmVmcmVzaGVkX2F0IjoxNzY1OTYwNzY1LCJpcCI6IjEwMy4xNTguNzUuMTE5IiwiaWRlbnRpdHkiOiI2ODRmYWMzZDhlNTk1ODQ1NjQwZTUwN2E5MTIyYmQ1NSIsImRldmljZV93aXRoX2lwIjpbIjA3YzBlYjlhZWVlZTQxMTk4NmM0ZTBlZDczMWQ0OGQzIiwiMTAzLjE1OC43NS4xMTkiXX0.kR-zUaQ7Wp3bT0bGmqXx8Gki__d-T7oiPkA3_IcALM4; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%223a4905a5-e4a0-4763-a6f1-5edb178b6c7e%5C%22%2C%5B1765959509%2C600000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol_nsKDRYW3BKFir-WrjYIGQzhRueop_DLCdXVdbKZkJlKLpqDk2dohSEl2k7xcVCcX_sD6WaoCw0at9T8TK80gwB9aB6cOadaFu6Co06Q3cJV4W5DqeaY3kTNhirm2PIkLSaA2Ct4WiTdfv2feHDzSnedsZyA%3D%3D%22%5D%5D; _ga=GA1.1.350590212.1765439998; 87b5a3c3f1a55520_gr_cs1=wingsjackf; _ga_CDRWKZTDEX=GS2.1.s1765959475$o2$g1$t1765961005$j43$l0$h0'''
    # =====================================
    
    print("="*50)
    print("GitHub 自动化脚本")
    print("="*50)
    print(f"目标项目: {PROJECT_URL}")
    print(f"运行模式: {'无头模式' if HEADLESS else '可视模式'}")
    print("="*50 + "\n")
    
    # 创建自动化实例
    automation = GitHubAutomation(USERNAME, PASSWORD, PROJECT_URL, LEETCODE_COOKIE)
    
    # 运行Playwright
    with sync_playwright() as playwright:
        automation.run(playwright, headless=HEADLESS)


if __name__ == "__main__":
    main()

