# GitHub自动化脚本使用说明
# 注意！！！ 第一次运行的时候，需要在浏览器中手动设置提交快捷键

这个脚本使用Playwright自动化操作GitHub，包括登录和访问指定项目的代码库。

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 2. 安装Playwright浏览器驱动

```bash
playwright install chromium
```

### 3. 参考env_copy.txt新建 .env文件，完成相关信息配置



## 使用方法

### 1. 配置参数

打开 `github_automation.py` 文件，修改 `main()` 函数中的配置参数：

```python
USERNAME = "your_github_username"  # 改为你的GitHub用户名或邮箱
PASSWORD = "your_github_password"  # 改为你的GitHub密码
PROJECT_URL = "https://github.com/username/repository"  # 改为目标项目URL
HEADLESS = False  # True=无头模式（不显示浏览器），False=显示浏览器窗口
```

### 2. 运行脚本

```bash
python github_automation.py
```

## 功能说明

脚本会自动执行以下步骤：

1. ✅ 启动Chrome浏览器
2. ✅ 访问 GitHub.com
3. ✅ 点击登录按钮
4. ✅ 输入用户名和密码
5. ✅ 提交登录表单
6. ✅ 导航到指定的项目页面
7. ✅ 点击"Code"按钮
8. ✅ 访问代码库
9. ✅ 截图保存当前页面

## 注意事项

### 双因素认证(2FA)

如果你的GitHub账户启用了2FA：
- 脚本会在登录后等待20秒
- 请在这段时间内手动完成2FA验证
- 或者你可以修改等待时间

### 安全建议

⚠️ **警告**: 不要将包含真实密码的脚本提交到版本控制系统！

建议使用环境变量或配置文件来存储敏感信息：

```python
import os

USERNAME = os.getenv("GITHUB_USERNAME")
PASSWORD = os.getenv("GITHUB_PASSWORD")
```

然后在运行前设置环境变量：

```bash
export GITHUB_USERNAME="your_username"
export GITHUB_PASSWORD="your_password"
python github_automation.py
```

### 调试模式

如果遇到问题：
1. 设置 `HEADLESS = False` 来查看浏览器操作
2. 检查截图文件 `github_screenshot.png`
3. 查看终端输出的详细日志

## 扩展功能

你可以根据需求扩展脚本功能：

### 浏览特定文件

```python
def open_file(self, file_path: str):
    """打开项目中的特定文件"""
    file_url = f"{self.project_url}/blob/main/{file_path}"
    self.page.goto(file_url)
```

### 克隆仓库URL

```python
def get_clone_url(self):
    """获取克隆URL"""
    clone_url = self.page.locator('input[aria-label="Clone this repository"]').input_value()
    return clone_url
```

### 搜索代码

```python
def search_code(self, query: str):
    """在项目中搜索代码"""
    search_button = self.page.locator('button[aria-label="Search this repository"]')
    search_button.click()
    search_input = self.page.locator('input[name="q"]')
    search_input.fill(query)
    search_input.press("Enter")
```

## 常见问题

**Q: 为什么登录失败？**
A: 检查用户名和密码是否正确，GitHub的登录页面结构是否有变化

**Q: 脚本运行很慢？**
A: 这是正常的，因为脚本中添加了多个 `time.sleep()` 来等待页面加载

**Q: 如何加速执行？**
A: 使用 `page.wait_for_selector()` 替代 `time.sleep()`，使脚本更智能地等待元素加载

## 许可证

本脚本仅供学习和个人使用。请遵守GitHub的服务条款和使用政策。

