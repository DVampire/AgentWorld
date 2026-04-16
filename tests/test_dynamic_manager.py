"""Example usage of DynamicModuleManager showing class names, module paths, and import methods."""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.dynamic import dynamic_manager
import sys

# Example 1: Loading a class from code
code = """
class MyDynamicTool:
    def __init__(self, name: str):
        self.name = name
    
    def execute(self):
        return f"Executing {self.name}"
"""

# Load the class
my_class = dynamic_manager.load_class(code, class_name="MyDynamicTool")

# Check the class name and module path
print("=" * 60)
print("Example 1: Basic class loading")
print("=" * 60)
print(f"Class name: {my_class.__name__}")
print(f"Module path: {my_class.__module__}")
print(f"Full path: {my_class.__module__}.{my_class.__name__}")

# Check if the module exists in sys.modules
module_name = my_class.__module__
print(f"\nModule in sys.modules: {module_name in sys.modules}")
if module_name in sys.modules:
    module = sys.modules[module_name]
    print(f"Module object: {module}")
    print(f"Module has class: {hasattr(module, 'MyDynamicTool')}")

# Example 2: Importing the class using importlib
print("\n" + "=" * 60)
print("Example 2: Importing the dynamically loaded class")
print("=" * 60)
import importlib

# The module name is something like "_dynamic_module_1"
# You can import it like this:
try:
    # Method 1: Direct import using the module name
    module = importlib.import_module(my_class.__module__)
    imported_class = getattr(module, my_class.__name__)
    print(f"Imported class: {imported_class}")
    print(f"Same class object: {imported_class is my_class}")
    
    # Method 2: Using the class's __module__ attribute
    module_path = my_class.__module__
    class_name = my_class.__name__
    print(f"\nImport path: from {module_path} import {class_name}")
    
except ImportError as e:
    print(f"Import error: {e}")

# Example 3: Multiple classes in the same module
print("\n" + "=" * 60)
print("Example 3: Multiple classes in one module")
print("=" * 60)
multi_code = """
class ToolA:
    def run(self):
        return "Tool A"

class ToolB:
    def run(self):
        return "Tool B"
"""

# Load both classes (they'll be in the same module)
tool_a = dynamic_manager.load_class(multi_code, class_name="ToolA")
tool_b = dynamic_manager.load_class(multi_code, class_name="ToolB", module_name=tool_a.__module__)

print(f"ToolA module: {tool_a.__module__}")
print(f"ToolB module: {tool_b.__module__}")
print(f"Same module: {tool_a.__module__ == tool_b.__module__}")

# Example 4: Using the manager's get_module method
print("\n" + "=" * 60)
print("Example 4: Using get_module() to access the module")
print("=" * 60)
module = dynamic_manager.get_module(tool_a.__module__)
if module:
    print(f"Module retrieved: {module}")
    print(f"Module has ToolA: {hasattr(module, 'ToolA')}")
    print(f"Module has ToolB: {hasattr(module, 'ToolB')}")
    
    # Access classes directly from module
    tool_a_from_module = module.ToolA
    tool_b_from_module = module.ToolB
    print(f"ToolA from module: {tool_a_from_module}")
    print(f"ToolB from module: {tool_b_from_module}")

# Example 5: List all loaded modules
print("\n" + "=" * 60)
print("Example 5: List all loaded dynamic modules")
print("=" * 60)
loaded_modules = dynamic_manager.list_loaded_modules()
print(f"Loaded modules: {loaded_modules}")
for mod_name in loaded_modules:
    mod = dynamic_manager.get_module(mod_name)
    if mod:
        # Get all classes in the module
        classes = [name for name in dir(mod) if isinstance(getattr(mod, name), type) and not name.startswith('_')]
        print(f"  {mod_name}: {classes}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
1. Class name: The class name is the same as defined in the code (e.g., 'MyDynamicTool')
2. Module path: Generated as '_dynamic_module_1', '_dynamic_module_2', etc.
3. Full path: '{module_name}.{class_name}' (e.g., '_dynamic_module_1.MyDynamicTool')
4. Import methods:
   a) Direct: importlib.import_module(module_name)
   b) From module: getattr(module, class_name)
   c) Using manager: dynamic_manager.get_module(module_name)
5. The module is added to sys.modules, so it can be imported like a regular module
""")

