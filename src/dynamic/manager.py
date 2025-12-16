"""Dynamic Module Manager for runtime code execution and class/function loading."""

import importlib
import importlib.util
import sys
import inspect
import ast
from typing import Type, Optional, TypeVar, Any, Dict, List, Set, Callable

T = TypeVar('T')


class DynamicModuleManager:
    """Manager for dynamically creating Python modules and loading classes/functions.
    
    This class provides utilities for:
    - Creating virtual modules in memory (not on disk)
    - Loading classes and functions from source code strings
    - Managing dynamically generated code
    - Automatically injecting necessary imports based on code analysis
    """
    
    def __init__(self):
        """Initialize the dynamic module manager."""
        self._module_counter = 0
        self._loaded_modules: Dict[str, Any] = {}  # module_name -> module object
        # Symbol name -> object mapping for auto-injection
        self._symbol_registry: Dict[str, Any] = {}
        # Context-based import providers: context_name -> callable that returns imports dict
        self._context_providers: Dict[str, Callable[[], Dict[str, Any]]] = {}
    
    def _generate_module_name(self, prefix: str = "dynamic_module") -> str:
        """Generate a unique virtual module name.
        
        Args:
            prefix: Prefix for the module name
            
        Returns:
            A unique module name like "_dynamic_module_1", "_dynamic_module_2", etc.
        """
        self._module_counter += 1
        # This is a virtual module name - it will be added to sys.modules dynamically
        # It does NOT need to exist as a file on disk
        return f"_{prefix}_{self._module_counter}"
    
    def is_dynamic_class(self, cls: Type) -> bool:
        """Check if a class is dynamically generated (not from a real module file).
        
        Args:
            cls: The class to check
            
        Returns:
            True if the class appears to be dynamically generated
        """
        if not hasattr(cls, '__module__'):
            return True
        module_name = cls.__module__
        # Check if it's a dynamic module
        return (module_name in ('__main__', '<string>', '<exec>') or 
                module_name.startswith('_dynamic_') or
                '<' in module_name)
    
    def get_class_source_code(self, cls: Type) -> Optional[str]:
        """Extract source code of a class if possible.
        
        Args:
            cls: The class to extract source code from
            
        Returns:
            Source code string if available, None otherwise
        """
        try:
            return inspect.getsource(cls)
        except (OSError, TypeError):
            # Source code not available (e.g., dynamically generated, compiled, etc.)
            return None
    
    def extract_class_name_from_code(self, code: str) -> Optional[str]:
        """Extract the first class name from source code.
        
        Args:
            code: Source code string
            
        Returns:
            First class name found, or None
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
        except Exception:
            pass
        return None
    
    def register_symbol(self, name: str, obj: Any) -> None:
        """Register a symbol that can be auto-injected into dynamic code.
        
        Args:
            name: Symbol name (e.g., "TOOL", "Tool")
            obj: The object to inject
        """
        self._symbol_registry[name] = obj
    
    def register_context_provider(self, context_name: str, provider: Callable[[], Dict[str, Any]]) -> None:
        """Register a context-based import provider.
        
        Args:
            context_name: Context identifier (e.g., "tool", "agent")
            provider: Callable that returns a dict of {symbol_name: object} to inject
        """
        self._context_providers[context_name] = provider
    
    def _extract_used_symbols(self, code: str) -> Set[str]:
        """Extract all symbol names used in the code (excluding builtins and imports).
        
        Args:
            code: Source code string
            
        Returns:
            Set of symbol names used in the code
        """
        used_symbols = set()
        
        try:
            tree = ast.parse(code)
            
            class SymbolCollector(ast.NodeVisitor):
                def __init__(self):
                    self.imports = set()
                    self.names = set()
                    self.in_def = False  # Track if we're inside a function/class definition
                
                def visit_Import(self, node):
                    for alias in node.names:
                        self.imports.add(alias.asname or alias.name)
                
                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        self.imports.add(alias.asname or alias.name)
                
                def visit_FunctionDef(self, node):
                    self.imports.add(node.name)  # Function name is not a used symbol
                    self.generic_visit(node)
                
                def visit_AsyncFunctionDef(self, node):
                    self.imports.add(node.name)  # Function name is not a used symbol
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    self.imports.add(node.name)  # Class name is not a used symbol
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    # Only collect names that are loaded (used), not stored (assigned)
                    if isinstance(node.ctx, ast.Load):
                        self.names.add(node.id)
            
            collector = SymbolCollector()
            collector.visit(tree)
            
            # Return names that are used but not imported
            # Exclude common builtins and special names
            excluded = collector.imports | {
                'self', 'cls', 'super', '__name__', '__main__', '__file__', '__doc__',
                'True', 'False', 'None', 'Exception', 'BaseException',
                'object', 'type', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set'
            }
            used_symbols = collector.names - excluded
            
        except Exception:
            # If parsing fails, return empty set
            pass
        
        return used_symbols
    
    def _auto_inject_imports(self, code: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Automatically determine which imports to inject based on code analysis.
        
        Args:
            code: Source code string
            context: Optional context name (e.g., "tool", "agent") for context-specific imports
            
        Returns:
            Dict of {symbol_name: object} to inject
        """
        imports = {}
        
        # Add context-specific imports if context is provided
        if context and context in self._context_providers:
            context_imports = self._context_providers[context]()
            imports.update(context_imports)
        
        # Extract symbols used in code
        used_symbols = self._extract_used_symbols(code)
        
        # Auto-inject symbols that are used and registered
        for symbol_name in used_symbols:
            if symbol_name in self._symbol_registry:
                imports[symbol_name] = self._symbol_registry[symbol_name]
        
        return imports
    
    def load_code(self, code: str, module_name: Optional[str] = None, 
                  context: Optional[str] = None,
                  inject_imports: Optional[Dict[str, Any]] = None) -> str:
        """Load code into a virtual module and return the module name.
        
        Args:
            code: Source code string to execute
            module_name: Optional module name. If None, a unique name will be generated.
            context: Optional context name (e.g., "tool", "agent") for auto-injection
            inject_imports: Optional dict of {symbol_name: object} to inject manually.
                           If None, will auto-detect based on code analysis.
            
        Returns:
            The module name that was used
            
        Raises:
            Exception: If code execution fails
        """
        if module_name is None:
            module_name = self._generate_module_name()
        
        # Create a new module object (virtual, in memory only)
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        module = importlib.util.module_from_spec(spec)
        
        # Determine imports to inject
        if inject_imports is None:
            inject_imports = self._auto_inject_imports(code, context)
        else:
            # Merge with auto-detected imports
            auto_imports = self._auto_inject_imports(code, context)
            inject_imports = {**auto_imports, **inject_imports}
        
        # Inject imports into module namespace
        for name, obj in inject_imports.items():
            setattr(module, name, obj)
        
        # Execute the code in the module namespace
        exec(code, module.__dict__)
        
        # Add to sys.modules so Python treats it as a real importable module
        # This is a runtime virtual module - no file on disk needed
        sys.modules[module_name] = module
        
        # Store reference
        self._loaded_modules[module_name] = module
        
        return module_name
    
    def load_class(self, 
                   code: str, 
                   class_name: Optional[str] = None, 
                   base_class: Optional[Type[T]] = None, 
                   module_name: Optional[str] = None,
                   context: Optional[str] = None,
                   inject_imports: Optional[Dict[str, Any]] = None) -> Type[T]:
        """Dynamically load a class from source code.
        
        This function creates a virtual Python module in memory (not on disk) by:
        1. Generating a unique module name (doesn't need to exist as a file)
        2. Creating a module object using importlib
        3. Executing the code in the module's namespace
        4. Adding it to sys.modules so Python treats it as a real module
        
        Args:
            code: Source code string containing the class definition
            class_name: Name of the class to extract. If None, will try to extract from code.
            base_class: Optional base class to validate against (e.g., Tool, Agent)
            module_name: Optional module name. If None, a unique name will be generated.
            context: Optional context name (e.g., "tool", "agent") for auto-injection
            inject_imports: Optional dict of {symbol_name: object} to inject manually.
                           If None, will auto-detect based on code analysis.
            
        Returns:
            The loaded class
            
        Raises:
            ValueError: If the class cannot be found or loaded, or doesn't inherit from base_class
        """
        # Determine context from base_class if not provided
        if context is None and base_class is not None:
            # Try to infer context from base class name
            base_name = base_class.__name__.lower()
            if 'tool' in base_name:
                context = "tool"
            elif 'agent' in base_name:
                context = "agent"
        
        # Load code into module first (needed to find classes)
        if module_name is None:
            module_name = self.load_code(code, context=context, inject_imports=inject_imports)
        else:
            if module_name not in self._loaded_modules:
                self.load_code(code, module_name, context=context, inject_imports=inject_imports)
        
        # Get module
        module = self._loaded_modules[module_name]
        
        # Extract class name if not provided
        if class_name is None:
            if base_class is not None:
                # Find all classes that inherit from base_class
                candidate_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, base_class) and 
                        attr is not base_class):
                        candidate_classes.append((attr_name, attr))
                
                if len(candidate_classes) == 0:
                    raise ValueError(f"No class found in code that inherits from {base_class.__name__}")
                elif len(candidate_classes) == 1:
                    class_name = candidate_classes[0][0]
                else:
                    # Multiple candidates - prefer classes with @AGENT/@TOOL decorator or matching naming patterns
                    preferred = []
                    for name, cls in candidate_classes:
                        try:
                            source = inspect.getsource(cls)
                            if '@AGENT' in source or '@TOOL' in source or name.endswith('Agent') or name.endswith('Tool'):
                                preferred.append((name, cls))
                        except:
                            if name.endswith('Agent') or name.endswith('Tool'):
                                preferred.append((name, cls))
                    
                    if preferred:
                        class_name = preferred[0][0]
                    else:
                        # Fall back to first candidate
                        class_name = candidate_classes[0][0]
            else:
                # No base_class provided, extract first class from code
                class_name = self.extract_class_name_from_code(code)
                if not class_name:
                    raise ValueError("Cannot determine class name from code. Please provide class_name or base_class.")
        
        # Extract the class
        if not hasattr(module, class_name):
            raise ValueError(f"Class {class_name} not found in the provided code")
        
        cls = getattr(module, class_name)
        
        # Validate base class if provided
        if base_class is not None:
            if not issubclass(cls, base_class):
                raise ValueError(f"Class {class_name} is not a subclass of {base_class.__name__}")
        
        return cls
    
    def load_function(self, code: str, function_name: Optional[str] = None,
                     module_name: Optional[str] = None) -> Any:
        """Dynamically load a function from source code.
        
        Args:
            code: Source code string containing the function definition
            function_name: Name of the function to extract. If None, will try to extract from code.
            module_name: Optional module name. If None, a unique name will be generated.
            
        Returns:
            The loaded function
            
        Raises:
            ValueError: If the function cannot be found
        """
        # Extract function name if not provided
        if function_name is None:
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_name = node.name
                        break
            except Exception:
                pass
            
            if not function_name:
                raise ValueError("Cannot determine function name from code. Please provide function_name.")
        
        # Load code into module
        if module_name is None:
            module_name = self.load_code(code)
        else:
            if module_name not in self._loaded_modules:
                self.load_code(code, module_name)
        
        # Get module
        module = self._loaded_modules[module_name]
        
        # Extract the function
        if not hasattr(module, function_name):
            raise ValueError(f"Function {function_name} not found in the provided code")
        
        return getattr(module, function_name)
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get a loaded module by name.
        
        Args:
            module_name: The module name
            
        Returns:
            The module object, or None if not found
        """
        return self._loaded_modules.get(module_name)
    
    def list_loaded_modules(self) -> List[str]:
        """List all loaded dynamic module names.
        
        Returns:
            List of module names
        """
        return list(self._loaded_modules.keys())


# Global instance for convenience
dynamic_manager = DynamicModuleManager()

