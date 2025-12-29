"""
Base optimizer module.
Contains shared logic for all optimizers, including variable extraction and cache management.
"""

from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Any, Optional, Union, Dict, Set
from jinja2 import Environment, Template, meta
from collections import defaultdict
from functools import partial

from src.logger import logger


class Variable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(description="The name of the variable.")
    type: str = Field(description="The type of the variable.")
    description: str = Field(description="The description of the variable.")
    require_grad: bool = Field(default=False, description="Whether the variable requires gradient.")
    template: Optional[str] = Field(default=None, description="The template of the variable.")
    variables: Optional[Union[List['Variable'], Any]] = Field(default=None, description="The elements of the variable.")
    
    # gradient related attributes
    gradients: Set['Variable'] = Field(default_factory=set, description="Text gradients for this variable.")
    gradients_context: Dict['Variable', str] = Field(default_factory=lambda: defaultdict(lambda: None), description="Context for gradients.")
    grad_fn: Optional[Any] = Field(default=None, description="Gradient function for backward pass.")
    predecessors: Set['Variable'] = Field(default_factory=set, description="Predecessor variables in computation graph.")
    reduce_meta: List[Dict] = Field(default_factory=list, description="Metadata for gradient reduction.")
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return id(self) == id(other)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Recursively construct Variable tree from nested dict."""
        subvars = data.get("variables")
        if isinstance(subvars, list):
            subvars = [cls.from_dict(v) for v in subvars]
        elif subvars is not None and not isinstance(subvars, list):
            pass
        return cls(
            name=data["name"],
            type=data.get("type", ""),
            description=data.get("description", ""),
            require_grad=data.get("require_grad", False),
            template=data.get("template"),
            variables=subvars,
        )
    
    def render(self, modules: Dict[str, Any]) -> str:
        """Render the template with the given modules."""
        if self.template is None:
            return ""
        
        env = Environment()
        ast = env.parse(self.template)
        vars_used = meta.find_undeclared_variables(ast)
        ctx = dict(modules)
        
        for var in vars_used:
            if var in modules:
                val = modules[var]
                if isinstance(val, str):
                    # render it as a template
                    try:
                        temp_template = Template(val)
                        ctx[var] = temp_template.render(**ctx)
                    except:
                        # If rendering fails, use the raw string
                        ctx[var] = val
                else:
                    ctx[var] = val
        
        return Template(self.template).render(**ctx)
    
    def get_modules(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get all modules (variables) from this variable and its children."""
        result: Dict[str, Any] = {}
        ctx = dict(context or {})

        # First, process child variables to build context
        if isinstance(self.variables, list):
            for child in self.variables:
                child_modules = child.get_modules(ctx)
                result.update(child_modules)
                ctx.update(child_modules)
        elif isinstance(self.variables, Variable):
            child_modules = self.variables.get_modules(ctx)
            result.update(child_modules)
            ctx.update(child_modules)
        elif self.variables is not None:
            # Direct value assignment
            result[self.name] = self.variables
            ctx[self.name] = self.variables

        # Then render template if it exists
        if self.template is not None:
            try:
                rendered = Template(self.template).render(**ctx)
                result[self.name] = rendered
                ctx[self.name] = rendered
            except Exception as e:
                # If template rendering fails, use the raw template
                result[self.name] = self.template
                ctx[self.name] = self.template

        return result
    
    def get_value(self) -> str:
        """Get the complete value of the variable - through rendering template and sub-variables"""
        if self.template is None:
            # If no template, directly return sub-variable values
            if isinstance(self.variables, list):
                return " ".join([child.get_value() for child in self.variables])
            elif isinstance(self.variables, Variable):
                return self.variables.get_value()
            elif self.variables is not None:
                return str(self.variables)
            else:
                return ""
        
        # Use template rendering
        modules = self.get_modules()
        return self.render(modules)
    
    def __repr__(self):
        return f"Variable(name={self.name}, value={self.get_value()}, role={self.description}, grads={len(self.gradients)})"
    
    def __str__(self):
        return self.get_value()
    
    def __add__(self, to_add):
        """Support variable addition operation, build computation graph"""
        if isinstance(to_add, Variable):
            # Create new variable representing addition result
            result = Variable(
                name=f"{self.name}_plus_{to_add.name}",
                type="computed",
                description=f"{self.description} and {to_add.description}",
                require_grad=(self.require_grad or to_add.require_grad),
                template="{{var1}} {{var2}}",  # Simple concatenation template
                variables=[self, to_add],
                predecessors={self, to_add}
            )
            # Set gradient function
            result.set_grad_fn(partial(
                self._backward_idempotent,
                variables=[self, to_add],
                summation=result,
            ))
            return result
        else:
            return to_add.__add__(self)
    
    def set_grad_fn(self, grad_fn):
        """Set gradient function"""
        self.grad_fn = grad_fn
    
    def get_grad_fn(self):
        """Get gradient function"""
        return self.grad_fn
    
    def reset_gradients(self):
        """Reset gradients - recursively handle nested variables"""
        self.gradients = set()
        self.gradients_context = defaultdict(lambda: None)
        self.reduce_meta = []
        
        # Recursively reset gradients of sub-variables
        if isinstance(self.variables, list):
            for child in self.variables:
                if isinstance(child, Variable):
                    child.reset_gradients()
        elif isinstance(self.variables, Variable):
            self.variables.reset_gradients()
    
    def get_gradient_text(self) -> str:
        """Get aggregated gradient text"""
        return "\n".join([g.get_value() for g in self.gradients])
    
    def get_short_value(self, n_words_offset: int = 10) -> str:
        """Get short representation of the variable"""
        value = self.get_value()
        words = value.split(" ")
        if len(words) <= 2 * n_words_offset:
            return value
        short_value = " ".join(words[:n_words_offset]) + " (...) " + " ".join(words[-n_words_offset:])
        return short_value
    
    def backward(self, engine: Any = None):
        """Backward propagation, compute text gradients - supports nested structure"""
        # Topological sort all predecessor nodes
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)
        
        build_topo(self)
        
        # Backward propagate gradients
        self.gradients = set()
        for v in reversed(topo):
            if v.require_grad:
                v.gradients = self._check_and_reduce_gradients(v, engine)
                if v.get_grad_fn() is not None:
                    v.grad_fn(backward_engine=engine)
    
    def _check_and_reduce_gradients(self, variable: 'Variable', backward_engine=None) -> Set['Variable']:
        """Check and reduce gradients"""
        if variable.reduce_meta == []:
            return variable.gradients
        if variable.get_gradient_text() == "":
            return variable.gradients
        
        if len(variable.gradients) == 1:
            return variable.gradients
        
        # Implement gradient aggregation logic
        id_to_gradient_set = defaultdict(set)
        id_to_op = {}
        
        for gradient in variable.gradients:
            for reduce_item in gradient.reduce_meta:
                id_to_gradient_set[reduce_item["id"]].add(gradient)
                id_to_op[reduce_item["id"]] = reduce_item["op"]
        
        new_gradients = set()
        for group_id, gradients in id_to_gradient_set.items():
            new_gradients.add(id_to_op[group_id](gradients, backward_engine))
        
        return new_gradients
    
    def _backward_idempotent(self, variables: List['Variable'], summation: 'Variable', backward_engine=None):
        """Idempotent backward propagation, used for variable addition operations"""
        summation_gradients = summation.get_gradient_text()
        for variable in variables:
            if summation_gradients == "":
                variable_gradient_value = ""
            else:
                variable_gradient_value = f"Here is the combined feedback for this specific {variable.description} and other variables: {summation_gradients}."
            
            var_gradients = Variable(
                name=f"gradient_for_{variable.name}",
                type="gradient",
                description=f"feedback to {variable.description}",
                require_grad=False,
                variables=variable_gradient_value
            )
            variable.gradients.add(var_gradients)
            
            if summation.reduce_meta != []:
                var_gradients.reduce_meta.extend(summation.reduce_meta)
                variable.reduce_meta.extend(summation.reduce_meta)
    
    def generate_graph(self, print_gradients: bool = False):
        """Generate computation graph visualization - supports nested structure"""
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("Please install graphviz to visualize the computation graphs.")
        
        def wrap_text(text, width=40):
            words = text.split()
            wrapped_text = ""
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > width:
                    wrapped_text += line + "<br/>"
                    line = word
                else:
                    if line:
                        line += " "
                    line += word
            wrapped_text += line
            return wrapped_text
        
        def wrap_and_escape(text, width=40):
            return wrap_text(text.replace("<", "&lt;").replace(">", "&gt;"), width)
        
        # Build topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)
        
        build_topo(self)
        
        graph = Digraph(comment=f'Computation Graph starting from {self.description}')
        graph.attr(rankdir='TB')
        graph.attr(ranksep='0.2')
        graph.attr(bgcolor='lightgrey')
        graph.attr(fontsize='7.5')
        
        for v in reversed(topo):
            label_color = 'darkblue'
            
            node_label = (
                f"<b><font color='{label_color}'>Name: </font></b> {wrap_and_escape(v.name)}"
                f"<br/><b><font color='{label_color}'>Description: </font></b> {wrap_and_escape(v.description)}"
                f"<br/><b><font color='{label_color}'>Value: </font></b> {wrap_and_escape(v.get_value())}"
            )
            
            if v.grad_fn is not None:
                node_label += f"<br/><b><font color='{label_color}'>Grad Fn: </font></b> {wrap_and_escape(str(v.grad_fn))}"
            
            if print_gradients:
                node_label += f"<br/><b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(v.get_gradient_text())}"
            
            graph.node(
                str(id(v)),
                label=f"<{node_label}>",
                shape='rectangle',
                style='filled',
                fillcolor='lavender',
                fontsize='8',
                fontname="Arial",
                margin='0.1',
                pad='0.1',
                width='1.2',
            )
            
            for predecessor in v.predecessors:
                graph.edge(str(id(predecessor)), str(id(v)))
        
        return graph
    
    def get_all_variables(self) -> List['Variable']:
        """Get all nested variables - for batch operations"""
        all_vars = [self]
        
        if isinstance(self.variables, list):
            for child in self.variables:
                if isinstance(child, Variable):
                    all_vars.extend(child.get_all_variables())
        elif isinstance(self.variables, Variable):
            all_vars.extend(self.variables.get_all_variables())
        
        return all_vars
    
    def filter_trainable_sub_variables(self) -> 'Variable':
        """Create a copy of this variable with only trainable sub-variables.
        
        Returns:
            Variable: A new Variable instance with filtered sub-variables (only require_grad=True)
        """
        # If no sub-variables (None, empty list, or non-Variable value), return self as-is
        if not isinstance(self.variables, (list, Variable)):
            return self
        
        # Filter sub-variables to only include those with require_grad=True or their trainable descendants
        filtered_sub_vars = []
        if isinstance(self.variables, list):
            for child in self.variables:
                if isinstance(child, Variable):
                    # Check if child has any trainable variables (itself or descendants)
                    trainable_in_child = child.get_trainable_variables()
                    if trainable_in_child:
                        # Recursively filter child's sub-variables
                        filtered_child = child.filter_trainable_sub_variables()
                        filtered_sub_vars.append(filtered_child)
        elif isinstance(self.variables, Variable):
            # Check if child has any trainable variables (itself or descendants)
            trainable_in_child = self.variables.get_trainable_variables()
            if trainable_in_child:
                filtered_sub_vars = self.variables.filter_trainable_sub_variables()
            else:
                # No trainable sub-variables, return self with empty variables
                return Variable(
                    name=self.name,
                    type=self.type,
                    description=self.description,
                    require_grad=self.require_grad,
                    template=self.template,
                    variables=None,
                    gradients=self.gradients.copy(),
                    gradients_context=self.gradients_context.copy(),
                    grad_fn=self.grad_fn,
                    predecessors=self.predecessors.copy(),
                    reduce_meta=self.reduce_meta.copy()
                )
        
        # Create a copy of this variable with filtered sub-variables
        return Variable(
            name=self.name,
            type=self.type,
            description=self.description,
            require_grad=self.require_grad,
            template=self.template,
            variables=filtered_sub_vars if filtered_sub_vars else None,
            gradients=self.gradients.copy(),
            gradients_context=self.gradients_context.copy(),
            grad_fn=self.grad_fn,
            predecessors=self.predecessors.copy(),
            reduce_meta=self.reduce_meta.copy()
        )
    
    def get_trainable_variables(self) -> List['Variable']:
        """Get trainable variables.
        
        Logic:
        - If the variable has sub-variables (list or Variable), recursively get trainable 
          variables from sub-variables only (filter by require_grad).
        - If the variable has no sub-variables (None, empty list, or non-Variable value), 
          return the variable itself if it requires gradients.
        
        Returns:
            List[Variable]: List of all trainable variables (sub-variables with require_grad=True, 
                          or the variable itself if it has no sub-variables and require_grad=True)
        """
        trainable_vars: List['Variable'] = []
        
        # Check if there are sub-variables (Variable objects, not strings or other types)
        has_sub_variables = False
        if isinstance(self.variables, list) and len(self.variables) > 0:
            has_sub_variables = True
            # Recursively get trainable variables from sub-variables
            for child in self.variables:
                if isinstance(child, Variable):
                    trainable_vars.extend(child.get_trainable_variables())
        elif isinstance(self.variables, Variable):
            has_sub_variables = True
            # Recursively get trainable variables from sub-variable
            trainable_vars.extend(self.variables.get_trainable_variables())
        
        # If no sub-variables, check if the variable itself requires gradients
        if not has_sub_variables and self.require_grad:
            trainable_vars.append(self)
        
        return trainable_vars

class Optimizer(BaseModel):
    """Base optimizer that provides shared functionality such as variable extraction and cache management."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    agent: Any = Field(description="The agent instance to optimize.")
    
    def __init__(self,
                 agent, 
                 workdir: str,
                 model_name: Optional[str] = None,
                 prompt_name: Optional[str] = None,
                 prompt_modules: Optional[Dict[str, Any]] = None,
                 memory_name: Optional[str] = None,
                 max_steps: int = 3,
                 **kwargs
                 ):
        super().__init__(agent=agent, **kwargs)
        
        # Set working directory
        self.workdir = workdir

        # Set prompt name and modules
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.model_name = model_name

        # Setup prompt modules
        self.prompt_modules = prompt_modules or {}
        self.max_steps = max_steps if max_steps > 0 else int(1e8)
        self.prompt_modules["workdir"] = self.workdir
        
        # Setup optimizable variables
        self.optimizable_vars = []
        self.var_mapping = {}
        self.prompt_mapping = None
        
    
    def find_prompt_objects_with_variables(self) -> List[Tuple[Any, str]]:
        """
        Find all prompt objects on the agent that contain `Variable` instances.

        The method traverses the prompt manager to discover prompt objects.

        Returns:
            List[Tuple[prompt_obj, prompt_name]]: A list of (prompt object, prompt name) pairs.
        """
        prompt_objects = []
        
        # Search through the prompt manager for prompt objects.
        if hasattr(self.agent, 'prompt_manager'):
            pm = self.agent.prompt_manager
            
            # SystemPrompt.
            if hasattr(pm, 'system_prompt') and hasattr(pm.system_prompt, 'prompt'):
                prompt_objects.append((pm.system_prompt, 'prompt_manager.system_prompt'))
            
            # AgentMessagePrompt.
            if hasattr(pm, 'agent_message_prompt') and hasattr(pm.agent_message_prompt, 'prompt'):
                prompt_objects.append((pm.agent_message_prompt, 'prompt_manager.agent_message_prompt'))
        
        return prompt_objects
    
    def _extract_from_variable_recursive(self, var, parent_name: str = ""):
        """
        Recursively extract variables with `require_grad=True`.

        Args:
            var: Variable instance.
            parent_name: Parent variable name used during recursion.

        Returns:
            List[orig_var]: List of optimizable variables.
        """
        result = []
        
        # Check whether the current variable should be optimized.
        if hasattr(var, 'require_grad') and var.require_grad:
            result.append(var)
        
        # Recursively process child variables.
        if hasattr(var, 'variables'):
            if isinstance(var.variables, list):
                for child in var.variables:
                    if hasattr(child, 'require_grad'):
                        result.extend(self._extract_from_variable_recursive(
                            child, f"{parent_name}.{var.name}" if parent_name else var.name
                        ))
            elif hasattr(var.variables, 'require_grad'):
                result.extend(self._extract_from_variable_recursive(
                    var.variables, f"{parent_name}.{var.name}" if parent_name else var.name
                ))
        
        return result
    
    def extract_optimizable_variables(self) -> Tuple[List[Any], Dict[Any, Any]]:
        """
        Extract optimizable variables (`require_grad=True`) from all prompt objects on the agent.

        This is a generic extraction method; subclasses may override it to return a custom structure.

        Returns:
            Tuple[List[orig_var], Dict[orig_var -> prompt_obj]]:
                (List of optimizable variables, mapping from variable to owning prompt object.)
        """
        all_optimizable_vars = []
        prompt_mapping = {}  # orig_var -> prompt_obj
        
        prompt_objects = self.find_prompt_objects_with_variables()
        
        for prompt_obj, prompt_name in prompt_objects:
            if hasattr(prompt_obj, 'prompt'):
                prompt_var = prompt_obj.prompt
                optimizable_vars = self._extract_from_variable_recursive(prompt_var)
                all_optimizable_vars.extend(optimizable_vars)
                
                # Record which prompt object owns each variable.
                for orig_var in optimizable_vars:
                    prompt_mapping[orig_var] = prompt_obj
        
        logger.info(f"| 📊 Found {len(all_optimizable_vars)} optimizable variables from {len(prompt_objects)} prompt object(s):")
        for orig_var in all_optimizable_vars:
            prompt_obj = prompt_mapping.get(orig_var)
            prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__ if prompt_obj else 'unknown'
            var_name = orig_var.name if hasattr(orig_var, 'name') else 'unknown'
            var_desc = orig_var.description if hasattr(orig_var, 'description') else f"Prompt module: {var_name}"
            logger.info(f"|   - [{prompt_name}] {var_name}: {var_desc}")
        
        return all_optimizable_vars, prompt_mapping
    
    def clear_prompt_caches(self, vars_to_clear: Optional[List[Any]] = None):
        """
        Clear the cached prompts for any prompt objects that contain the given variables.

        Args:
            vars_to_clear: List of variables whose prompt caches should be cleared.
                If None, all recorded variables are considered.

        Reloading details:
        - After clearing the cache, when the agent calls `prompt_obj.get_message()` (typically inside `_get_messages()`),
          if `reload=False` and `message` is `None`, the prompt automatically re-renders (`prompt.render()`).
          At that moment the updated variable values are applied.
        - Relevant locations:
          * `ToolCallingAgent._get_messages()` -> `prompt_manager.get_system_message()`
          * `SystemPrompt.get_message()` -> if `message` is `None`, `prompt.render(modules)` runs
        """
        if vars_to_clear is None:
            vars_to_clear = self.optimizable_vars
        
        # Collect all prompt objects whose caches should be cleared (deduplicated).
        prompt_objects_to_clear = set()
        
        # Look up prompt objects using `var_mapping` (used by Reflection-based optimizers).
        for orig_var in vars_to_clear:
            if orig_var in self.var_mapping:
                prompt_obj = self.var_mapping[orig_var]
                prompt_objects_to_clear.add(prompt_obj)
        
        # Look up prompt objects using `prompt_mapping` (used by TextGrad-based optimizers).
        # Subclasses can override this method for specialized handling.
        
        # Clear the cache on each prompt object.
        for prompt_obj in prompt_objects_to_clear:
            # Both `SystemPrompt` and `AgentMessagePrompt` expose a `message` attribute.
            if hasattr(prompt_obj, 'message'):
                prompt_obj.message = None
                prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__
                logger.debug(f"| 🗑️ Cleared cache for {prompt_name}")
        
        if prompt_objects_to_clear:
            logger.info(f"| 🗑️ Cleared cache for {len(prompt_objects_to_clear)} prompt object(s)")
    
    def get_variable_value(self, var: Any) -> str:
        """
        Return the current value of the variable as a string.

        Args:
            var: Variable instance.

        Returns:
            str: Variable value.
        """
        if hasattr(var, 'get_value'):
            return var.get_value()
        elif hasattr(var, 'variables'):
            return str(var.variables)
        elif hasattr(var, 'value'):
            return str(var.value)
        else:
            return str(var)
    
    def set_variable_value(self, var: Any, value: str):
        """
        Assign a new value to the variable.

        Args:
            var: Variable instance.
            value: New value.
        """
        if hasattr(var, 'variables'):
            var.variables = value
        elif hasattr(var, 'value'):
            var.value = value
        else:
            raise ValueError(f"Cannot set value for variable {type(var)}")
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3,
        **kwargs
    ):
        """
        Execute the optimization routine (abstract; subclasses must implement).

        Args:
            task: Task description.
            files: Optional list of attachment paths.
            optimization_steps: Number of optimization iterations.
            **kwargs: Additional optimizer-specific parameters.
        """
        raise NotImplementedError(f"``optimize`` function for {type(self).__name__} is not implemented!")
    
    def close(self):
        """Close the optimizer and release resources."""
        raise NotImplementedError(f"``close`` function for {type(self).__name__} is not implemented!")

