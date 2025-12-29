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

class Variable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(description="The name of the variable.")
    type: str = Field(description="The type of the variable.")
    description: str = Field(description="The description of the variable.")
    require_grad: bool = Field(default=False, description="Whether the variable requires gradient.")
    template: Optional[str] = Field(default=None, description="The template of the variable.")
    variables: Optional[Union[Dict[str, 'Variable'], 'Variable', Any]] = Field(default=None, description="The elements of the variable. Can be a dict (keyed by name), single Variable, or direct value.")
    
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
        """Recursively construct Variable tree from nested dict.
        
        Supports dict format for variables:
        - Dict format: {"var1": {"name": "var1", ...}, "var2": {"name": "var2", ...}}
        """
        subvars = data.get("variables")
        if isinstance(subvars, dict):
            # Dict format: recursively process each variable
            subvars = {k: cls.from_dict(v) if isinstance(v, dict) and "name" in v else v 
                      for k, v in subvars.items()}
        elif subvars is not None and not isinstance(subvars, dict):
            # Direct value (string, etc.) - keep as is
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
        if isinstance(self.variables, dict):
            # Dict format: iterate over values
            for child in self.variables.values():
                if isinstance(child, Variable):
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
            if isinstance(self.variables, dict):
                # Dict format: iterate over values
                return " ".join([child.get_value() for child in self.variables.values() if isinstance(child, Variable)])
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
        if isinstance(self.variables, dict):
            # Dict format: iterate over values
            for child in self.variables.values():
                if isinstance(child, Variable):
                    child.reset_gradients()
        elif isinstance(self.variables, Variable):
            self.variables.reset_gradients()
    
    def get_gradient_text(self) -> str:
        """Get aggregated gradient text"""
        return "\n".join([g.get_value() for g in self.gradients])

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
        
        if isinstance(self.variables, dict):
            # Dict format: iterate over values
            for child in self.variables.values():
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
        # If no sub-variables (None, empty dict, or non-Variable value), return self as-is
        if not isinstance(self.variables, (dict, Variable)):
            return self
        
        # Filter sub-variables to only include those with require_grad=True or their trainable descendants
        filtered_sub_vars = {}
        if isinstance(self.variables, dict):
            # Dict format: iterate over key-value pairs
            for key, child in self.variables.items():
                if isinstance(child, Variable):
                    # Check if child has any trainable variables (itself or descendants)
                    trainable_in_child = child.get_trainable_variables()
                    if trainable_in_child:
                        # Recursively filter child's sub-variables
                        filtered_child = child.filter_trainable_sub_variables()
                        filtered_sub_vars[key] = filtered_child
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
        if isinstance(self.variables, dict) and len(self.variables) > 0:
            # Dict format: iterate over values
            has_sub_variables = True
            # Recursively get trainable variables from sub-variables
            for child in self.variables.values():
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
    
    def __init__(self,
                 workdir: str,
                 model_name: Optional[str] = None,
                 prompt_name: Optional[str] = None,
                 memory_name: Optional[str] = None,
                 max_steps: int = 3,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        # Set working directory
        self.workdir = workdir

        # Set prompt name and modules
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.model_name = model_name

        # Setup max steps
        self.max_steps = max_steps if max_steps > 0 else int(1e8)
        
        # Setup optimizable variables
        self.optimizable_vars = []
        self.var_mapping = {}
        self.prompt_mapping = None
        
    async def get_trainable_variables(self):
        """Get trainable variables from the optimizer."""
        raise NotImplementedError(f"``get_trainable_variables`` function for {type(self).__name__} is not implemented!")
    
    async def set_trainable_variables(self, variables: List['Variable']):
        """Set trainable variables to the optimizer."""
        raise NotImplementedError(f"``set_trainable_variables`` function for {type(self).__name__} is not implemented!")
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Execute the optimization routine (abstract; subclasses must implement).

        Args:
            task: Task description.
            files: Optional list of attachment paths.
        """
        raise NotImplementedError(f"``optimize`` function for {type(self).__name__} is not implemented!")
    
    def close(self):
        """Close the optimizer and release resources."""
        raise NotImplementedError(f"``close`` function for {type(self).__name__} is not implemented!")

