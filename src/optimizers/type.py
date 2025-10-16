from pydantic import BaseModel, Field
from typing import List, Any, Optional, Union, Dict, Literal
from jinja2 import Environment, Template, meta

class Variable(BaseModel):
    name: str = Field(description="The name of the variable.")
    type: str = Field(description="The type of the variable.")
    description: str = Field(description="The description of the variable.")
    require_grad: bool = Field(default=False, description="Whether the variable requires gradient.")
    template: Optional[str] = Field(default=None, description="The template of the variable.")
    variables: Optional[Union[List['Variable'], Any]] = Field(default=None, description="The elements of the variable.")
    
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
                    # If the value is a string that might contain template syntax,
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
                # Update context with child modules for template rendering
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