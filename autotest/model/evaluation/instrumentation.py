import ast
import inspect

from typing import Any
from autotest.model.context.scene import Scene
from autotest.model.modules.rule_set import RuleSet, rule_set_scope


class Instrumentation:
    
    def __init__(self, rule_set: RuleSet) -> None:
        self.rule_set = rule_set()
        self.instrumented_rule_set = self.instrument(rule_set)
        self.results = {}

    def instrument(self, rule_set: RuleSet) -> RuleSet:
        source = inspect.getsource(rule_set)
        rule_set_ast = ast.parse(source)        
        Instrumenter().visit(rule_set_ast)
        ast.fix_missing_locations(rule_set_ast)

        scope = rule_set_scope()
        code = compile(rule_set_ast, filename='', mode='exec')  # Compile the rule set.
        exec(code, scope)  # Execute the compiled rule set in the given scope.
        
        instrumented_rule_set = scope.get('RuleSet')  # Extract the callable rule set from the scope.
        return instrumented_rule_set()

    def process(self, scene: Scene):
        if scene.id not in self.results:
            self.results[scene.id] = []
        
        executed_statements = self.instrumented_rule_set.executed_statements
        self.results[scene.id].append(executed_statements)

class Instrumenter(ast.NodeTransformer):
    """Will instrument an AST containing, such in each of the bodies of every if-else statement, the test condition's
     line number is added to the ordered executed statement list (the variable called executed_statements)."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """ Resets, and adds the executed path variable to the global scope, so it can be modified within the function.
        def function():
            self.executed_statements = []
            ...
        """
        self.generic_visit(node)
        
        declare_node = ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(
                    id='self', 
                    ctx=ast.Load()
                ), 
                attr='executed_statements', ctx=ast.Store()
            )], 
            value=ast.List(
                elts=[], 
                ctx=ast.Load()
            ), 
            type_comment=None
        )
        
        node.body.insert(0, declare_node)
        
        return node

    def visit_If(self, node: ast.If) -> Any:
        """ Appends the line number of the statement to the list of executed statements within the if and else body.
        if x: <--- e.g. line number, lineno = 3
            self.executed_statements.append(lineno)
            ...
        else:
            self.executed_statements.append(lineno)
            ...
        """
        self.generic_visit(node)
        
        append_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(
                            id='self', 
                            ctx=ast.Load()
                        ), 
                        attr='executed_statements', 
                        ctx=ast.Load()
                    ), 
                    attr='append', 
                    ctx=ast.Load()
                ),
                args=[ast.Constant(
                    value=node.lineno, 
                    kind=None
                )],
                keywords=[]
            )
        )
        
        node.body.insert(0, append_node)
        node.orelse.insert(0, append_node)
        
        return node
