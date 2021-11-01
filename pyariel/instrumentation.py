import ast
from typing import Any


class Instrumenter(ast.NodeTransformer):
    """Will instrument an AST containing, such in each of the bodies of every if-else statement, the test condition's
     line number is added to the ordered executed statement list (the variable called executed_path)."""

    def visit_Module(self, node: ast.Module) -> Any:
        """ Declares the variable that stores the executed path.
        executed_path = []
        ...
        """
        self.generic_visit(node)
        declare_node = ast.Assign(
            targets=[ast.Name(id='path_lines', ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load())
        )
        node.body.insert(0, declare_node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """ Resets, and adds the executed path variable to the global scope, so it can be modified within the function.
        def f(x):
            global executed_path
            executed_path = []
            ...
        """
        self.generic_visit(node)
        global_node = ast.Global(names=['path_lines'])
        reset_node = ast.Assign(
            targets=[ast.Name(id='path_lines', ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load())
        )
        node.body.insert(0, reset_node)
        node.body.insert(0, global_node)
        return node

    def visit_If(self, node: ast.If) -> Any:
        """ Appends the line number of the statement's condition to executed path within the if and else body.
        if x: <--- e.g. line number, lineno = 3
            executed_path.append(lineno)
            ...
        else:
            executed_path.append(lineno)
            ...
        """
        self.generic_visit(node)
        append_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='path_lines', ctx=ast.Load()),
                    attr='append', ctx=ast.Load()
                ),
                args=[ast.Constant(value=node.lineno)],
                keywords=[]
            )
        )
        node.body.insert(0, append_node)
        node.orelse.insert(0, append_node)
        return node
