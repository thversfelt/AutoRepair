import ast
import random
from typing import Any


class Instrumenter(ast.NodeTransformer):
    """Will instrument an AST containing, such in each of the bodies of every if-else statement, the test condition's
     line number is added to the ordered executed statement list (the variable called executed_path)."""
    def visit_Module(self, node: ast.Module) -> Any:
        self.generic_visit(node)
        declare_node = ast.Assign(
            targets=[ast.Name(id='executed_path', ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load())
        )
        node.body.insert(0, declare_node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.generic_visit(node)
        global_node = ast.Global(names=['executed_path'])
        reset_node = ast.Assign(
            targets=[ast.Name(id='executed_path', ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load())
        )
        node.body.insert(0, reset_node)
        node.body.insert(0, global_node)
        return node

    def visit_If(self, node: ast.If) -> Any:
        self.generic_visit(node)
        append_node = ast.Expr(  # executed_path.append(lineno), where lineno = the test condition's line number.
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='executed_path', ctx=ast.Load()),
                    attr='append', ctx=ast.Load()
                ),
                args=[ast.Constant(value=node.lineno)],
                keywords=[]
            )
        )
        node.body.insert(0, append_node)
        node.orelse.insert(0, append_node)
        return node


class ModifyMutator(ast.NodeTransformer):
    """Implements the three modify mutation operators."""
    def visit_Compare(self, node: ast.Compare) -> Any:
        modifications = [
            self.change_threshold_value,
            self.change_relational_direction,
            self.change_arithmetic_operation
        ]
        modification = random.choice(modifications)
        return modification(node)

    def change_threshold_value(self, node: ast.Compare) -> Any:
        constant = node.comparators[0]
        constant.value = random.randint(-100, 100)
        return node

    def change_relational_direction(self, node: ast.Compare):
        relational_operator = type(node.ops[0])
        inverse = {
            ast.Eq: ast.NotEq(),
            ast.NotEq: ast.Eq(),
            ast.Lt: ast.Gt(),
            ast.Gt: ast.Lt(),
            ast.LtE: ast.GtE(),
            ast.GtE: ast.LtE(),
        }
        new_relational_operator = inverse[relational_operator]
        node.ops = [new_relational_operator]
        return node

    def change_arithmetic_operation(self, node: ast.Compare) -> Any:
        inverse = {
            ast.Add: ast.Sub(),
            ast.Sub: ast.Add(),
            ast.Mult: ast.Div(),
            ast.Div: ast.Mult()
        }

        binary_operations = []
        for child in ast.iter_child_nodes(node):
            if type(child) == ast.BinOp and type(child.op) in inverse.keys():
                binary_operations.append(child)

        if len(binary_operations) > 0:
            binary_operation = random.choice(binary_operations)
            arithmetic_operation = type(binary_operation.op)
            binary_operation.op = inverse[arithmetic_operation]

        return node


class ShiftMutator(ast.NodeTransformer):
    """Implements the shift mutation operator."""
    pass
