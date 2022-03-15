import ast
import inspect
import pandas as pd

from autotest.auto_test import AutoTest
from autotest.model.modules.planning import Planning
from pyariel.instrumentation import Instrumenter


if __name__ == '__main__':
    
    
    source = inspect.getsource(Planning)  # Retrieve the source code of the rule set.
    rule_set = ast.parse(source)  # Parse the source of the rule set into an AST.
    Instrumenter().visit(rule_set)  # Instrument the rule set.
    ast.fix_missing_locations(rule_set)  # Fix the missing line numbers and other fields in the AST.

    code = compile(rule_set, '<ast>', 'exec')  # Compile the instrumented AST of the rule set.
    exec(code)  # Execute the compiled AST of the rule set in the given scope.
    callable_rule_set = test_scope['rule_set']  # Extract the rule set function definition from the scope.
    
    print(source)
    print(rule_set())
    
    scene_ids = [11]  # 96
    autotest = AutoTest()
    results = autotest.run(scene_ids)
    print(pd.DataFrame.from_dict(results))
