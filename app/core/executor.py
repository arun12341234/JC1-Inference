"""
executor.py - Secure Python Code Execution Engine
--------------------------------------------------
🔹 Features:
- Allows LLM to execute Python code dynamically
- Supports mathematical calculations, data processing, and function execution
- Runs inside a secure sandboxed environment
- Restricts access to sensitive modules and system calls

📌 Dependencies:
- RestrictedPython (for safer execution)
"""

import sys
import traceback
from RestrictedPython import safe_globals, compile_restricted

### 🔧 SECURITY RESTRICTIONS ###
# Whitelisted functions & safe built-ins
SAFE_GLOBALS = {
    "__builtins__": {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "next": next,
        "all": all,
        "any": any,
    },
    "print": lambda *args: args,  # Disable actual print (prevents logging abuse)
}

### 📂 CODE EXECUTOR CLASS ###
class CodeExecutor:
    """
    A secure sandboxed environment for executing Python code.
    """

    def __init__(self):
        self.globals = safe_globals.copy()

    def execute(self, code: str) -> dict:
        """
        Executes the given Python code securely.

        :param code: Python code string to execute
        :return: Dictionary containing 'output' or 'error'
        """
        try:
            compiled_code = compile_restricted(code, filename="<executor>", mode="exec")

            # Local namespace (isolated environment)
            local_namespace = {}

            # Execute in restricted environment
            exec(compiled_code, self.globals, local_namespace)

            return {"output": local_namespace}
        except Exception as e:
            return {"error": traceback.format_exc()}


### 🛠️ EXAMPLE USAGE ###
if __name__ == "__main__":
    executor = CodeExecutor()

    # Safe Code Execution
    safe_code = """
x = sum([1, 2, 3, 4])
y = max(10, 20, 30)
result = x * y
"""
    print(f"Safe Execution: {executor.execute(safe_code)}")

    # Malicious Code Attempt (should be blocked)
    unsafe_code = """
import os
os.system('rm -rf /')  # This should NOT execute
"""
    print(f"Unsafe Execution: {executor.execute(unsafe_code)}")
