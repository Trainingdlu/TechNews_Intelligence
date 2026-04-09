import os
import sys
import warnings

# Add the project root directory to sys.path so that tests can import the modules
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ignore LangGraph deprecation warnings temporarily to avoid test collection failures
warnings.filterwarnings("ignore", message=".*AgentStatePydantic has been moved.*")
