"""Test-wide settings: never load the heavy phase backends from /health during the tests."""

import os

os.environ.setdefault("ARCA_RAG_DISABLED", "1")
os.environ.setdefault("ARCA_TOOLS_DISABLED", "1")
os.environ.setdefault("ARCA_AGENT_DISABLED", "1")
# An empty value is "set", so the API's load_dotenv() does not fill it from a local .env and
# /health reports phase 1 as pending instead of loading the 1.7B model.
os.environ["ARCA_RLM_ADAPTER"] = ""
