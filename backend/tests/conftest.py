"""
pytest conftest.py — mocks Neo4j and ChromaDB for offline unit tests.
Gap Detector and ingestion tests run without any external services.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Add backend root to sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock neo4j in sys.modules so it doesn't need to be installed
mock_neo4j = MagicMock()
mock_neo4j.GraphDatabase = MagicMock()
mock_neo4j.Driver = MagicMock()
sys.modules["neo4j"] = mock_neo4j

# ── Mock Neo4j driver so tests don't need a real DB ───────────────────────
import app.services.graph_db as gdb

_mock_driver = MagicMock()
_mock_session = MagicMock()
_mock_session.__enter__ = lambda self: self
_mock_session.__exit__ = MagicMock(return_value=False)
_mock_driver.session.return_value = _mock_session
gdb._driver = _mock_driver


def mock_run_query(cypher, params=None):
    """Return empty list by default — override in individual tests if needed."""
    return []

gdb.run_query = mock_run_query
gdb.write_person_properties = MagicMock()
gdb.batch_load_graph = MagicMock()
