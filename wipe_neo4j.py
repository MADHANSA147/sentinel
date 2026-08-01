"""Wipe all Person nodes and MESSAGED edges, then re-ingest all 4 datasets."""
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "backend", ".env"))

uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USER"]
password = os.environ["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(uri, auth=(user, password))

print("Wiping Neo4j database...")
with driver.session() as session:
    result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
    deleted = result.single()["deleted"]
    print(f"  Deleted {deleted} nodes.")

driver.close()
print("Done. Ready to re-ingest.")
