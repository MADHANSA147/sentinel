import requests

datasets = [
    ("case-dataset2", "whatsapp_synthetic_35_messages.json"),
    ("case-dataset3", "corrupted_whatsapp_30_messages.json"),
    ("case-dataset4", "network_test_45_messages.json"),
]

for case_id, filename in datasets:
    path = "data/synthetic/" + filename
    with open(path, "rb") as f:
        r = requests.post("http://localhost:8000/api/v1/ingest", files={"file": f})
    data = r.json()
    print(case_id + ": " + filename)
    print("  total=" + str(data["total"]) + " validated=" + str(data["validated"]) + " quarantined=" + str(data["quarantined"]))
    print("  sha256=" + data["sha256_hash"][:16] + "...")
    print()

print("All datasets ingested successfully!")
