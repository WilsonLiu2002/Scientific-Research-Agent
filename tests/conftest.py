import os


# Unit and regression tests must never consume live provider quota from a developer's `.env`.
os.environ["SCIENTIFIC_AGENT_OFFLINE"] = "1"
