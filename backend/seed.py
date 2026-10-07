import asyncio
from sampledata import get_all_samples_manifest
from db import Database

async def main():
    db = Database()
    connected = await db.connect()
    if not connected:
        print("MongoDB not available. Nothing was seeded.")
        return
    for sample in get_all_samples_manifest():
        await db.insert_analysis({
            "type": "synthetic_sample",
            "sample_id": sample["sample_id"],
            "label": sample["label"],
            "synthetic": True
        })
    print("Synthetic sample metadata seeded.")

if __name__ == "__main__":
    asyncio.run(main())
