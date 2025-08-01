from pymilvus import connections, Collection
import json
import time

CONFIG = {
    "host": "",
    "port": "",
    "user": "",
    "password": "",
    "db_name": "",
    "collection_name": "",
    "backup_file": "",
    "batch_size": 1000
}

def connect():
    print("🔌 Connecting to Milvus...")
    connections.connect(
        alias="default",
        host=CONFIG["host"],
        port=CONFIG["port"],
        user=CONFIG["user"],
        password=CONFIG["password"],
        db_name=CONFIG["db_name"]
    )

def get_primary_key_field(collection):
    return next((f.name for f in collection.schema.fields if f.is_primary), "pk")

def fetch_first_16k(collection, pk_field):
    print("📥 Fetching first 16,384 entries using offset...")
    docs = []
    pks = []
    MAX_OFFSET = 15384  # so offset+limit never exceeds 16384

    for offset in range(0, MAX_OFFSET + 1, CONFIG["batch_size"]):
        limit = min(CONFIG["batch_size"], 16384 - offset)
        print(f" → Fetching offset={offset}, limit={limit}")
        batch = collection.query(
            expr="",
            output_fields=[f.name for f in collection.schema.fields],
            offset=offset,
            limit=limit
        )
        docs.extend(batch)
        pks.extend(doc[pk_field] for doc in batch)
        time.sleep(0.1)

    return docs, pks

def fetch_remaining_by_pk_range(collection, pk_field, already_fetched_pks):
    print("📥 Fetching remaining documents using `pk > last_pk`...")
    docs = []
    seen = set(already_fetched_pks)

    # Use the last sorted PK from already fetched ones
    sorted_pks = sorted(seen)
    last_pk = sorted_pks[-1] if sorted_pks else ""
    print(f" → Starting from pk > '{last_pk}'")

    while True:
        expr = f'{pk_field} > "{last_pk}"'
        batch = collection.query(
            expr=expr,
            output_fields=[f.name for f in collection.schema.fields],
            limit=CONFIG["batch_size"]
        )

        if not batch:
            break

        new_docs = [doc for doc in batch if doc[pk_field] not in seen]
        if not new_docs:
            break

        docs.extend(new_docs)
        for doc in new_docs:
            seen.add(doc[pk_field])

        last_pk = sorted(doc[pk_field] for doc in new_docs)[-1]
        print(f" → Fetched {len(new_docs)} more docs, total so far: {len(docs)}")

        if len(new_docs) < CONFIG["batch_size"]:
            break

        time.sleep(0.1)

    return docs


def save_to_file(docs):
    print(f"💾 Saving {len(docs)} documents to file: {CONFIG['backup_file']}")
    with open(CONFIG["backup_file"], "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("✅ Backup complete.")

def check_howmany_docs_inJSON():
    try:
        with open(CONFIG["backup_file"], "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"📄 Total documents in JSON: {len(data)}")
    except FileNotFoundError:
        print("⚠️ Backup file not found. Please run the script to create it first.")
    except json.JSONDecodeError:
        print("⚠️ Error decoding JSON. The backup file may be corrupted.")

def main():
    # connect()
    # collection = Collection(CONFIG["collection_name"])
    # pk_field = get_primary_key_field(collection)

    # docs_phase1, pks_phase1 = fetch_first_16k(collection, pk_field)
    # docs_phase2 = fetch_remaining_by_pk_range(collection, pk_field, pks_phase1)

    # all_docs = docs_phase1 + docs_phase2
    # save_to_file(all_docs)
    check_howmany_docs_inJSON()

if __name__ == "__main__":
    main()
