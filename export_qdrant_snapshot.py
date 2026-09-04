"""
Qdrant Snapshot Tool for PT-RAG.

Quickly exports or restores the consolidated `pt_books` collection snapshot.

Usage:
  1. Create a snapshot locally:
     python export_qdrant_snapshot.py --create

  2. List snapshots:
     python export_qdrant_snapshot.py --list

  3. Restore a snapshot on the server:
     python export_qdrant_snapshot.py --restore path/to/pt_books.snapshot
"""

import argparse
import sys
import qdrant_store


def main():
    parser = argparse.ArgumentParser(description="PT-RAG Qdrant Snapshot Tool")
    parser.add_argument("--create", "-c", action="store_true", help="Create snapshot of pt_books collection")
    parser.add_argument("--list", "-l", action="store_true", help="List all snapshots for pt_books")
    parser.add_argument("--restore", "-r", type=str, help="Restore collection from a snapshot file path or URL")

    args = parser.parse_args()
    client = qdrant_store.get_client()
    collection_name = qdrant_store.QDRANT_COLLECTION

    try:
        if args.create:
            print(f"Creating snapshot for collection '{collection_name}'...")
            snapshot = client.create_snapshot(collection_name=collection_name)
            print(f"✓ Snapshot created successfully:")
            print(f"  Name: {snapshot.name}")
            print(f"  Size: {snapshot.size / (1024*1024):.2f} MB" if snapshot.size else "")
            print(f"  Creation time: {snapshot.creation_time}")
            return 0

        if args.list:
            snapshots = client.list_snapshots(collection_name=collection_name)
            print(f"Snapshots for '{collection_name}':")
            for s in snapshots:
                print(f"  - {s.name} ({s.creation_time})")
            return 0

        if args.restore:
            print(f"Restoring collection '{collection_name}' from snapshot: {args.restore}...")
            client.recover_snapshot(
                collection_name=collection_name,
                location=args.restore,
            )
            count = qdrant_store.count_points(client, collection_name)
            print(f"✓ Collection restored successfully with {count:,} points.")
            return 0

        parser.print_help()
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
