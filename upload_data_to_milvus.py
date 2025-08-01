#!/usr/bin/env python3
"""
Improved Milvus Collection Migration Script
This script downloads all data from an existing collection, drops it, 
creates a new collection with updated schema (nullable fields), and uploads the data back.
"""

import json
import time
import numpy as np
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MilvusMigrator:
    def __init__(self, host: str = "localhost", port: str = "19530", user: str = "", password: str = "", db_name: str = "default"):
        """Initialize Milvus connection"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.connection = None
        
    def connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.db_name
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}, database: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False
    
    def backup_collection_data_complete(self, collection_name: str) -> List[Dict]:
        """
        Complete backup using multiple strategies to get ALL data
        """
        try:
            logger.info(f"Starting complete backup of collection: {collection_name}")
            
            if not utility.has_collection(collection_name):
                logger.error(f"Collection {collection_name} does not exist")
                return []
            
            collection = Collection(collection_name)
            collection.load()
            
            count = collection.num_entities
            logger.info(f"Total entities to backup: {count}")
            
            if count == 0:
                return []
            
            all_data = []
            
            # Strategy 1: Use query with pagination up to the limit
            logger.info("Strategy 1: Using query with pagination...")
            query_data = self._backup_with_query_pagination(collection, count)
            all_data.extend(query_data)
            
            # Strategy 2: If we didn't get all data, use search-based approach
            if len(all_data) < count:
                logger.info(f"Got {len(all_data)}/{count} with query. Using search strategy for remaining...")
                search_data = self._backup_with_search_strategy(collection, all_data, count)
                all_data.extend(search_data)
            
            # Strategy 3: If still missing data, use iterator approach
            if len(all_data) < count:
                logger.info(f"Got {len(all_data)}/{count} total. Using iterator strategy for remaining...")
                iterator_data = self._backup_with_iterator(collection, all_data, count)
                all_data.extend(iterator_data)
            
            # Strategy 4: If still missing data, use expression-based queries
            if len(all_data) < count:
                logger.info(f"Got {len(all_data)}/{count} total. Using expression-based strategy for remaining...")
                expr_data = self._backup_with_expression_queries(collection, all_data, count)
                all_data.extend(expr_data)
            
            # Strategy 5: If still missing data, use range-based queries on numeric fields
            if len(all_data) < count:
                logger.info(f"Got {len(all_data)}/{count} total. Using range-based strategy for remaining...")
                range_data = self._backup_with_range_queries(collection, all_data, count)
                all_data.extend(range_data)
            
            logger.info(f"Final backup result: {len(all_data)}/{count} entities")
            return all_data
            
        except Exception as e:
            logger.error(f"Error in complete backup: {e}")
            return []
    
    def _backup_with_query_pagination(self, collection: Collection, total_count: int) -> List[Dict]:
        """Strategy 1: Query with pagination (limited by 16384 window)"""
        all_data = []
        batch_size = 1000
        max_offset = min(16384 - batch_size, total_count)
        
        offset = 0
        while offset <= max_offset:
            try:
                limit = min(batch_size, total_count - offset)
                if limit <= 0:
                    break
                
                results = collection.query(
                    expr="",
                    offset=offset,
                    limit=limit,
                    output_fields=["*"]
                )
                
                all_data.extend(results)
                offset += limit
                logger.info(f"Query strategy: {len(all_data)} entities backed up")
                
            except Exception as e:
                logger.error(f"Query failed at offset {offset}: {e}")
                break
        
        return all_data
    
    def _backup_with_search_strategy(self, collection: Collection, existing_data: List[Dict], total_count: int) -> List[Dict]:
        """Strategy 2: Use search to find remaining entities"""
        try:
            existing_pks = {str(item.get('pk')) for item in existing_data}
            logger.info(f"Search strategy: Looking for {total_count - len(existing_data)} remaining entities")
            
            # Get schema to determine vector dimension
            schema = collection.schema
            vector_field = None
            vector_dim = 1024  # default
            
            for field in schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    vector_field = field.name
                    vector_dim = field.params.get('dim', 1024)
                    break
            
            if not vector_field:
                logger.error("No vector field found for search strategy")
                return []
            
            # Use multiple random vectors to search different parts of the space
            remaining_data = []
            search_rounds = 10  # Try multiple search vectors
            
            for round_num in range(search_rounds):
                try:
                    # Generate random search vector
                    dummy_vector = np.random.random(vector_dim).tolist()
                    
                    search_results = collection.search(
                        data=[dummy_vector],
                        anns_field=vector_field,
                        param={"metric_type": "IP", "params": {"nprobe": 128}},
                        limit=16384,
                        output_fields=["*"]
                    )
                    
                    new_entities = 0
                    for hit in search_results[0]:
                        pk = str(hit.entity.get('pk'))
                        if pk not in existing_pks:
                            entity_dict = {}
                            for field_name in hit.entity.fields:
                                entity_dict[field_name] = hit.entity.get(field_name)
                            remaining_data.append(entity_dict)
                            existing_pks.add(pk)
                            new_entities += 1
                    
                    logger.info(f"Search round {round_num + 1}: Found {new_entities} new entities")
                    
                    if len(existing_data) + len(remaining_data) >= total_count:
                        break
                        
                except Exception as e:
                    logger.error(f"Search round {round_num + 1} failed: {e}")
                    continue
            
            logger.info(f"Search strategy found {len(remaining_data)} additional entities")
            return remaining_data
            
        except Exception as e:
            logger.error(f"Error in search strategy: {e}")
            return []
    
    def _backup_with_iterator(self, collection: Collection, existing_data: List[Dict], total_count: int) -> List[Dict]:
        """Strategy 3: Use query iterator (if available in your Milvus version)"""
        try:
            existing_pks = {str(item.get('pk')) for item in existing_data}
            remaining_data = []
            
            # Try using query iterator (available in newer Milvus versions)
            try:
                from pymilvus import QueryIterator
                
                iterator = collection.query_iterator(
                    batch_size=1000,
                    output_fields=["*"]
                )
                
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    
                    for entity in batch:
                        pk = str(entity.get('pk'))
                        if pk not in existing_pks:
                            remaining_data.append(entity)
                            existing_pks.add(pk)
                
                iterator.close()
                logger.info(f"Iterator strategy found {len(remaining_data)} additional entities")
                
            except ImportError:
                logger.info("QueryIterator not available, skipping iterator strategy")
            except Exception as e:
                logger.error(f"Iterator strategy failed: {e}")
            
            return remaining_data
            
        except Exception as e:
            logger.error(f"Error in iterator strategy: {e}")
            return []
    
    def _backup_with_expression_queries(self, collection: Collection, existing_data: List[Dict], total_count: int) -> List[Dict]:
        """Strategy 4: Use expression-based queries to find remaining entities"""
        try:
            existing_pks = {str(item.get('pk')) for item in existing_data}
            remaining_data = []
            
            logger.info(f"Expression strategy: Looking for {total_count - len(existing_data)} remaining entities")
            
            # Get some sample PKs to create NOT IN expressions
            if existing_data:
                # Create batches of PKs to exclude
                pk_list = list(existing_pks)
                batch_size = 100  # Milvus has limits on expression length
                
                for i in range(0, len(pk_list), batch_size):
                    try:
                        pk_batch = pk_list[i:i + batch_size]
                        # Create NOT IN expression
                        pk_str_list = [f'"{pk}"' for pk in pk_batch]
                        expr = f"pk not in [{','.join(pk_str_list)}]"
                        
                        # Query with this expression
                        results = collection.query(
                            expr=expr,
                            limit=1000,  # Small limit since we're doing many queries
                            output_fields=["*"]
                        )
                        
                        new_entities = 0
                        for entity in results:
                            pk = str(entity.get('pk'))
                            if pk not in existing_pks:
                                remaining_data.append(entity)
                                existing_pks.add(pk)
                                new_entities += 1
                        
                        if new_entities > 0:
                            logger.info(f"Expression batch {i//batch_size + 1}: Found {new_entities} new entities")
                        
                        if len(existing_data) + len(remaining_data) >= total_count:
                            break
                            
                    except Exception as e:
                        logger.error(f"Expression query failed for batch {i//batch_size + 1}: {e}")
                        continue
            
            logger.info(f"Expression strategy found {len(remaining_data)} additional entities")
            return remaining_data
            
        except Exception as e:
            logger.error(f"Error in expression strategy: {e}")
            return []
    
    def _backup_with_range_queries(self, collection: Collection, existing_data: List[Dict], total_count: int) -> List[Dict]:
        """Strategy 5: Use range-based queries on numeric/string fields"""
        try:
            existing_pks = {str(item.get('pk')) for item in existing_data}
            remaining_data = []
            
            logger.info(f"Range strategy: Looking for {total_count - len(existing_data)} remaining entities")
            
            # Try different field-based queries
            schema = collection.schema
            
            # 1. Try queries on the index field (numeric)
            try:
                # Get min/max values from existing data for index field
                if existing_data:
                    index_values = [item.get('index', 0) for item in existing_data if item.get('index') is not None]
                    if index_values:
                        min_idx = min(index_values)
                        max_idx = max(index_values)
                        
                        # Query ranges that might have been missed
                        ranges = [
                            f"index < {min_idx}",
                            f"index > {max_idx}",
                            f"index >= 0 and index <= {min_idx // 2}" if min_idx > 0 else None,
                            f"index >= {max_idx * 2}" if max_idx > 0 else None
                        ]
                        
                        for range_expr in ranges:
                            if range_expr is None:
                                continue
                            try:
                                results = collection.query(
                                    expr=range_expr,
                                    limit=5000,
                                    output_fields=["*"]
                                )
                                
                                new_entities = 0
                                for entity in results:
                                    pk = str(entity.get('pk'))
                                    if pk not in existing_pks:
                                        remaining_data.append(entity)
                                        existing_pks.add(pk)
                                        new_entities += 1
                                
                                if new_entities > 0:
                                    logger.info(f"Range query '{range_expr}': Found {new_entities} new entities")
                                
                            except Exception as e:
                                logger.error(f"Range query '{range_expr}' failed: {e}")
                                continue
            except Exception as e:
                logger.error(f"Index range queries failed: {e}")
            
            # 2. Try queries on string fields with different patterns
            try:
                string_fields = ['label', 'title_ref', 'hash']
                for field in string_fields:
                    # Get some sample values
                    if existing_data:
                        field_values = [item.get(field, '') for item in existing_data[:100] if item.get(field)]
                        if field_values:
                            # Try different string patterns
                            patterns = []
                            for val in field_values[:5]:  # Just try a few
                                if len(val) > 1:
                                    patterns.extend([
                                        f'{field} like "{val[0]}%"',
                                        f'{field} like "%{val[-1]}"',
                                    ])
                            
                            for pattern in patterns[:10]:  # Limit patterns
                                try:
                                    results = collection.query(
                                        expr=pattern,
                                        limit=1000,
                                        output_fields=["*"]
                                    )
                                    
                                    new_entities = 0
                                    for entity in results:
                                        pk = str(entity.get('pk'))
                                        if pk not in existing_pks:
                                            remaining_data.append(entity)
                                            existing_pks.add(pk)
                                            new_entities += 1
                                    
                                    if new_entities > 0:
                                        logger.info(f"Pattern '{pattern}': Found {new_entities} new entities")
                                    
                                except Exception as e:
                                    continue  # Skip failed patterns
            except Exception as e:
                logger.error(f"String pattern queries failed: {e}")
            
            logger.info(f"Range strategy found {len(remaining_data)} additional entities")
            return remaining_data
            
        except Exception as e:
            logger.error(f"Error in range strategy: {e}")
            return []
    
    def save_backup_to_file(self, data: List[Dict], filename: str = "milvus_backup.json"):
        """Save backup data to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Backup saved to {filename} ({len(data)} entities)")
            return True
        except Exception as e:
            logger.error(f"Error saving backup to file: {e}")
            return False
    
    def load_backup_from_file(self, filename: str = "milvus_backup.json") -> List[Dict]:
        """Load backup data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            logger.info(f"Backup loaded from {filename} ({len(data)} entities)")
            return data
        except Exception as e:
            logger.error(f"Error loading backup from file: {e}")
            return []
    
    def create_new_collection_schema(self) -> CollectionSchema:
        """Create new collection schema with nullable fields"""
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                description="Primary key",
                is_primary=True, 
                auto_id=True,
                max_length=100
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8092),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=50), 
            FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=30),
            FieldSchema(name="group_ref", dtype=DataType.VARCHAR, max_length=100, nullable=True),  # Made nullable
            FieldSchema(name="title_ref", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="index", dtype=DataType.INT64),
            FieldSchema(
                name="dense_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=1024,
                description="Dense vector embeddings"
            )
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Migrated collection with nullable group_ref field"
        )
        
        return schema
    
    def drop_collection(self, collection_name: str) -> bool:
        """Drop the existing collection"""
        try:
            if utility.has_collection(collection_name):
                # Release collection first if it's loaded
                try:
                    collection = Collection(collection_name)
                    collection.release()
                    logger.info(f"Released collection: {collection_name}")
                except:
                    pass  # Collection might not be loaded
                
                utility.drop_collection(collection_name)
                logger.info(f"Successfully dropped collection: {collection_name}")
                return True
            else:
                logger.info(f"Collection {collection_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Error dropping collection: {e}")
            return False
    
    def create_new_collection(self, collection_name: str) -> bool:
        """Create new collection with updated schema"""
        try:
            # First, check if collection already exists and drop it
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists, dropping it first...")
                if not self.drop_collection(collection_name):
                    return False
                time.sleep(2)  # Wait for cleanup
            
            schema = self.create_new_collection_schema()
            
            collection = Collection(
                name=collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"Successfully created new collection: {collection_name}")
            
            # Create index for vector field
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            collection.create_index(
                field_name="dense_vector",
                index_params=index_params
            )
            
            logger.info("Created index for dense_vector field")
            return True
            
        except Exception as e:
            logger.error(f"Error creating new collection: {e}")
            return False
    
    def upload_data_to_collection(self, collection_name: str, data: List[Dict]) -> bool:
        """Upload data to the new collection"""
        try:
            if not data:
                logger.info("No data to upload")
                return True
            
            collection = Collection(collection_name)
            
            # Process data in batches
            batch_size = 1000
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            logger.info(f"Uploading {len(data)} entities in {total_batches} batches")
            
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                
                # Prepare batch data in the correct format
                batch_entities = []
                for entity in batch_data:
                    # Handle nullable fields properly
                    processed_entity = {}
                    for key, value in entity.items():
                        if key == 'group_ref' and value is None:
                            processed_entity[key] = None  # Keep as None for nullable field
                        # remove 'pk' if it's auto-generated
                        elif key == 'pk':
                            continue
                        else:
                            processed_entity[key] = value
                    batch_entities.append(processed_entity)
                
                # Insert batch
                collection.insert(batch_entities)
                
                batch_num = (i // batch_size) + 1
                logger.info(f"Uploaded batch {batch_num}/{total_batches}")
            
            # Flush to ensure data is persisted
            collection.flush()
            logger.info("Data upload completed and flushed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error uploading data: {e}")
            return False
        

    def read_file_data(self, filename: str = "milvus_backup.json") -> List[Dict]:
        """Read data from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entities from file: {filename}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {filename}")
            return []
    
    def migrate_collection(self, collection_name: str, backup_file: str = None) -> bool:
        """Complete migration process"""
        try:
            # logger.info(f"Starting migration for collection: {collection_name}")
            
            # Step 1: Backup existing data using improved method
            # backup_data = self.backup_collection_data_complete(collection_name + "_old")
            
            # if not backup_data:
            #     logger.error("No data was backed up, aborting migration")
            #     return False
            
            # logger.info(f"Successfully backed up {len(backup_data)} entities")
            
            # Step 2: Save backup to file (optional)
            # if backup_file:
            #     self.save_backup_to_file(backup_data, backup_file)

            backup_data = self.read_file_data("collection_backup_complete.json")
            
            # Step 3: Create new collection (will drop existing if needed)
            new_collection_name = collection_name + "_new"
            if not self.create_new_collection(new_collection_name):
                logger.error("Failed to create new collection, aborting migration")
                return False
            
            # Step 4: Upload data to new collection
            if not self.upload_data_to_collection(new_collection_name, backup_data):
                logger.error("Failed to upload data to new collection")
                return False
            
            logger.info(f"Migration completed successfully!")
            logger.info(f"Original collection: {collection_name}_old ({len(backup_data)} entities)")
            logger.info(f"New collection: {new_collection_name} ({len(backup_data)} entities)")
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

def main():
    """Main execution function"""
    # Configuration - Update these values for your setup
    CONFIG = {
        "host": "",
        "port": "",
        "user": "",
        "password": "",
        "db_name": "",
        "collection_name": "",
        "backup_file": ""
    }
    
    # Create migrator instance
    migrator = MilvusMigrator(
        host=CONFIG["host"],
        port=CONFIG["port"],
        user=CONFIG["user"],
        password=CONFIG["password"],
        db_name=CONFIG["db_name"]
    )
    
    # Connect to Milvus
    if not migrator.connect():
        logger.error("Failed to connect to Milvus, exiting")
        return
    
    # Perform migration
    success = migrator.migrate_collection(
        collection_name=CONFIG["collection_name"],
        backup_file=CONFIG["backup_file"]
    )
    
    if success:
        logger.info("🎉 Migration completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Verify the new collection has all data")
        logger.info("2. Test the new collection functionality")
        logger.info("3. Drop the old collection when satisfied")
    else:
        logger.error("❌ Migration failed!")

if __name__ == "__main__":
    main()