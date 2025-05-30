from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, RequestError
import logging
import json

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Elasticsearch client for molecular search operations"""
    
    def __init__(self, host: str = 'localhost', port: int = 9200):
        """Initialize Elasticsearch client"""
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to Elasticsearch"""
        try:
            self.client = Elasticsearch([{'host': self.host, 'port': self.port}])
            if not self.client.ping():
                logger.error(f"Could not connect to Elasticsearch at {self.host}:{self.port}")
                return False
            return True
        except ConnectionError as e:
            logger.error(f"Elasticsearch connection error: {e}")
            return False
    
    def create_index(self, index_name: str, mapping: Dict) -> bool:
        """Create an index with the given mapping"""
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
            
            self.client.indices.create(index=index_name, body={'mappings': mapping})
            return True
        except RequestError as e:
            logger.error(f"Error creating index {index_name}: {e}")
            return False
    
    def index_document(self, index_name: str, document: Dict, doc_id: Optional[str] = None) -> bool:
        """Index a document in the specified index"""
        try:
            self.client.index(index=index_name, body=document, id=doc_id)
            return True
        except RequestError as e:
            logger.error(f"Error indexing document: {e}")
            return False
    
    def search(self, index_name: str, query: Dict, size: int = 10) -> List[Dict]:
        """Perform a search query"""
        try:
            results = self.client.search(index=index_name, body=query, size=size)
            return results['hits']['hits']
        except RequestError as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    def bulk_index(self, index_name: str, documents: List[Dict]) -> bool:
        """Index multiple documents in bulk"""
        try:
            actions = []
            for doc in documents:
                action = {
                    "index": {
                        "_index": index_name
                    }
                }
                actions.extend([action, doc])
            
            self.client.bulk(body=actions)
            return True
        except RequestError as e:
            logger.error(f"Error performing bulk index: {e}")
            return False
    
    def get_document(self, index_name: str, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID"""
        try:
            result = self.client.get(index=index_name, id=doc_id)
            return result['_source']
        except RequestError as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_index(self, index_name: str) -> bool:
        """Delete an index"""
        try:
            self.client.indices.delete(index=index_name)
            return True
        except RequestError as e:
            logger.error(f"Error deleting index {index_name}: {e}")
            return False
    
    def update_document(self, index_name: str, doc_id: str, document: Dict) -> bool:
        """Update an existing document"""
        try:
            self.client.update(index=index_name, id=doc_id, body={'doc': document})
            return True
        except RequestError as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
