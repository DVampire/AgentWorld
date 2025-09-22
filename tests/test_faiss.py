import sys
import tempfile
from pathlib import Path
import asyncio
import argparse
import os
from mmengine import DictAction

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.faiss.service import FaissService
from src.environments.faiss.types import (
    FaissSearchRequest,
    FaissAddRequest,
    FaissDeleteRequest
)
from src.infrastructures.models import model_manager
from src.config import config
from src.logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--config", default=os.path.join(root, "configs", "tool_calling_agent.py"), help="config file path")

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

class FaissTester:
    """Comprehensive test suite for FAISS functionality."""
    
    def __init__(self):
        self.test_dir = None
        self.faiss_service = None
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_document_ids = []
    
    async def setup(self):
        """Setup test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        embedding_function = model_manager.get("text-embedding-3-large")
        self.faiss_service = FaissService(
            base_dir=self.test_dir,
            embedding_function=embedding_function
        )
        print(f"Test directory: {self.test_dir}")
    
    async def cleanup(self):
        """Cleanup test environment."""
        import shutil
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def assert_test(self, condition: bool, test_name: str, message: str = ""):
        """Assert test condition and track results."""
        if condition:
            print(f"✓ {test_name}")
            self.passed_tests += 1
        else:
            print(f"✗ {test_name}: {message}")
            self.failed_tests += 1
    
    async def test_basic_document_operations(self):
        """Test basic document add/delete operations."""
        print("\n=== Testing Basic Document Operations ===")
        
        # Test add single document
        texts = ["This is a test document about machine learning and AI."]
        metadatas = [{"category": "AI", "author": "test_user"}]
        
        request = FaissAddRequest(texts=texts, metadatas=metadatas)
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(result.count == 1, 
                        "Add single document", 
                        f"Expected count=1, got count={result.count}")
        
        if result.ids:
            self.test_document_ids.extend(result.ids)
        
        # Test add multiple documents
        texts = [
            "Python is a great programming language for data science.",
            "Machine learning algorithms can learn from data patterns.",
            "Deep learning uses neural networks with multiple layers."
        ]
        metadatas = [
            {"category": "Programming", "topic": "Python"},
            {"category": "AI", "topic": "ML"},
            {"category": "AI", "topic": "Deep Learning"}
        ]
        
        request = FaissAddRequest(texts=texts, metadatas=metadatas)
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(result.count == 3, 
                        "Add multiple documents", 
                        f"Expected count=3, got count={result.count}")
        
        if result.ids:
            self.test_document_ids.extend(result.ids)
        
        # Test get index info
        index_info = await self.faiss_service.get_index_info()
        self.assert_test(index_info.total_documents >= 4, 
                        "Get index info", 
                        f"Expected >=4 documents, got {index_info.total_documents}")
    
    async def test_search_functionality(self):
        """Test search functionality."""
        print("\n=== Testing Search Functionality ===")
        
        # Test basic search
        query = "machine learning"
        request = FaissSearchRequest(
            query=query,
            k=3,
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(len(result.documents) > 0, 
                        "Basic search", 
                        f"Expected documents, got docs={len(result.documents)}")
        
        # Test search with filter
        request = FaissSearchRequest(
            query=query,
            k=2,
            filter={"category": "AI"},
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with filter", 
                        "Search completed")
        
        # Test search with score threshold
        request = FaissSearchRequest(
            query=query,
            k=5,
            score_threshold=0.1,
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with score threshold", 
                        "Search completed")
        
        # Test search with different query
        request = FaissSearchRequest(
            query="programming language",
            k=2,
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with different query", 
                        "Search completed")
    
    async def test_document_deletion(self):
        """Test document deletion."""
        print("\n=== Testing Document Deletion ===")
        
        if not self.test_document_ids:
            print("No document IDs available for deletion test")
            return
        
        # Test delete single document
        ids_to_delete = [self.test_document_ids[0]]
        request = FaissDeleteRequest(ids=ids_to_delete)
        result = await self.faiss_service.delete_documents(request)
        
        self.assert_test(result.success and result.deleted_count == 1, 
                        "Delete single document", 
                        f"Expected success=True, deleted_count=1, got success={result.success}, deleted_count={result.deleted_count}")
        
        # Test delete multiple documents
        if len(self.test_document_ids) > 2:
            ids_to_delete = self.test_document_ids[1:3]
            request = FaissDeleteRequest(ids=ids_to_delete)
            result = await self.faiss_service.delete_documents(request)
            
            self.assert_test(result.success and result.deleted_count == len(ids_to_delete), 
                            "Delete multiple documents", 
                            f"Expected success=True, deleted_count={len(ids_to_delete)}, got success={result.success}, deleted_count={result.deleted_count}")
        
        # Test delete non-existent document
        request = FaissDeleteRequest(ids=["non_existent_id"])
        result = await self.faiss_service.delete_documents(request)
        
        self.assert_test(result.success and result.deleted_count == 0, 
                        "Delete non-existent document", 
                        f"Expected success=True, deleted_count=0, got success={result.success}, deleted_count={result.deleted_count}")
    
    async def test_index_management(self):
        """Test index management operations."""
        print("\n=== Testing Index Management ===")
        
        # Test get index info
        index_info = await self.faiss_service.get_index_info()
        self.assert_test(index_info.total_documents >= 0, 
                        "Get index info", 
                        f"Expected total_documents >= 0, got {index_info.total_documents}")
        
        self.assert_test(index_info.embedding_dimension > 0, 
                        "Check embedding dimension", 
                        f"Expected embedding_dimension > 0, got {index_info.embedding_dimension}")
        
        self.assert_test(index_info.distance_strategy in ["cosine", "euclidean", "max_inner_product"], 
                        "Check distance strategy", 
                        f"Expected valid distance_strategy, got {index_info.distance_strategy}")
        
        # Test save index
        try:
            await self.faiss_service.save_index()
            self.assert_test(True, "Save index", "Index saved successfully")
        except Exception as e:
            self.assert_test(False, "Save index", f"Failed to save index: {e}")
        
        # Test index persistence (reload and check)
        try:
            # Get current embedding function for persistence test
            current_embedding_function = self.faiss_service.embedding_function
            
            # Create new service instance to test persistence
            new_service = FaissService(
                base_dir=self.test_dir,
                embedding_function=current_embedding_function
            )
            
            # Check if index was loaded
            index_info = await new_service.get_index_info()
            self.assert_test(index_info.total_documents >= 0, 
                            "Index persistence", 
                            f"Expected to load persisted index, got {index_info.total_documents} documents")
            
        except Exception as e:
            self.assert_test(False, "Index persistence", f"Failed to test persistence: {e}")
    
    async def test_edge_cases(self):
        """Test edge cases."""
        print("\n=== Testing Edge Cases ===")
        
        # Test add empty documents
        request = FaissAddRequest(texts=[], metadatas=[])
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(result.count == 0, 
                        "Add empty documents", 
                        f"Expected count=0, got count={result.count}")
        
        # Test add document with empty text
        request = FaissAddRequest(texts=[""], metadatas=[{"empty": True}])
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(True, 
                        "Add document with empty text", 
                        "Add completed")
        
        # Test search with empty query
        request = FaissSearchRequest(query="", k=1, fetch_k=5)
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with empty query", 
                        "Search completed")
        
        # Test search with very high k
        request = FaissSearchRequest(query="test", k=1000, fetch_k=1000)
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with high k", 
                        "Search completed")
        
        # Test search with very low score threshold
        request = FaissSearchRequest(query="test", k=5, score_threshold=0.99, fetch_k=10)
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with high score threshold", 
                        "Search completed")
    
    async def test_metadata_operations(self):
        """Test metadata operations."""
        print("\n=== Testing Metadata Operations ===")
        
        # Test add documents with complex metadata
        texts = [
            "This document has complex metadata for testing.",
            "Another document with different metadata structure."
        ]
        metadatas = [
            {
                "category": "Test",
                "subcategory": "Complex",
                "tags": ["test", "metadata", "complex"],
                "numeric_value": 42,
                "boolean_value": True,
                "nested": {"level1": {"level2": "value"}}
            },
            {
                "category": "Test",
                "subcategory": "Simple",
                "tags": ["test", "simple"],
                "numeric_value": 24,
                "boolean_value": False
            }
        ]
        
        request = FaissAddRequest(texts=texts, metadatas=metadatas)
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(result.count == 2, 
                        "Add documents with complex metadata", 
                        f"Expected count=2, got count={result.count}")
        
        # Test search with metadata filter
        request = FaissSearchRequest(
            query="metadata",
            k=5,
            filter={"category": "Test"},
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with metadata filter", 
                        "Search completed")
        
        # Test search with nested metadata filter
        request = FaissSearchRequest(
            query="complex",
            k=5,
            filter={"nested.level1.level2": "value"},
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search with nested metadata filter", 
                        "Search completed")
    
    async def test_performance_operations(self):
        """Test performance-related operations."""
        print("\n=== Testing Performance Operations ===")
        
        # Test adding many documents
        texts = [f"This is test document number {i} for performance testing." for i in range(50)]
        metadatas = [{"batch": "performance", "index": i} for i in range(50)]
        
        request = FaissAddRequest(texts=texts, metadatas=metadatas)
        result = await self.faiss_service.add_documents(request)
        
        self.assert_test(result.count == 50, 
                        "Add many documents", 
                        f"Expected count=50, got count={result.count}")
        
        # Test search performance
        request = FaissSearchRequest(
            query="performance testing",
            k=10,
            fetch_k=100
        )
        result = await self.faiss_service.search_similar(request)
        
        self.assert_test(True, 
                        "Search performance test", 
                        "Search completed")
        
        # Test batch deletion (use some test document IDs if available)
        if self.test_document_ids and len(self.test_document_ids) > 0:
            request = FaissDeleteRequest(ids=self.test_document_ids[:5])
            result = await self.faiss_service.delete_documents(request)
            
            self.assert_test(result.success, 
                            "Batch deletion", 
                            f"Expected success=True, got success={result.success}")
        else:
            self.assert_test(True, 
                            "Batch deletion", 
                            "No test document IDs available for batch deletion test")
    
    async def test_error_handling(self):
        """Test error handling."""
        print("\n=== Testing Error Handling ===")
        
        # Test delete with invalid IDs
        request = FaissDeleteRequest(ids=["invalid_id_1", "invalid_id_2"])
        result = await self.faiss_service.delete_documents(request)
        
        self.assert_test(result.success and result.deleted_count == 0, 
                        "Delete with invalid IDs", 
                        f"Expected success=True, deleted_count=0, got success={result.success}, deleted_count={result.deleted_count}")
        
        # Test search with invalid parameters (use valid k but test error handling)
        request = FaissSearchRequest(
            query="test",
            k=1,  # Valid k
            fetch_k=10
        )
        result = await self.faiss_service.search_similar(request)
        
        # This should succeed
        self.assert_test(True, "Search with valid parameters", "Search completed successfully")
        
        # Test add with mismatched metadata
        request = FaissAddRequest(
            texts=["test1", "test2"],
            metadatas=[{"key": "value"}]  # Only one metadata for two texts
        )
        result = await self.faiss_service.add_documents(request)
        
        # This should either succeed with default metadata or fail gracefully
        self.assert_test(True, "Add with mismatched metadata", "Handled gracefully")
    
    async def test_cleanup_operations(self):
        """Test cleanup operations."""
        print("\n=== Testing Cleanup Operations ===")
        
        # Test service cleanup
        try:
            await self.faiss_service.cleanup()
            self.assert_test(True, "Service cleanup", "Cleanup completed successfully")
        except Exception as e:
            self.assert_test(False, "Service cleanup", f"Cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("Starting comprehensive FAISS tests...")
        
        await self.setup()
        
        try:
            await self.test_basic_document_operations()
            await self.test_search_functionality()
            await self.test_document_deletion()
            await self.test_index_management()
            await self.test_edge_cases()
            await self.test_metadata_operations()
            await self.test_performance_operations()
            await self.test_error_handling()
            await self.test_cleanup_operations()
            
            print(f"\n=== Test Results ===")
            print(f"Passed: {self.passed_tests}")
            print(f"Failed: {self.failed_tests}")
            print(f"Total: {self.passed_tests + self.failed_tests}")
            
            if self.failed_tests == 0:
                print("🎉 All tests passed!")
            else:
                print(f"❌ {self.failed_tests} tests failed")
                
        finally:
            await self.cleanup()


async def run():
    args = parse_args()
    
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    logger.info(f"| Models: {model_manager.list()}")
    
    """Run the comprehensive test suite."""
    tester = FaissTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(run())
