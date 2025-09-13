import os
import sys
import tempfile
import json
from pathlib import Path
import asyncio

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environments.filesystem import FileSystemService, FileSystem
from src.environments.filesystem.types import FileReadRequest
from src.environments.filesystem.handlers import TextHandler, JsonHandler, HandlerRegistry


class FileSystemTester:
    """Comprehensive test suite for filesystem functionality."""
    
    def __init__(self):
        self.test_dir = None
        self.fs = None
        self.service = None
        self.passed_tests = 0
        self.failed_tests = 0
    
    async def setup(self):
        """Setup test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.fs = FileSystem(base_dir=self.test_dir)
        self.service = FileSystemService(self.test_dir)
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
    
    async def test_basic_file_operations(self):
        """Test basic file read/write operations."""
        print("\n=== Testing Basic File Operations ===")
        
        # Test write file
        test_file = str(self.test_dir / "test.txt")
        result = await self.fs.write_file(test_file, "Hello, World!")
        self.assert_test("Successfully overwritten" in result, result)
        
        # Test read file
        result = await self.fs.read_file(test_file)
        self.assert_test("Hello, World!" in result, result)
        
        # Test append mode
        result = await self.fs.write_file(test_file, "\nAppended text", mode="a")
        self.assert_test("Successfully appended" in result, result)
        
        # Test read with line range
        result = await self.fs.read_file(test_file, start_line=1, end_line=2)
        self.assert_test("Hello, World!" in result, result)
        
        # Test file info
        result = await self.fs.get_file_info(test_file)
        self.assert_test("File" in result and "Size:" in result, result)

    
    async def test_directory_operations(self):
        """Test directory operations."""
        print("\n=== Testing Directory Operations ===")
        
        # Test create directory
        test_dir = str(self.test_dir / "test_dir")
        result = await self.fs.create_directory(test_dir)
        self.assert_test("Successfully created directory" in result, result)
        
        # Test create nested directory
        nested_dir = str(self.test_dir / "test_dir" / "nested")
        result = await self.fs.create_directory(nested_dir)
        self.assert_test("Successfully created directory" in result, result)
        
        # Test tree structure
        result = await self.fs.tree_structure(str(self.test_dir), max_depth=2)
        self.assert_test("test_dir" in result, result)
        
        # Test delete directory
        result = await self.fs.delete_directory(test_dir)
        self.assert_test("Successfully deleted directory" in result, result)
        
        # Test describe
        result = await self.fs.describe()
        self.assert_test("File System Overview" in result, result)
    
    async def test_file_manipulation(self):
        """Test file manipulation operations."""
        print("\n=== Testing File Manipulation ===")
        
        # Create test file
        test_file = str(self.test_dir / "manipulation.txt")
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        await self.fs.write_file(test_file, content)
        
        # Test string replacement
        result = await self.fs.replace_file_str(test_file, "Line 2", "Modified Line 2")
        self.assert_test("Successfully replaced 1 occurrences" in result, result)
        
        # Test string replacement with line range
        result = await self.fs.replace_file_str(test_file, "Line", "Item", start_line=3, end_line=4)
        self.assert_test("Successfully replaced 2 occurrences" in result, result)
        
        # Test copy file
        copy_file = str(self.test_dir / "copy.txt")
        result = await self.fs.copy_file(test_file, copy_file)
        self.assert_test("Successfully copied file" in result, result)
        
        # Test move file
        move_file = str(self.test_dir / "moved.txt")
        result = await self.fs.move_file(copy_file, move_file)
        self.assert_test("Successfully moved file" in result, result)
        
        # Test rename file
        rename_file = str(self.test_dir / "renamed.txt")
        result = await self.fs.rename_file(move_file, rename_file)
        self.assert_test("Successfully renamed" in result, result)
        
        # Test delete file
        result = await self.fs.delete_file(rename_file)
        self.assert_test("Successfully deleted file" in result, result)
    
    async def test_search_functionality(self):
        """Test search functionality."""
        print("\n=== Testing Search Functionality ===")
        
        # Create test files
        test_file1 = str(self.test_dir / "search1.txt")
        test_file2 = str(self.test_dir / "search2.txt")
        await self.fs.write_file(test_file1, "This is a test file with search content")
        await self.fs.write_file(test_file2, "Another file with different content")
        
        # Test name search
        result = await self.fs.search_files(str(self.test_dir), "search1", search_type="name")
        self.assert_test("search1.txt" in result, result)
        
        # Test content search
        result = await self.fs.search_files(str(self.test_dir), "test file", search_type="content")
        self.assert_test("search1.txt" in result, result)
        
        # Test single file content search
        result = await self.fs.search_files(test_file1, "search content", search_type="content")
        self.assert_test("Found 1 matches" in result, result)
    
    async def test_permissions(self):
        """Test file permissions."""
        print("\n=== Testing File Permissions ===")
        
        # Create test file
        test_file = str(self.test_dir / "perms.txt")
        await self.fs.write_file(test_file, "Test content")
        
        # Test change permissions
        result = await self.fs.change_permissions(test_file, "644")
        self.assert_test("Successfully changed permissions" in result, result)
    
    async def test_service_layer(self):
        """Test the service layer directly."""
        print("\n=== Testing Service Layer ===")
        
        # Test service write/read
        test_path = Path("service_test.txt")
        await self.service.write_text(test_path, "Service test content")
        
        request = FileReadRequest(path=test_path)
        result = await self.service.read(request)
        self.assert_test(result.content_text == "Service test content", result)
        
        # Test service directory operations
        await self.service.mkdir(Path("service_dir"))
        files = await self.service.listdir(Path("."))
        self.assert_test("service_dir" in files, result)
        
        # Test service file stats
        stat = await self.service.stat(test_path)
        self.assert_test(stat.st_size > 0, result)
    
    async def test_handlers(self):
        """Test content handlers."""
        print("\n=== Testing Content Handlers ===")
        
        # Test TextHandler
        text_handler = TextHandler()
        test_data = b"Hello, Handler World!"
        request = FileReadRequest(path=Path("test.txt"))
        result = await text_handler.decode(test_data, request)
        self.assert_test(result.content_text == "Hello, Handler World!", result)
        
        # Test JsonHandler
        json_handler = JsonHandler()
        json_data = b'{"name": "test", "value": 123}'
        request = FileReadRequest(path=Path("test.json"))
        result = await json_handler.decode(json_data, request)
        self.assert_test("JSON Object with keys" in result.preview, result)
        
        # Test HandlerRegistry
        registry = HandlerRegistry()
        registry.register(text_handler)
        registry.register(json_handler)
        
        handler = registry.find_for_extension('.txt')
        self.assert_test(handler is not None, result)
        
        # Test supported extensions
        extensions = registry.get_supported_extensions()
        self.assert_test('.txt' in extensions and '.json' in extensions, result)
    
    async def test_error_handling(self):
        """Test error handling."""
        print("\n=== Testing Error Handling ===")
        
        # Test file not found
        result = await self.fs.read_file(str(self.test_dir / "nonexistent.txt"))
        self.assert_test("Error:" in result, result)
        
        # Test invalid path
        try:
            result = await self.fs.read_file("/invalid/absolute/path")
            self.assert_test("Error:" in result, result)
        except Exception:
            self.assert_test(True, "Handle invalid path (exception)")
        
        # Test directory not found for tree
        result = await self.fs.tree_structure(str(self.test_dir / "nonexistent"))
        self.assert_test("Error:" in result, result)
        
        # Test invalid search type
        result = await self.fs.search_files(str(self.test_dir), "test", search_type="invalid")
        self.assert_test("Error:" in result, result)
    
    async def test_edge_cases(self):
        """Test edge cases."""
        print("\n=== Testing Edge Cases ===")
        
        # Test empty file
        empty_file = str(self.test_dir / "empty.txt")
        result = await self.fs.write_file(empty_file, "")
        self.assert_test("Successfully overwritten" in result, result)
        
        # Test large content
        large_content = "Line " + "\n".join([f"{i}" for i in range(1000)])
        large_file = str(self.test_dir / "large.txt")
        result = await self.fs.write_file(large_file, large_content)
        self.assert_test("Successfully overwritten" in result, result)
        
        # Test special characters
        special_file = str(self.test_dir / "special.txt")
        special_content = "特殊字符: 中文, émojis: 🚀, symbols: @#$%"
        result = await self.fs.write_file(special_file, special_content)
        self.assert_test("Successfully overwritten" in result, result)
        
        # Test line range edge cases
        result = await self.fs.read_file(large_file, start_line=1, end_line=10)
        self.assert_test("Line 0" in result and "9" in result, result)
    
    async def test_describe_functionality(self):
        """Test describe functionality."""
        print("\n=== Testing Describe Functionality ===")
        
        # Create some test content
        await self.fs.write_file(str(self.test_dir / "desc1.txt"), "Description test 1")
        await self.fs.write_file(str(self.test_dir / "desc2.txt"), "Description test 2")
        await self.fs.create_directory(str(self.test_dir / "desc_dir"))
        
        # Test describe
        result = await self.fs.describe()
        self.assert_test("File System Overview" in result, result)
        self.assert_test("desc1.txt" in result, result)
        self.assert_test("desc_dir" in result, result)
    
    async def run_all_tests(self):
        """Run all tests."""
        print("Starting comprehensive filesystem tests...")
        
        await self.setup()
        
        try:
            await self.test_basic_file_operations()
            await self.test_directory_operations()
            await self.test_file_manipulation()
            await self.test_search_functionality()
            await self.test_permissions()
            await self.test_service_layer()
            await self.test_handlers()
            await self.test_error_handling()
            await self.test_edge_cases()
            await self.test_describe_functionality()
            
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
    """Run the comprehensive test suite."""
    tester = FileSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(run())
