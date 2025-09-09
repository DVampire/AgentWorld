# Search Tools

This directory contains various search engine tools for web search functionality.

## Brave Search

A search engine tool that queries the Brave search API.

### Setup
```shell
1. Get API key from https://brave.com/search/api/
2. Set BRAVE_SEARCH_API_KEY=xxx in `.env`
```

### Usage
```python
from src.tools.default_tools.search.brave_search import BraveSearch

# Basic usage
brave_search = BraveSearch()

# With custom search parameters
brave_search = BraveSearch.from_search_kwargs({"num_results": 5, "country": "us", "lang": "en"})

# Execute search
result = await brave_search._arun("Python programming tutorial", num_results=5, country="us", lang="en")
```

## Baidu Search

A search engine tool that queries Baidu search, particularly useful for Chinese content.

### Setup
```shell
# No API key required for basic usage
# Baidu search uses the baidusearch library
```

### Usage
```python
from src.tools.default_tools.search.baidu_search import BaiduSearch

# Basic usage
baidu_search = BaiduSearch()

# With custom search parameters
baidu_search = BaiduSearch.from_search_kwargs({
    "num_results": 5,
    "country": "cn",
    "lang": "zh"
})

# Execute search
result = await baidu_search._arun("Python编程教程", num_results=5, country="cn", lang="zh")
```

### Supported Parameters
- `query`: Query to search for (required)
- `num_results`: Number of search results to return (default: 5)
- `country`: Country to search in (default: "us")
- `lang`: Language to search in (default: "en")
- `filter_year`: Year to filter results by (default: None)

## Bing Search

A search engine tool that queries Bing search engine using web scraping.

### Setup
```shell
# No API key required
# Uses web scraping to get search results from Bing
```

### Usage
```python
from src.tools.default_tools.search.bing_search import BingSearch

# Basic usage
bing_search = BingSearch()

# With custom search parameters
bing_search = BingSearch.from_search_kwargs({"num_results": 5, "country": "us", "lang": "en"})

# Execute search
result = await bing_search._arun("Python programming tutorial", num_results=5, country="us", lang="en")
```

## DuckDuckGo Search

A search engine tool that queries DuckDuckGo search engine.

### Setup
```shell
# No API key required
# Uses DuckDuckGo's public search API
```

### Usage
```python
from src.tools.default_tools.search.ddgo_search import DuckDuckGoSearch

# Basic usage
duckduckgo_search = DuckDuckGoSearch()

# With custom search parameters
duckduckgo_search = DuckDuckGoSearch.from_search_kwargs({"num_results": 5, "country": "us", "lang": "en"})

# Execute search
result = await duckduckgo_search._arun("Python programming tutorial", num_results=5, country="us", lang="en")
```

## Firecrawl Search

A search engine tool that queries Firecrawl search engine with advanced web scraping capabilities.

### Setup
```shell
1. Get API key from https://firecrawl.dev/
2. Set FIRECRAWL_API_KEY=xxx in `.env`
```

### Usage
```python
from src.tools.default_tools.search.firecrawl_search import FirecrawlSearch

# Basic usage
firecrawl_search = FirecrawlSearch()

# With custom search parameters
firecrawl_search = FirecrawlSearch.from_search_kwargs({"num_results": 5, "filter_year": 2023})

# Execute search
result = await firecrawl_search._arun("Python programming tutorial", num_results=5, filter_year=2023)
```

## Google Search

A search engine tool that queries Google search engine with both API and web scraping support.

### Setup
```shell
# Option 1: Use Google Search API (recommended)
1. Get API key from your Google Search API provider
2. Set SKYWORK_GOOGLE_SEARCH_API=xxx in `.env`

# Option 2: Use web scraping (no API key required)
# No setup required, but may be less reliable
```

### Usage
```python
from src.tools.default_tools.search.google_search import GoogleSearch

# Basic usage
google_search = GoogleSearch()

# With custom search parameters
google_search = GoogleSearch.from_search_kwargs({"num_results": 5, "filter_year": 2023})

# Execute search
result = await google_search._arun("Python programming tutorial", num_results=5, filter_year=2023)
```

## Common Features

All search tools provide:
- **Unified Parameter Interface**: All tools use `SearchToolArgs` for consistent parameter handling
- **Asynchronous and synchronous execution**: Both `_arun()` and `_run()` methods
- **Structured search results**: Standardized `SearchItem` format
- **JSON-formatted output**: Consistent `ToolResponse` format
- **Error handling**: Comprehensive exception handling
- **Configurable search parameters**: Support for `num_results`, `country`, `lang`, `filter_year`

### Unified Parameters

All search tools support the same parameter interface:

```python
class SearchToolArgs(BaseModel):
    query: str = Field(description="The query to search for")
    num_results: int = Field(default=5, description="Number of results to return")
    country: Optional[str] = Field(default="us", description="Country to search in")
    lang: Optional[str] = Field(default="en", description="Language to search in")
    filter_year: Optional[int] = Field(default=None, description="Year to filter results by")
```

## SearchItem Format

All search tools return results in a standardized format:

```python
class SearchItem(BaseModel):
    title: str
    url: str
    description: Optional[str] = None
```

## Tool Selection Guide

Choose the right search tool based on your needs:

| Tool | Best For | API Key Required | Special Features |
|------|----------|------------------|------------------|
| **BraveSearch** | High-quality results, privacy-focused | ✅ | Advanced filtering, multiple result types |
| **BaiduSearch** | Chinese content, Chinese language queries | ❌ | Optimized for Chinese search |
| **BingSearch** | General web search, Microsoft ecosystem | ❌ | Web scraping, no API limits |
| **DuckDuckGoSearch** | Privacy-focused search, no tracking | ❌ | Privacy-first, instant answers |
| **FirecrawlSearch** | Advanced web scraping, content extraction | ✅ | Powerful scraping, content analysis |
| **GoogleSearch** | Comprehensive results, familiar interface | Optional | API + scraping fallback |

### Quick Recommendations

- **For Chinese content**: Use `BaiduSearch`
- **For privacy**: Use `DuckDuckGoSearch` or `BraveSearch`
- **For reliability**: Use `GoogleSearch` (with API key)
- **For advanced scraping**: Use `FirecrawlSearch`
- **For general use**: Use `BingSearch` (no API key needed)