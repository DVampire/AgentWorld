"""
LeetCode problem fetcher class for testing.
This is a simple class (not a Tool) that can fetch LeetCode problem information.
"""

import aiohttp
from typing import Optional, Dict, Any, ClassVar
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
# Load .env file, suppress warnings for parsing issues
load_dotenv(verbose=False)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)


from src.logger import logger


class LeetCodeFetcher:
    """A simple class for fetching LeetCode problem information asynchronously."""
    
    # LeetCode GraphQL endpoint
    GRAPHQL_URL: ClassVar[str] = "https://leetcode.com/graphql"
    API_URL: ClassVar[str] = "https://leetcode.com/api/problems/all/"
    
    def __init__(self):
        """Initialize the LeetCode fetcher."""
        pass

    async def get_problem(self, slug: Optional[str] = None, problem_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch LeetCode problem information by slug or problem ID.

        Args:
            slug (str, optional): The problem slug (e.g., 'two-sum'). Defaults to None.
            problem_id (int, optional): The problem ID (e.g., 1). Defaults to None.
        
        Returns:
            Dict[str, Any]: The problem data, or None if not found.
        """
        try:
            if not slug and not problem_id:
                logger.error("Either 'slug' or 'problem_id' must be provided")
                return None
            
            # If only problem_id is provided, we need to get the slug first
            if problem_id and not slug:
                slug = await self._get_slug_by_id(problem_id)
                if not slug:
                    logger.error(f"Problem with ID {problem_id} not found")
                    return None
            
            # Fetch problem details using GraphQL
            problem_data = await self._fetch_problem_details(slug)
            
            if not problem_data:
                logger.error(f"Failed to fetch problem details for slug: {slug}")
                return None
            
            return problem_data
            
        except Exception as e:
            logger.error(f"Error fetching LeetCode problem: {e}")
            return None

    async def get_formatted_problem(self, slug: Optional[str] = None, problem_id: Optional[int] = None) -> Optional[str]:
        """
        Fetch LeetCode problem and return formatted string.

        Args:
            slug (str, optional): The problem slug (e.g., 'two-sum'). Defaults to None.
            problem_id (int, optional): The problem ID (e.g., 1). Defaults to None.
        
        Returns:
            str: Formatted problem information, or None if not found.
        """
        problem_data = await self.get_problem(slug=slug, problem_id=problem_id)
        if not problem_data:
            return None
        
        return self._format_problem_response(problem_data)

    async def _get_slug_by_id(self, problem_id: int) -> Optional[str]:
        """Get problem slug by problem ID."""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://leetcode.com/problemset/',
                    'Origin': 'https://leetcode.com',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin'
                }
                async with session.get(self.API_URL, headers=headers) as response:
                    if response.status == 200:
                        # LeetCode sometimes returns JSON with text/html content-type
                        # Try to parse as JSON regardless of content-type
                        try:
                            data = await response.json()
                        except Exception as json_error:
                            # If JSON parsing fails, read as text and try to parse manually
                            text = await response.text()
                            content_type = response.headers.get('Content-Type', '')
                            logger.warning(f"Failed to parse JSON (Content-Type: {content_type}). Response preview: {text[:200]}")
                            # Try to parse text as JSON if it looks like JSON
                            if text.strip().startswith('{'):
                                try:
                                    import json
                                    data = json.loads(text)
                                except:
                                    return None
                            else:
                                return None
                        
                        questions = data.get('stat_status_pairs', [])
                        for question in questions:
                            if question['stat']['question_id'] == problem_id:
                                return question['stat']['question__title_slug']
                    else:
                        logger.warning(f"API returned status {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error getting slug by ID: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting slug by ID: {e}")
            return None

    async def _fetch_problem_details(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch problem details using LeetCode GraphQL API."""
        query = """
        query getQuestionDetail($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                questionFrontendId
                title
                titleSlug
                content
                difficulty
                likes
                dislikes
                isLiked
                similarQuestions
                contributors {
                    username
                    profileUrl
                    avatarUrl
                }
                topicTags {
                    name
                    slug
                    translatedName
                }
                codeSnippets {
                    lang
                    langSlug
                    code
                }
                stats
                hints
                solution {
                    id
                    canSeeDetail
                    paidOnly
                    hasVideoSolution
                    paidOnlyVideo
                }
                status
                sampleTestCase
                exampleTestcases
                metaData
                judgerAvailable
                judgeType
                mysqlSchemas
                enableRunCode
                enableTestMode
                enableDebugger
                envInfo
                libraryUrl
                adminUrl
                challengeQuestion {
                    id
                    date
                    incompleteChallengeCount
                    streakCount
                    type
                }
                note
            }
        }
        """
        
        variables = {"titleSlug": slug}
        
        payload = {
            "operationName": "getQuestionDetail",
            "variables": variables,
            "query": query
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://leetcode.com/problems/two-sum/',
                    'Origin': 'https://leetcode.com',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin'
                }
                async with session.post(self.GRAPHQL_URL, json=payload, headers=headers) as response:
                    if response.status == 200:
                        # Try to parse as JSON regardless of content-type
                        try:
                            data = await response.json()
                        except Exception as json_error:
                            # If JSON parsing fails, read as text and try to parse manually
                            text = await response.text()
                            content_type = response.headers.get('Content-Type', '')
                            logger.warning(f"Failed to parse JSON (Content-Type: {content_type}). Response preview: {text[:200]}")
                            # Try to parse text as JSON if it looks like JSON
                            if text.strip().startswith('{'):
                                try:
                                    import json
                                    data = json.loads(text)
                                except:
                                    return None
                            else:
                                return None
                        
                        if 'data' in data and 'question' in data['data']:
                            return data['data']['question']
                        elif 'errors' in data:
                            logger.warning(f"GraphQL errors: {data['errors']}")
                    else:
                        logger.warning(f"Failed to fetch problem details. Status: {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching problem details: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching problem details: {e}")
            return None

    def _format_problem_response(self, problem_data: Dict[str, Any]) -> str:
        """Format problem data into a readable string."""
        lines = []
        
        # Basic information
        lines.append(f"# {problem_data.get('title', 'Unknown')}")
        lines.append(f"**Problem ID:** {problem_data.get('questionFrontendId', 'N/A')}")
        lines.append(f"**Difficulty:** {problem_data.get('difficulty', 'Unknown')}")
        lines.append("")
        
        # Content
        content = problem_data.get('content', '')
        if content:
            lines.append("## Problem Description")
            lines.append(content)
            lines.append("")
        
        # Tags
        tags = problem_data.get('topicTags', [])
        if tags:
            tag_names = [tag.get('name', '') for tag in tags]
            lines.append(f"**Tags:** {', '.join(tag_names)}")
            lines.append("")
        
        # Code snippets
        code_snippets = problem_data.get('codeSnippets', [])
        if code_snippets:
            lines.append("## Code Snippets")
            for snippet in code_snippets:
                lang = snippet.get('lang', 'Unknown')
                code = snippet.get('code', '')
                lines.append(f"### {lang}")
                lines.append(f"```{snippet.get('langSlug', '')}")
                lines.append(code)
                lines.append("```")
                lines.append("")
        
        # Hints
        hints = problem_data.get('hints', [])
        if hints:
            lines.append("## Hints")
            for i, hint in enumerate(hints, 1):
                lines.append(f"{i}. {hint}")
            lines.append("")
        
        # Stats
        stats = problem_data.get('stats', '')
        if stats:
            try:
                stats_data = json.loads(stats)
                lines.append("## Statistics")
                lines.append(f"- **Total Accepted:** {stats_data.get('totalAccepted', 'N/A')}")
                lines.append(f"- **Total Submissions:** {stats_data.get('totalSubmission', 'N/A')}")
                lines.append(f"- **Acceptance Rate:** {stats_data.get('acRate', 'N/A')}")
                lines.append("")
            except:
                pass
        
        return "\n".join(lines)


async def main():
    """Example usage of LeetCodeFetcher."""
    fetcher = LeetCodeFetcher()
    
    # Example 1: Fetch by slug
    print("Fetching problem by slug: 'two-sum'")
    problem = await fetcher.get_problem(slug="two-sum")
    if problem:
        print(f"Title: {problem.get('title')}")
        print(f"Difficulty: {problem.get('difficulty')}")
    
    # Example 2: Fetch by ID
    print("\nFetching problem by ID: 1")
    formatted = await fetcher.get_formatted_problem(problem_id=1)
    if formatted:
        print(formatted)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

