from .serpapi import Searcher as SerpAPISearcher
from .bing_search import Searcher as BingSearcher
from .searcher import SearchResult, SearcherInterface
name = "bing"

def create_searcher(name: str) -> SearcherInterface:
    if name == "serpapi":
        return SerpAPISearcher()
    elif name == "bing":
        return BingSearcher()
    else:
        raise NotImplementedError()