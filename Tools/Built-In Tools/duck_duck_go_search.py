from langchain_community.tools import DuckDuckGoSearchRun
from ddgs import DDGS

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('Top news in India today')

print("Results : ", results )