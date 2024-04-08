'''
created by jake on march 30 2024. 
first attempt at creating and coordinating a team of ai agents
objective: create a useful and insightful analysis of a publicly traded company, saving me a full day of work (or more ideally)
'''
import time
start = time.time()
import os
from dotenv import load_dotenv
# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
)

from crewai import Agent, Task, Process, Crew

print('CrewAI imported successfully')
from langchain_community.tools import DuckDuckGoSearchRun
print('DuckDuckGoSearchRun imported successfully')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print('Environment variables loaded successfully')

print(f"loading and importing all modules took {time.time() - start} seconds")

search_tool = WebsiteSearchTool()
scrape_tool = ScrapeWebsiteTool()
# Create the tools

# HERE BELONGS THE DEFINITION OF THE TOOLS

search_tool = DuckDuckGoSearchRun()

# Create the agents
stock_finder = Agent(
    role='Stock Finder',
    goal='Identify one company that is publicly traded and has a market capitalization of less than $200 million in a promising industry',
    backstory='''Expert in industry selection, with a deep understanding of market trends and potential growth areas.''',
    verbose = True,
    allow_delegation=False,
    tools=[search_tool]
)
# Create the agents

quant_overview_researcher = Agent(
    role='Data Collector for Company Overview',
    goal='Gather comprehensive data on the company ',
    backstory='''Expert investor with a background in data analysis and financial research, skilled in extracting important information''',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool]

)

qual_overview_researcher = Agent(
    role='Data Collector for Company Overview',
    goal='Gather comprehensive background information on the company including history and business model',
    backstory='''Expert investor with a background in data analysis and financial research, skilled in extracting important information''',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool]
)

total_overview_agent = Agent(
    role='Lead Investor, director of Investment Pitch Creation',
    goal='create a comprehensive overview of the company for the beginning of the investment pitch',
    backstory='''Experienced investor with a background in financial analysis and investment strategy, skilled in creating grounded investment pitches''',
    verbose=True,
    allow_delegation=True,
)


#   Define the tasks
task_quant_research = Task(
    description="""Collect quantitative data on the company including financials, stock price, and market cap using the otc website and other documents necessary """, 
    agent=quant_overview_researcher,
    verbose=True,

    expected_output=""" a dictionary containing the following keys and their respective values:{
        "company_name": "",
        "ticker_symbol": "",
        "share_price": "",
        "Fully-diluted Shares Outstanding": "",
        "market_cap": "",
        "net_cash": "",
        
    }"""

    )
task_qual_research = Task(
    description="""Collect qualitative data on the company including history, business model, and competitive landscape using the otc website and other documents necessary. be thorough using a range of recent news and company reports""",
    agent=qual_overview_researcher,
    expected_output = """
        "a few paragraphs outlining the company's history, business model, and competitive landscape, including specific statistics descriptions and data"
    """

)

task_total_overview = Task(
    description="""Create a comprehensive overview of the company including both quantitative and qualitative data for the beginning of the investment pitch""",
    agent=total_overview_agent,
    expected_output = """a leading paragraph structurefd as follows:
    Name and ticker symbol
    Current share price and currency
    Fully-diluted shares outstanding
    Market capitalization
    Net cash position
followed by an investment summary for the company based on the provided data:

- Company Overview: Brief description of the company, its industry, and primary activities.
- Financial Performance: Key financial metrics, including revenue growth, EBIT margin, market capitalization, and net cash position.
- Market Position: Overview of the company’s market position, competitive advantages, and main customers.
- Strategic Initiatives: Description of recent strategic initiatives, such as new product launches or expansion plans.
- Market Trends and Compliance: Discussion of market trends affecting the industry and the company’s compliance with relevant regulations.
- Business Segments: Analysis of the company’s business segments and their contributions to growth and profitability.
- Future Outlook: Insights into the company’s future growth prospects, potential market opportunities, and financial forecasts.

Use this data to create a comprehensive summary that highlights the investment appeal of [Company Name].
""")
task_pick_company = Task(
    description="""Decide on ONE publicly traded company to analyze that has a market capitalization of less than $200 million. Choose this based on size of company and industry(promising&fast growing). find it from otcmarkets.com""",
    agent=stock_finder,
    expected_output="the name and ticker of one company that fits the criteria of being in a promising industry and having a market cap under $200 million."
)
# Instantiate the Crew
overview_crew = Crew(
    agents=[stock_finder, total_overview_agent,  quant_overview_researcher, qual_overview_researcher],
    tasks=[task_pick_company, task_total_overview, task_qual_research, task_quant_research],
    verbose=2,
    max_rpm=50,
    process=Process.sequential
)

# Execute the process
start = time.time()
result = overview_crew.kickoff()
print("######################")
print(result)
print(f"the crewai process took {time.time() - start} seconds to complete")


# financial_analyst = Agent(
#     role='Financial Analyst',
#     goal=f'Analyze financial statements, compute key financial ratios, and assess the financial health of the company',
#     backstory="""Specializes in financial analysis, with capabilities in extracting and interpreting financial data 
#     from SEC filings, earnings reports, and financial databases.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool, financial_analysis_tool]
# )

# business_segments_analyst = Agent(
#     role='Business Segments Analyst',
#     goal=f'Examine the details of the company’s business segments, operational efficiency, and product/service lines',
#     backstory="""Expert in dissecting company operations and segment performance, adept at gathering detailed segment 
#     information from annual reports and sector-specific studies.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# strategic_init_researcher = Agent(
#     role='Strategic Initiatives Researcher',
#     goal=f'Identify and assess the company’s strategic plans, growth initiatives, and market catalysts',
#     backstory="""Skilled in uncovering strategic initiatives from press releases, interviews with executives, and 
#     analysis of industry trends to determine growth catalysts.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# competetive_analyst = Agent(
#     role='Competitive Analyst',
#     goal=f'Research and analyze the competitive landscape and industry position of the company',
#     backstory="""Has expertise in competitive analysis and industry positioning, utilizing competitive intelligence 
#     databases and industry analysis reports to evaluate the company’s market standing.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# risk_analyst = Agent(
#     role='Risk Analyst',
#     goal=f'Identify potential risks and the company’s strategies for mitigation',
#     backstory="""Focused on risk assessment and mitigation strategies, adept at analyzing risk factors from company 
#     filings and industry risk reports.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[search_tool]
# )

# valuation_analyst = Agent(
#     role='Valuation Analyst',
#     goal=f'Conduct a valuation analysis and synthesize the investment thesis',
#     backstory="""Specializes in financial modeling and valuation, using financial data to build DCF models and 
#     articulate a compelling investment thesis.""",
#     verbose=True,
#     allow_delegation=False,
# )