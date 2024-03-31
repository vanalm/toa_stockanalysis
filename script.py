'''
created by jake on march 30 2024. 
first attempt at creating and coordinating a team of ai agents
objective: create a useful and insightful analysis of a publicly traded company, saving me a full day of work (or more ideally)
'''
import time
start = time.time()
import os
from dotenv import load_dotenv
# from crewai_tools import SerperDevTool
# search_tool = SerperDevTool()
# search_tool = SerperDevTool()
from crewai import Agent, Task, Process, Crew

print('CrewAI imported successfully')
from langchain_community.tools import DuckDuckGoSearchRun
print('DuckDuckGoSearchRun imported successfully')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print('Environment variables loaded successfully')

print(f"loading and importing all modules took {time.time() - start} seconds")

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
# company_name = 'Tesla'

data_gatherer = Agent(
    role='Financial Data Researcher',
    goal=f'Gather comprehensive financial data, recent news, and SEC filings for the company in question',
    backstory="""Expert in financial data mining, with skills in identifying and retrieving critical information 
    from various financial databases and news outlets to provide a solid foundation for investment analysis.""",
    verbose=True,
    allow_delegation=False,
    # Assuming search_tool is already defined to use for web scraping or API calls
    tools=[search_tool]
)

analyst = Agent(
    role='Financial Analyst',
    goal='Analyze the financial data and market information to assess the performance and potential of the penny stocks',
    backstory="""Specializes in financial analysis and market research, adept at interpreting complex datasets to 
    extract actionable insights and identify trends that could impact investment decisions.""",
    verbose=True,
    allow_delegation=True
    # Tools for data analysis would be integrated here if needed
)

evaluator = Agent(
    role='Investment Evaluator',
    goal='Assess the investment potential of each penny stock based on the analysis, highlighting risks and opportunities',
    backstory="""Expert in investment strategy and risk assessment, with the ability to synthesize financial analyses 
    into strategic investment advice, focusing on identifying high-potential penny stocks for investment.""",
    verbose=True,
    allow_delegation=True
    # Tools for evaluation and decision-making could be included
)

#  Define the tasks
task_pick_company = Task(
    description="""Decide on ONE publicly traded company to analyze that has a market capitalization of less than $200 million. Choose this based on size of company and industry(promising&fast growing).""",
    agent=stock_finder,
    expected_output="the name and ticker of one company that fits the criteria of being in a promising industry and having a market cap under $200 million."
)
task_retrieve_data = Task(
    description="""Collect comprehensive financial data, latest news articles, and SEC filings for the targeted company. 
    Focus on extracting key financial metrics, market trends, and significant regulatory events that could impact the stocks' value.""",
    agent=data_gatherer,
    expected_output="several dense paragraphs listing FACTS qualitative and quantitative about the company, its financials, and the industry it operates in."
)
# ) in the future we could have separte agents for each source of info (ie SEC, Reddit, Yahoo, etc)
# For the quantitative and qualitative traits used to judge a company's potential, here are specific places where you can find information:

# Quantitative Traits
# Revenue Growth: Financial statements on investor relations pages of the company's website or financial databases like Yahoo Finance or Morningstar.
# Profit Margins: Company's quarterly or annual financial reports, Bloomberg, or Reuters.
# Return on Equity (ROE): Financial analysis sections on sites like Investopedia, or financial data providers like Zacks Investment Research.
# Debt-to-Equity Ratio: Financial metrics available on market research platforms such as MarketWatch or CNBC.
# Earnings Per Share (EPS): Detailed in a company’s earnings report, or platforms like Seeking Alpha for analysis.
# Market Share: Market research reports from firms like Gartner, IDC, or Statista.
# Cash Flow: Cash flow statements in the company’s annual report or on financial information websites like the SEC’s EDGAR database.
# Qualitative Traits
# Management Team: Company’s official website, particularly the ‘About Us’ section, or professional networking sites like LinkedIn.
# Brand Strength: Market research reports from firms like Nielsen or Brand Finance, and news articles analyzing brand value.
# Competitive Advantage: Business analysis pieces in media like Harvard Business Review or The Economist.
# Customer Loyalty: Consumer feedback and reviews on social media platforms, or customer satisfaction survey results published by firms like J.D. Power.
# Industry Position and Trends: Industry reports from consultancies like McKinsey & Company or Bain & Company, and articles from industry-specific news outlets.
# Corporate Governance: Governance reports from agencies like Glass Lewis or the Corporate Library, and the company’s annual proxy statement.
# Innovation Capability: Articles and case studies in technology and business publications like TechCrunch, MIT Technology Review, or Forbes.
# Regulatory Environment: Regulatory filings on the SEC’s EDGAR database, or reports from regulatory bodies like the FDA or FCC.

task_analyze_data = Task(
    description="""Parse the provided company data into tabular form, separating qualitative and quantitative data.""",
    agent=analyst,
    expected_output="a tabular presentation of financial data, market trends, and regulatory events, with a focus on key metrics and insights that could impact the stock's value. provided this in a format that a datascientist could use to build a model."
)

task_evaluate_stocks = Task(
    description="""Review the analysis provided by the Analyst to determine the investment potential of each penny stock. 
    Assess the risks and opportunities, market conditions, and projected growth to make informed investment recommendations.""",
    agent=evaluator,
    expected_output="a detailed investment report outlining the potential risks and opportunities of the company in question. provide a recommendation on whether to invest in the stock or not, with supporting evidence and analysis."
)

# Instantiate the Crew
crew = Crew(
    agents=[stock_finder, data_gatherer, analyst, evaluator],
    tasks=[task_pick_company, task_retrieve_data, task_analyze_data, task_evaluate_stocks],
    verbose=2,
    process=Process.sequential
)

# Execute the process
start = time.time()
result = crew.kickoff()
print("######################")
print(result)
print(f"the crewai process took {time.time() - start} seconds to complete")