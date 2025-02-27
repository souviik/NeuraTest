from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from playwright.async_api import Browser
from pydantic import SecretStr
from browser_use import Agent, Browser, BrowserConfig
import asyncio
import os
from dotenv import load_dotenv
from browser_use import BrowserConfig


load_dotenv ()
os.getenv('ANONYMIZED_TELEMETRY')

prompt = """"
Go to "http://www.google.com/" and search for "amazon.in"
then open the first url to open Amazon then verify that Amazon.in is indeed open
Then search for the product "iphone 16" and click the search icon in the search box on Amazon
Then get the minimum price of the product from the search result list and give me url of the product
Then close the browser
"""

config = BrowserConfig(
    headless=False,
    disable_security=True
)

llm1 = ChatGoogleGenerativeAI(
    api_key=SecretStr(os.getenv('GEMINI_API_KEY')),
    # model="gemini-2.0-flash",
    use_vision=True,
    save_conversation_path="logs/conversation",
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

llm=ChatOllama(
    model="qwen2.5:latest",
    num_ctx=32000,
)

browser = Browser(config=config)
async def main():
    agent = Agent (
        browser=browser,
        task=prompt,
        llm = llm1
    )
    result = await agent.run()
    print (result)

asyncio.run(main())