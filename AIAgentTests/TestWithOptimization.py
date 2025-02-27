import logging
import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from browser_use import Agent, Browser, BrowserConfig
from pydantic import SecretStr

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = BrowserConfig(
    headless=False,
    disable_security=False
)

def create_llm():
    if os.getenv("USE_OLLAMA"):
        return ChatOllama(
            model="qwen2.5:latest",
            num_ctx=32000,
            temperature=0
        )
    return ChatGoogleGenerativeAI(
        api_key=SecretStr(os.getenv("GEMINI_API_KEY")),
        model="gemini-1.5-flash",
        # model="gemini-2.0-flash",
        use_vision=True,
        temperature=0,
        max_tokens=None,
        timeout=30000,
        max_retries=2,
        save_conversation_path="logs/conversation"

    )

TASK_PROMPT = """
Go to "google.com" and search for "iphone 16" show the result and give me the first 5 product names and fetch price
Then print the price and the product name in console
Then close the browser
"""

async def main():
    try:
        browser = Browser(config=DEFAULT_CONFIG)
        agent = Agent(
            browser=browser,
            task=TASK_PROMPT,
            llm=create_llm()
        )

        result = await agent.run()
        logger.info(f"Automation result: {result}")
        return result

    except Exception as error:
        logger.error(f"Automation failed: {str(error)}")
        # await browser.capture_screenshot("error.png")
        raise

if __name__ == "__main__":
    asyncio.run(main())