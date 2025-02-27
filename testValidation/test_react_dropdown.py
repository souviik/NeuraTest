"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList

# llm = ChatOpenAI(model='gpt-4o')
llm =  ChatGoogleGenerativeAI(
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
# browser = Browser(config=BrowserConfig(headless=False))

agent = Agent(
	task=(
		'go to https://codepen.io/shyam-king/pen/ByBJoOv and select "Tiger" dropdown and read the text given in "Selected Animal" box (it can be empty as well)'
	),
	llm=llm,
	browser_context=BrowserContext(
		browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
	),
)


async def test_dropdown():
	history: AgentHistoryList = await agent.run(10)
	# await controller.browser.close(force=True)

	result = history.final_result()
	assert result is not None
	print('result: ', result)
	# await browser.close()


if __name__ == '__main__':
	asyncio.run(test_dropdown())
