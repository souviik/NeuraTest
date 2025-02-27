"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
from pprint import pprint

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from AIAgentTests.browser.using_cdp import controller
from browser_use.browser.browser import Browser, BrowserConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList, Controller

# llm = ChatOpenAI(model='gpt-4o')
# controller = Controller()

llm = ChatGoogleGenerativeAI(
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
# use this test to ask the model questions about the page like
# which color do you see for bbox labels, list all with their label
# whats the smallest bboxes with labels and


@controller.registry.action(description='explain what you see on the screen and ask user for input')
async def explain_screen(text: str) -> str:
	pprint(text)
	answer = input('\nuser input next question: \n')
	return answer


@controller.registry.action(description='done')
async def done(text: str) -> str:
	# pprint(text)
	return 'call explain_screen'


agent = Agent(
	task='call explain_screen all the time the user asks you questions e.g. about the page like bbox which you see are labels  - your task is to expalin it and get the next question',
	llm=llm,
	controller=controller,
	browser=Browser(config=BrowserConfig(disable_security=True, headless=False)),
)


@pytest.mark.skip(reason='this is for local testing only')
async def test_vision():
	history: AgentHistoryList = await agent.run(20)
