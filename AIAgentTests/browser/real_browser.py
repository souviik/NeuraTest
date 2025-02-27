import os
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use.agent.views import ActionResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

browser = Browser(
	config=BrowserConfig(
		# NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
		chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		headless=False,
		disable_security=True
	)
)
model = ChatGoogleGenerativeAI(
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

async def main():
	agent = Agent(
		task='Open chrome and open a new tab'
			 'then open docs.google.com. Close any overlay if present by clicking Got It button'
			 'then write a small letter to Principal, give title of the document as "Sir", and the select Document Tabs - tab 1'
			 'then write in the body of the letter write a Thank you note to the principal. Make the note detailed then save the document',
		# llm=ChatOpenAI(model='gpt-4o'),
		llm=model,
		browser=browser
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())
