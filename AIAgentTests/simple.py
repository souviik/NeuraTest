import asyncio
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

load_dotenv()

# llm = ChatOpenAI(
# 	model='gpt-4o',
# 	temperature=0.0,
# )

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

task = 'Open google.com and search for the founders of Artifical Intelligence and scroll to the bottom of the page. Give me the name of the founders"'

agent = Agent(task=task, llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
