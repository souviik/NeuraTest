"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use import ActionResult, Agent, Controller

load_dotenv()

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


class Post(BaseModel):
    post_title: str
    post_url: str
    num_price: str
    # hours_since_post: int


class Posts(BaseModel):
    posts: List[Post]


controller = Controller(output_model=Posts)


async def main():
    # task = 'Go to hackernews show hn and give me the first 5 posts'
    task = 'Go to "google.com" and search for "iphone 15 pro" show the result and give me the first 5 posts and fetch price is present'
    # model = ChatOpenAI(model='gpt-4o')
    agent = Agent(task=task, llm=model, controller=controller)

    history = await agent.run()

    result = history.final_result()
    if result:
        parsed: Posts = Posts.model_validate_json(result)

        for post in parsed.posts:
            print('\n--------------------------------')
            print(f'Title:            {post.post_title}')
            print(f'URL:              {post.post_url}')
            print(f'Price:         {post.num_price}')
            # print(f'Hours since post: {post.hours_since_post}')
    else:
        print('No result')


if __name__ == '__main__':
    asyncio.run(main())
