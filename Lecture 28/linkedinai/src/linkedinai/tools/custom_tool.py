from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from linkedin_api import Linkedin
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain.agents import Tool
from crewai.tools import tool
import requests
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)
from dotenv import load_dotenv
load_dotenv()



class LinkedInPostTool:

    @tool("Linkedin Post")
    def post(content: str) -> str:
        """
        Tool to share a post on LinkedIn using the LinkedIn API.

        Args:   
            content (str): The content of the post.
        """
        
        url = 'https://api.linkedin.com/v2/ugcPosts'

        post_data = {
            "author": os.environ.get('AUTHOR_URN'), #AUTHOR_URN,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": content
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }

        headers = {
            'Authorization': f'Bearer {os.environ.get("LINKEDIN_ACCESS_TOKEN")}', #LINKEDIN_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }

        response = requests.post(url, headers=headers, json=post_data)

        if response.status_code == 201:
            print("Post shared successfully on LinkedIn!")
        else:
            print("Error:", response.status_code, response.text)

        return response
