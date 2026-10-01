import asyncio
import os
import sys

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from mcp_client import MCPClient

load_dotenv()

model = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
anthropic = AsyncAnthropic()  # 讀取環境變數 ANTHROPIC_API_KEY


async def chat(client: MCPClient, messages: list[dict]) -> str:
    """送出對話給 Claude;Claude 要求用工具時,透過 MCP client 執行並回傳結果,直到得到最終文字回覆。"""
    tools = [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in await client.list_tools()
    ]

    while True:
        response = await anthropic.messages.create(
            model=model,
            max_tokens=1024,
            messages=messages,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"[tool] {block.name}({block.input})")
            result = await client.call_tool(block.name, block.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "".join(
                        c.text for c in result.content if c.type == "text"
                    ),
                    "is_error": result.isError,
                }
            )
        messages.append({"role": "user", "content": tool_results})


async def main():
    async with MCPClient(command=sys.executable, args=["mcp_server.py"]) as client:
        messages: list[dict] = []
        print("Document assistant. 輸入 exit 離開。")
        while True:
            query = (await asyncio.to_thread(input, "\n> ")).strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            messages.append({"role": "user", "content": query})
            print(await chat(client, messages))


if __name__ == "__main__":
    asyncio.run(main())
