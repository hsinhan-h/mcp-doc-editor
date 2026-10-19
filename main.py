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


async def ask(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def run_prompt(client: MCPClient, name: str) -> str | None:
    """依 MCP prompt 的參數互動詢問使用者,回傳展開後的提示文字;取消則回傳 None。"""
    prompts = {p.name: p for p in await client.list_prompts()}
    prompt = prompts.get(name)
    if prompt is None:
        print(f"找不到提示 /{name}。輸入 / 查看可用提示。")
        return None

    args: dict[str, str] = {}
    for arg in prompt.arguments or []:
        if arg.name == "doc_id":
            doc_ids = await client.read_resource("docs://documents")
            for i, doc_id in enumerate(doc_ids, 1):
                print(f"  {i}. {doc_id}")
            choice = await ask("選擇文件編號: ")
            if not choice.isdigit() or not 1 <= int(choice) <= len(doc_ids):
                print("已取消。")
                return None
            args[arg.name] = doc_ids[int(choice) - 1]
        else:
            args[arg.name] = await ask(f"{arg.name}: ")

    messages = await client.get_prompt(name, args)
    return "\n".join(m.content.text for m in messages if m.content.type == "text")


async def main():
    async with MCPClient(command=sys.executable, args=["mcp_server.py"]) as client:
        messages: list[dict] = []
        print("Document assistant. 輸入 / 查看提示,輸入 exit 離開。")
        while True:
            query = await ask("\n> ")
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            if query.startswith("/"):
                name = query[1:].strip()
                if not name:
                    for p in await client.list_prompts():
                        print(f"  /{p.name} - {p.description}")
                    continue
                query = await run_prompt(client, name)
                if query is None:
                    continue
            messages.append({"role": "user", "content": query})
            print(await chat(client, messages))


if __name__ == "__main__":
    asyncio.run(main())
