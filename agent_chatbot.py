#conda env langgraph_env ,Python版本 3.13.5
#如果要使用fastapi server,先把user_api.py跑起来：python user_api.py
#如果要使用自定义prompt，查看prompt.md文件
import os
import json
from typing import TypedDict, Annotated, Sequence
import operator
from datetime import datetime
import requests
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from pprint import pprint
from langchain.tools import tool
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# 【新增】定义调用 FastAPI 服务的工具
API_BASE_URL = "http://127.0.0.1:8000" # 定义 API 的基础 URL

@tool
def get_user_info(user_id: str) -> str:
    """从公司内部系统中查询指定ID的用户信息。

    这个工具用于访问内部用户数据库，以获取特定用户的详细资料，
    如用户名、电子邮件地址和会员等级。

    Args:
        user_id (str): 需要查询的用户的唯一标识符，例如 "user_101"。

    Returns:
        str: 一个描述用户信息的字符串。如果查询成功，会包含用户名、邮箱和会员等级。
             如果用户ID不存在或发生其他错误，会返回一条明确的错误信息。
    """
    print(f"--- [Tool] Calling User API with user_id: {user_id} ---")
    try:
        response = requests.get(f"{API_BASE_URL}/users/{user_id}")
        
        # 检查 HTTP 响应状态
        if response.status_code == 200:
            user_data = response.json()
            return f"用户信息查询成功：用户名 {user_data['username']}, 邮箱 {user_data['email']}, 会员等级 {user_data['membership_level']}。"
        elif response.status_code == 404:
            return f"查询失败：未找到ID为 '{user_id}' 的用户。"
        else:
            # 对于其他可能的 HTTP 错误
            return f"API 请求失败，状态码: {response.status_code}, 详情: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "API 连接失败。请确保 FastAPI 服务正在运行中。"
    except Exception as e:
        return f"调用 API 时发生未知错误: {e}"

@tool
def get_today() -> str:
    """获取当前系统的日期。

    这个函数不接收任何参数，它会返回今天的日期。

    Returns:
        str: 一个表示今天日期的字符串，格式为 'YYYY-MM-DD'。
    """
    # 建议使用 ISO 8601 (YYYY-MM-DD) 格式，因为它是一种国际标准，
    # 对机器（包括LLM）和人类都非常友好和明确。
    return datetime.today().strftime('%Y-%m-%d')

@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气情况。

    这个工具可以查询全球任何一个主要城市的当前天气信息。
    你只需要提供城市名称即可。

    Args:
        city (str): 需要查询天气的城市名称，例如 "Beijing" 或 "上海"。

    Returns:
        str: 一个描述该城市当前天气状况的字符串，包括天气现象、温度、体感温度和湿度。
             如果查询失败，会返回一条错误信息。
    """
    # 注意：这个 API 实际上并不需要 `date` 参数来获取实时天气，
    # 为了让工具更简洁、LLM 更易于使用，我们将其移除。
    try:
        # 添加 format=j1 参数可以获取 JSON 格式的简洁数据，对 LLM 更友好
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()  # 如果请求失败 (如 404)，会抛出 HTTPError 异常
        weather_data = response.json()
        
        # 从JSON中提取关键信息并格式化为对LLM友好的字符串
        current_condition = weather_data.get('current_condition', [{}])[0]
        temp_c = current_condition.get('temp_C')
        feels_like_c = current_condition.get('FeelsLikeC')
        weather_desc_list = current_condition.get('weatherDesc', [{}])
        weather_desc = weather_desc_list[0].get('value') if weather_desc_list else "未知"
        humidity = current_condition.get('humidity')

        # 检查是否获取到了关键数据
        if all([temp_c, feels_like_c, weather_desc, humidity]):
            return (
                f"{city} 当前天气：{weather_desc}，气温 {temp_c}°C，"
                f"体感温度 {feels_like_c}°C，湿度 {humidity}%。"
            )
        else:
            return f"未能获取 {city} 的完整天气数据，请稍后重试。"

    except requests.exceptions.RequestException as e:
        return f"获取天气时网络连接失败：{e}"
    except Exception as e:
        return f"处理 {city} 的天气数据时发生未知错误: {e}"


@tool
def get_historical_events_on_date(month: int, day: int) -> str:
    """
    查询在指定月份和日期，历史上发生了哪些重大事件。

    这个工具需要你提供月份和日期两个数字作为参数。
    例如，要查询5月24日的历史事件，你应该调用此工具并传入 month=5, day=24。

    Args:
        month (int): 月份，一个从 1 到 12 的数字。
        day (int): 日期，一个从 1 到 31 的数字。

    Returns:
        str: 一个描述当天历史事件的字符串列表，如果查询失败则返回错误信息。
    """
    # Numbers API 是一个很棒的免费 API，用于获取关于数字和日期的趣闻
    api_url = f"http://numbersapi.com/{month}/{day}/date"
    
    try:
        response = requests.get(api_url, params={"json": True}) # 请求 JSON 格式
        response.raise_for_status() # 检查 HTTP 错误
        
        event_data = response.json()
        
        # API 返回的格式是: {"text": "...", "number": ..., "found": ..., "type": "date"}
        if event_data.get("found"):
            # 返回找到的历史事件描述
            return f"在 {month}月{day}日，历史上发生的一件大事是：{event_data['text']}"
        else:
            return f"未找到关于 {month}月{day}日 的历史事件记录。"

    except requests.exceptions.RequestException as e:
        return f"查询历史事件时网络连接失败：{e}"
    except Exception as e:
        return f"处理历史事件数据时发生未知错误: {e}"

# -------------------- 1. 定义状态 --------------------
class AgentState(TypedDict):
    """定义 Agent 在图中的状态，所有节点共享和修改此状态。"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# -------------------- 2. 定义 Agent 类 --------------------
class ReActAgent:
    """
    一个基于 LangGraph 实现的、具备工具调用能力的 ReAct 风格 Agent。
    """
    def __init__(self, model: BaseChatModel, tools: list):
        """
        初始化 Agent。
        - model: 一个绑定了工具的 LangChain ChatModel 实例。
        - tools: 一个包含 LangChain 工具实例的列表。
        """
        self.model = model
        self.tools = {t.name: t for t in tools} # 将工具列表转换为字典，方便按名称查找
        self.graph = self._build_graph()
        self.conversation_history = [] # 新增一个列表来存储历史

    def _build_graph(self) -> StateGraph:
        """构建并编译 LangGraph 图。"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("agent", self._call_model)
        workflow.add_node("action", self._call_tool)

        # 设置入口点
        workflow.set_entry_point("agent")

        # 添加条件边
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )

        # 添加普通边
        workflow.add_edge("action", "agent")

        # 编译图
        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """
        私有方法：调用大模型。
        这是图中的 "agent" 节点。
        """
        messages = state['messages']
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def _call_tool(self, state: AgentState) -> dict:
        """
        私有方法：调用工具。
        这是图中的 "action" 节点。
        """
        last_message = state['messages'][-1]
        
        if not last_message.tool_calls:
            return {}

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name in self.tools:
                tool_to_call = self.tools[tool_name]
                try:
                    # 调用工具并获取输出
                    tool_output = tool_to_call.invoke(tool_call['args'])
                    # 将结构化输出序列化为字符串
                    tool_output_str = json.dumps(tool_output, ensure_ascii=False)
                    
                    tool_messages.append(
                        ToolMessage(
                            content=tool_output_str,
                            tool_call_id=tool_call['id'],
                        )
                    )
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e}"
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call['id'])
                    )
            else:
                # 如果模型尝试调用一个不存在的工具
                error_msg = f"Tool '{tool_name}' not found."
                tool_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call['id'])
                )
        
        return {"messages": tool_messages}

    def _should_continue(self, state: AgentState) -> str:
        """
        私有方法：决策下一步走向。
        这是图中的条件边逻辑。
        """
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def run(self, query: str, stream: bool = True) -> str:
        """
        运行 Agent 处理单个查询。
        - query: 用户的输入问题。
        - stream: 是否流式打印中间步骤 (默认为 True)。
        返回 Agent 的最终回答。
        """
        current_messages = self.conversation_history + [HumanMessage(content=query)]
        inputs = {"messages": current_messages}
        
        final_answer = "" # 初始化一个变量来存储最终答案

        if stream:
            print(f"--- Running query: {query} ---\n")
            
            # --- 只运行一次图 ---
            for output in self.graph.stream(inputs):
                # 1. 打印日志（保持不变）
                print("--- Node Output ---")
                pprint(output)
                print("\n")

                # 2. 智能捕获最终答案
                # 最终答案的特征是：它来自于'agent'节点，并且不包含工具调用
                for key, value in output.items():
                    if key == 'agent':
                        # 检查 'agent' 节点的输出中是否有消息
                        messages = value.get('messages', [])
                        if messages:
                            last_message = messages[-1]
                            # 如果最后一条消息不是工具调用，那么它就是最终答案
                            if not last_message.tool_calls:
                                final_answer = last_message.content
            if final_answer:
                # 1. 获取当前用户的提问
                user_message = HumanMessage(content=query)
                # 2. 将最终答案包装成 AIMessage
                agent_message = AIMessage(content=final_answer)
                # 3. 将这一对 Q&A 追加到长期历史中
                self.conversation_history.extend([user_message, agent_message])
            return final_answer
    
        else:
            # 非流式模式保持不变，因为它只运行一次 invoke
            final_state = self.graph.invoke(inputs)
            user_message = HumanMessage(content=query)
            agent_message = AIMessage(content=final_state['messages'][-1].content)
            self.conversation_history.extend([user_message, agent_message])
            return final_state['messages'][-1].content

    def chat(self):
        """启动一个交互式命令行聊天会话。"""
        print("你好！我是 ReAct Agent。输入 'quit' 或 'exit' 退出。")
        while True:
            user_input = input("你: ")
            if user_input.lower() in ["quit", "exit"]:
                print("ReAct Agent: 再见！")
                break
            
            try:
                response = self.run(user_input)
                print(f"ReAct Agent: {response}")
            except Exception as e:
                print(f"ReAct Agent: 抱歉，我遇到了一些麻烦：{e}")
            
            print("-" * 30)


# -------------------- 3. 主程序入口 --------------------
if __name__ == "__main__":

    # a. 初始化工具
    my_search_tool = TavilySearch(name="internet_search_engine", # 自定义 name
                                description="Use this tool to search the internet for up-to-date information.", # 自定义描述
                                max_results=2)
    tools_list = [
        my_search_tool, 
        get_today,          # @tool 装饰器让函数本身就可以被当作工具实例使用
        get_weather,
        get_historical_events_on_date,
        get_user_info
    ]

    # b. 初始化模型并绑定工具
    # llm = ChatOpenAI(temperature=0, model="gpt-4o") # 使用 gpt-4o 或 gpt-3.5-turbo 等
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0, streaming=True).bind_tools(tools_list)

    # c. 创建 Agent 实例
    my_agent = ReActAgent(model=llm, tools=tools_list)

    # --- 使用方式一：运行单个查询 ---
    # print("--- 单次查询示例 ---")
    # question = "2024年欧洲杯的冠军是哪支球队？"
    # answer = my_agent.run(question, stream=True) # 设置 stream=True 查看中间过程
    # print(f"\n最终答案:\n{answer}")

    # --- 使用方式二：启动交互式聊天 ---
    print("\n--- 交互式聊天示例 ---")
    my_agent.chat()