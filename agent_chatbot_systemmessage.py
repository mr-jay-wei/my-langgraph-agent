#conda env langgraph_env ,Python版本 3.13.5
#如果要使用fastapi server,先把user_api.py跑起来：python user_api.py
#如果要使用自定义prompt，查看prompt.md文件
#prompt管理使用SystemMessage每次在query的头部添加systemmessage
import os
import json
from typing import TypedDict, Annotated, Sequence
import operator
from datetime import datetime
import requests
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
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
    def __init__(self, model: BaseChatModel, tools: list, system_message: str | None = None):
        """
        初始化 Agent。
        - model: 一个绑定了工具的 LangChain ChatModel 实例。
        - tools: 一个包含 LangChain 工具实例的列表。
        """
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools} # 将工具列表转换为字典，方便按名称查找
        self.graph = self._build_graph()
        self.conversation_history = [] # 新增一个列表来存储历史
        self.system_message = SystemMessage(content=system_message) if system_message is not None else None

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
        # print(f"大模型回复：{response}")
        return {"messages": [response]}

    def _call_tool(self, state: AgentState) -> dict:
        """
        私有方法：调用工具。
        这是图中的 "action" 节点。
        """
        last_message = state['messages'][-1]
        '''
        last_message example
        非推理模型中如下，推理模型增加的部分是reasoning_content
        {
        'messages': 
            [AIMessage(
                content = '', 
                additional_kwargs = {
                    #reasoning_content部分仅仅在推理模型章出现
                    'reasoning_content': "我们被要求查询历史上前天的大事。\n 首先，我们需要确定“前天”的具体日期。今天的日期:  未知，需要查询。\n
                        - 前天的日期: 通过今天的日期计算得到（月、日）。\n\n
                        **反思**:\n  A. 目前还没有查询任何概念，所以都没有得到结果。\n  B. 我还没有得到任何要素。\n  C. 未得到的要素：今天的日期，前天的月份和日期。\n\n**思考**:\n  A. 要素间的依赖关系：\n      - 要得到前天的月份和日期，必须先得到今天的日期。\n      - 然后通过计算（减去两天）得到前天的日期，再提取月份和日期。\n
                        **推理**:\n  由于前天的日期依赖于今天的日期，所以第一步是获取今天的日期。\n\n
                        **计划**:\n  调用工具`get_today`，无参数。\n\n
                        **计划校验**:\n  A. 已 知常量：无。\n  B. 当前计划不涉及穷举文件记录。\n  C. 当前计划不依赖其他未获得的要素。\n  D. 当前计划没有对要素取值做假设。\n  E. 子任务未完成，需要继续。\n\n**计划改进**:\n  由于校验通过，我们执行第一步计划：调用`get_today`。\n\n 然后，当我们得到今天的日期后，再计划 下一步。",
                    'tool_calls': [{
                            'index': 0,
                            'id': 'call_0_e03c67f0-17d8-4f54-9a8f-840498745553',
                            'function': {
                                'arguments': '{"month":7,"day":13}',
                                'name': 'get_historical_events_on_date'
                                },
                            'type': 'function'
                            }]
                    }, 
                response_metadata = {
                    'finish_reason': 'tool_calls',
                    'model_name': 'deepseek-chat',
                    'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache'
                    }, 
                id = 'run--8c6be684-fc48-42e1-9b16-d94b0fb747f9-0', 
                tool_calls = [{
                        'name': 'get_historical_events_on_date',
                        'args': {
                            'month': 7,
                            'day': 13
                            },
                        'id': 'call_0_e03c67f0-17d8-4f54-9a8f-840498745553',
                        'type': 'tool_call'
                    }], 
                usage_metadata = {
                    'input_tokens': 2416,
                    'output_tokens': 27,
                    'total_tokens': 2443,
                    'input_token_details': {
                        'cache_read': 2368
                        },
                    'output_token_details': {}
                    }
                )
            ]
        }
        '''
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
        initial_messages = [self.system_message] if self.system_message else []
        current_messages = initial_messages + self.conversation_history + [HumanMessage(content=query)]
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

    my_system_prompt = '''
        你是一个名叫“智多星”的AI助手，说话风趣幽默，像一位博学的朋友。
        你的任务是尽力回答用户的问题。在回答时，请遵循以下规则：
        1. 优先使用你掌握的工具来获取最新、最准确的信息。
        2. 如果工具返回了结果，请基于工具的结果进行总结和回答，不要凭空想象。
        3. 可以使用emoji表情来增强回答的趣味性。
        4. 输出形式：
        根据以下格式说明，输出你的思考过程:
        **关键概念**: 任务中涉及的组合型概念或实体。已经明确获得取值的关键概念，将其取值完整备注在概念后。
        **概念拆解**: 将任务中的关键概念拆解为一系列待查询的子要素。每个关键概念一行，后接这个概念的子要素，每个子要素一行，行前以' -'开始。已经明确获得取值的子概念，将其取值完整备注在子概念后。
        **反思**: 自我反思，观察以前的执行记录，一步步思考以下问题:
            A. 是否每一个的关键概念或要素的查询都得到了准确的结果?
            B. 我已经得到哪个要素/概念? 得到的要素/概念取值是否正确?
            C. 从当前的信息中还不能得到哪些要素/概念。
        **思考**: 观察执行记录和你的自我反思，并一步步思考下述问题:
            A. 分析要素间的依赖关系，请将待查询的子要素带入'X'和'Y'进行以下思考:
                - 我需要获得要素X和Y的值
                - 是否需要先获得X的值/定义，才能通过X来获得Y?
                - 如果先获得X，是否可以通过X筛选Y，减少穷举每个Y的代价?
                - X和Y是否存在在同一数据源中，能否在获取X的同时获取Y?
                - 是否还有更高效或更聪明的办法来查询一个概念或要素?
                - 上一次尝试查询一个概念或要素时是否失败了? 如果是，是否可以尝试从另一个资源中再次查询?
                - 反思，我是否有更合理的方法来查询一个概念或要素?
            B. 根据以上分析，排列子要素间的查询优先级
            C. 找出当前需要获得取值的子要素
            D. 不可以使用“假设”：不要对要素的取值/定义做任何假设，确保你的信息全部来自明确的数据源！
        **推理**: 根据你的反思与思考，一步步推理被选择的子要素取值的获取方式。如果前一次的计划失败了，请检查输入中是否包含每个概念/要素的明确定义，并尝试细化你的查询描述。
        **计划**: 详细列出当前动作的执行计划。只计划一步的动作。PLAN ONE STEP ONLY!
        **计划校验**: 按照一些步骤一步步分析
            A. 有哪些已知常量可以直接代入此次分析。
            B. 当前计划是否涉及穷举一个文件中的每条记录?
                - 如果是，请给出一个更有效的方法，比如按某条件筛选，从而减少计算量;
                - 否则，请继续下一步。
            C. 上述分析是否依赖某个要素的取值/定义，且该要素的取值/定义尚未获得？如果是，重新规划当前动作，确保所有依赖的要素的取值/定义都已经获得。
            D. 当前计划是否对要素的取值/定义做任何假设？如果是，请重新规划当前动作，确保你的信息全部来自对给定的数据源的历史分析，或尝试重新从给定数据源中获取相关信息。
            E. 如果全部子任务已完成，则不必再调用工具，结束任务。
        **计划改进**:
            A. 如何计划校验中的某一步骤无法通过，请改进你的计划；
            B. 如果你的计划校验全部通过，则不必再调用工具，结束任务。
            C. 如果全部子任务已完成，则不必再调用工具，结束任务。
        '''
    


    # b. 初始化模型并绑定工具
    # llm = ChatOpenAI(temperature=0, model="gpt-4o") # 使用 gpt-4o 或 gpt-3.5-turbo 等
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0, streaming=True)

    # c. 创建 Agent 实例
    my_agent = ReActAgent(model=llm, tools=tools_list, system_message=my_system_prompt)

    # --- 使用方式一：运行单个查询 ---
    # print("--- 单次查询示例 ---")
    # question = "2024年欧洲杯的冠军是哪支球队？"
    # answer = my_agent.run(question, stream=True) # 设置 stream=True 查看中间过程
    # print(f"\n最终答案:\n{answer}")

    # --- 使用方式二：启动交互式聊天 ---
    print("\n--- 交互式聊天示例 ---")
    my_agent.chat()