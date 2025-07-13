当然可以！自主配置 Prompt 是 LangChain 和 `langgraph` 设计的一大亮点，它让你能够从框架的“默认行为”中脱离出来，实现更高级、更个性化的 Agent 行为。

当你觉得默认的 ReAct 风格 Prompt 不足以满足你的需求时，比如：
*   你想让 Agent 有特定的**身份或个性**（如“你是一个风趣幽默的旅游向导”）。
*   你想给出更**严格的指令**（如“永远不要猜测，如果信息不足就说不知道”）。
*   你想改变 Agent **响应的格式**。
*   你想实现一种**全新的、非 ReAct 风格**的 Agent 逻辑。

你完全可以接管 Prompt 的构建过程。主要有两种方式可以实现：

1.  **简单方式：使用 `SystemMessage`**
2.  **高级方式：使用 LangChain 表达式语言 (LCEL) 自定义 PromptTemplate**

---

### **方法一：使用 `SystemMessage` (最简单直接)**

这是最简单的方法，适用于在默认 ReAct 行为的基础上，增加系统级的指令、身份设定或规则。

你只需要在每次调用图之前，在 `messages` 列表的**最前面**插入一个 `SystemMessage`。

**如何修改你的代码：**

在你的 `ReActAgent` 类的 `run` 和 `chat` 方法中进行修改。

```python
# 在 ReActAgent 类中

# ... (其他代码不变)

# 【新增】在类的初始化中定义你的系统消息
def __init__(self, model: BaseChatModel, tools: list, system_message: str | None = None):
    self.model = model.bind_tools(tools)
    self.tools = {t.name: t for t in tools}
    self.graph = self._build_graph()
    self.conversation_history = []
    # 将系统消息字符串包装成 SystemMessage 对象
    self.system_message = SystemMessage(content=system_message) if system_message is not None else None

def run(self, query: str, stream: bool = True) -> str:
    # 【修改】在这里注入 SystemMessage
    initial_messages = [self.system_message] if self.system_message else []
    current_messages = initial_messages + self.conversation_history + [HumanMessage(content=query)]
    
    # 后续逻辑不变
    inputs = {"messages": current_messages}
    # ... (run 方法的其余部分)

def chat(self):
    # 【修改】chat 方法也要做类似修改，确保上下文一致
    # ...
    # 在准备输入时
    initial_messages = [self.system_message] if self.system_message else []
    current_messages = initial_messages + self.conversation_history + [HumanMessage(content=user_input)]
    inputs = {"messages": current_messages}
    # ...

# --- 主程序入口 ---
if __name__ == "__main__":
    # ... (工具和模型初始化不变)

    # 【新增】定义你想要的系统 Prompt
    my_system_prompt = (
        "你是一个名叫“智多星”的AI助手，说话风趣幽默，像一位博学的朋友。"
        "你的任务是尽力回答用户的问题。在回答时，请遵循以下规则：\n"
        "1. 优先使用你掌握的工具来获取最新、最准确的信息。\n"
        "2. 如果工具返回了结果，请基于工具的结果进行总结和回答，不要凭空想象。\n"
        "3. 在回答的结尾，俏皮地加上你的署名'——来自你的朋友，智多星'。"
    )

    # 【修改】在创建 Agent 实例时传入
    my_agent = ReActAgent(model=llm, tools=tools_list, system_message=my_system_prompt)
    
    my_agent.chat()
```

**工作原理：**
`SystemMessage` 在对话中扮演着一个“上帝视角”的角色，它为整个对话设定了基调和规则。LangChain 的 ReAct 框架会尊重这个 `SystemMessage`，并将其与它自己的 ReAct 指令结合起来，最终发送给 LLM。这样，LLM 就会在遵循 ReAct 循环的同时，努力扮演你设定的角色和遵守你给出的规则。

---

### **方法二：使用 LCEL 完全自定义 Prompt (更灵活、更强大)**

当你想要彻底改变 Agent 的思考方式，而不仅仅是添加一个身份时，你需要完全重写调用模型的节点。这通常涉及到使用 `ChatPromptTemplate`。

假设你想创建一个 Agent，它在调用工具前会先“自言自语”地解释它为什么这么做。

**如何修改你的代码：**

这次的修改会更集中在 `_call_model` 方法上。

```python
# 导入 ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

class ReActAgent:
    def __init__(self, model: BaseChatModel, tools: list):
        # 注意：这里我们不再在初始化时绑定工具，因为我们将在 prompt 中手动处理
        self.raw_model = model 
        self.tools = tools # 保存原始工具列表
        self.tools_map = {t.name: t for t in tools} # 保存工具字典
        self.graph = self._build_graph()
        self.conversation_history = []
    
    def _call_model(self, state: AgentState) -> dict:
        """
        【重大修改】私有方法：使用自定义 Prompt 调用大模型。
        """
        messages = state['messages']
        
        # 1. 定义你的 Prompt 模板
        #    这个模板会接收 'messages' 和 'tools' 作为输入变量
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一个强大的AI助手。你的任务是根据用户的对话历史，决定下一步是直接回答还是使用工具。\n"
                    "你可以使用以下工具：\n\n{tools}\n\n"
                    "请分析用户的最新问题。如果你需要使用工具，请以工具调用的形式回应。"
                    "如果你能直接回答，就直接给出答案。"
                ),
                ("placeholder", "{messages}"), # placeholder 会被 messages 列表替换
            ]
        )
        
        # 2. 将工具渲染成字符串，以便插入到 Prompt 中
        #    LangChain 有工具可以帮你做这个，但手动做也很简单
        rendered_tools = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        # 3. 将模型绑定工具，并与 Prompt 组合成一个 chain
        #    注意：我们在这里才进行工具绑定
        model_with_tools = self.raw_model.bind_tools(self.tools)
        chain = prompt | model_with_tools
        
        # 4. 调用 chain
        response = chain.invoke({"messages": messages, "tools": rendered_tools})
        
        return {"messages": [response]}

    # ... 其他方法基本不变，除了 __init__ ...
```

**工作原理：**

1.  **解绑工具**: 在 `__init__` 中，我们不再立即将模型与工具绑定，而是保存原始的模型和工具列表。这给了我们稍后在 Prompt 中操作的灵活性。
2.  **`ChatPromptTemplate`**: 我们定义了一个包含 `system` 消息和 `placeholder` 的模板。`placeholder` 是一个强大的功能，它允许我们将一个完整的消息列表（如对话历史）动态地插入到 Prompt 的正确位置。
3.  **渲染工具**: 我们手动将工具列表格式化成一个描述性字符串，并将其作为变量 `tools` 传给 Prompt。
4.  **构建 LCEL Chain**: `chain = prompt | model_with_tools` 这行代码是 LCEL 的精髓。它创建了一个执行管道：
    *   首先，输入数据（一个包含 `messages` 和 `tools` 的字典）被送入 `prompt`。
    *   `prompt` 将输入格式化成一个完整的、准备好的消息列表。
    *   这个消息列表被“管道”传送到 `model_with_tools`。
    *   模型执行调用并返回结果。
5.  **在 `_call_model` 中执行**: 我们用这个自定义的 `chain` 替换了原来简单的 `model.invoke()`。

### **总结与建议**

| 方法 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **`SystemMessage`** | **简单、快速**，不改变核心逻辑，与默认 ReAct 行为兼容性好。 | 控制力有限，无法改变 ReAct 的根本流程。 | 为 Agent 添加**身份、个性、高级规则**，或给出一般性指示。 |
| **LCEL 自定义 Prompt** | **控制力极强**，可以完全重新定义 Agent 的思考和行为模式。 | **更复杂**，需要理解 LCEL 和 Prompt Engineering。可能会“破坏”默认的 ReAct 循环，需要你仔细设计。 | 创建**非标准 Agent**（如 Plan-and-Execute Agent），或对 Prompt 的每个细节都有严格要求。 |

对于你的情况，我强烈建议**从方法一开始**。它已经能满足绝大多数的定制化需求，而且风险最低。当你发现 `SystemMessage` 已经无法满足你的需求，想要从根本上改变 Agent 的决策逻辑时，再转向更强大的方法二。