# LangGraph Agent 与 MCP Server 集成项目

这个项目展示了如何将 Model Context Protocol (MCP) 服务器集成到基于 LangGraph 构建的 ReAct Agent 中，使大语言模型能够自主调用 MCP 工具。

## 项目概述

本项目包含以下核心组件：

1. **LangGraph ReAct Agent** - 基于 LangGraph 构建的智能代理，支持工具调用和推理
2. **MCP Server** - 提供各种工具功能的服务器，如 BMI 计算、天气查询等
3. **MCP 工具适配器** - 将 MCP 工具转换为 LangChain 工具格式的适配层

## 文件结构

```
├── agent_chatbot_LCEL.py          # 主要的 Agent 实现，基于 LangChain Expression Language
├── agent_chatbot_systemmessage.py # 基于系统消息的 Agent 实现
├── fastmcp_server_streamhttp.py   # MCP 服务器实现
├── mcp_tools_adapter.py           # MCP 工具适配器，将 MCP 工具转换为 LangChain 工具
├── user_api.py                    # 用户信息 API 服务
└── README.md                      # 项目说明文档
```

## 环境要求

- Python 3.10+ (MCP Server 使用 3.10.18)
- Python 3.13+ (LangGraph Agent 使用 3.13.5)

## 安装依赖

### MCP Server 环境

```bash
# 创建 MCP 环境
conda create -n mcp_env python=3.10.18
conda activate mcp_env

# 安装依赖
pip install fastmcp requests python-dotenv
```

### LangGraph Agent 环境

```bash
# 创建 LangGraph 环境
conda create -n langgraph_env python=3.13.5
conda activate langgraph_env

# 安装依赖
pip install langchain langchain-openai langchain-deepseek langchain-community
pip install langgraph fastapi uvicorn requests python-dotenv
pip install fastmcp  # 用于 MCP 客户端功能
```

## 使用方法

### 步骤 1: 启动 MCP Server

首先，在 MCP 环境中启动 MCP 服务器：

```bash
conda activate mcp_env
python fastmcp_server_streamhttp.py
```

服务器将在 `http://127.0.0.1:8083/my-custom-path` 上运行，提供以下工具：
- `calculate_bmi` - 计算 BMI 指数
- `get_current_time` - 获取当前时间
- `get_weather` - 获取指定城市的天气情况

### 步骤 2: 启动 LangGraph Agent 并集成 MCP 工具

在 LangGraph 环境中启动 Agent，并通过命令行参数指定 MCP 服务器 URL：

```bash
conda activate langgraph_env
python agent_chatbot_LCEL.py --mcp http://127.0.0.1:8083/my-custom-path
```

Agent 将自动连接到 MCP 服务器，获取可用工具，并将它们添加到工具列表中。

## 工作原理

### MCP 工具适配器

`mcp_tools_adapter.py` 文件实现了一个适配器，将 MCP 工具转换为 LangChain 工具格式：

1. 连接到 MCP 服务器并获取工具列表
2. 为每个 MCP 工具创建对应的 LangChain 工具
3. 处理工具调用和结果转换

### LangGraph Agent 集成

Agent 通过以下步骤集成 MCP 工具：

1. 解析命令行参数，获取 MCP 服务器 URL
2. 使用适配器获取 MCP 工具列表
3. 将 MCP 工具添加到 Agent 的工具列表中
4. 在 Agent 的工作流中使用这些工具

## 示例对话

```
你好！我是 ReAct Agent。输入 'quit' 或 'exit' 退出。
输入 'stats' 可以查看当前会话的 token 统计信息。

--- 交互式聊天示例 ---

你: 我身高1.75米，体重70公斤，我的BMI是多少？

ReAct Agent: **关键概念**: BMI指数
- 身高: 1.75米
- 体重: 70公斤

**概念拆解**: 
BMI指数
 - 身高: 1.75米
 - 体重: 70公斤
 - BMI计算公式: 体重(kg) / 身高(m)²

**反思**: 
A. 我已经获得了身高和体重的数值，但还没有计算BMI。
B. 已知信息: 身高=1.75米，体重=70公斤。
C. 未获得: BMI指数的具体数值。

**思考**: 
A. 我需要获得BMI指数的值。
B. 我可以使用calculate_bmi工具，该工具需要身高和体重作为输入参数。
C. 当前需要获取的是BMI指数值。

**推理**: 
我已经有了身高(1.75米)和体重(70公斤)，可以直接调用calculate_bmi工具计算BMI值。

**计划**: 
调用calculate_bmi工具，参数为weight_kg=70, height_m=1.75。

**计划校验**: 
A. 已知常量: 身高=1.75米，体重=70公斤。
B. 当前计划不涉及穷举文件记录。
C. 当前计划不依赖其他未获得的子要素。
D. 当前计划没有对子要素取值做假设。
E. 子任务未完成，需要继续。

根据您提供的身高(1.75米)和体重(70公斤)，您的BMI指数是22.86。

这属于正常体重范围(18.5-24.9)，表明您的体重对健康是有益的。保持良好的饮食习惯和适当的运动可以维持这个健康的BMI值。
```

## 高级功能

### 命令行参数

`agent_chatbot_LCEL.py` 支持以下命令行参数：

- `--mcp <url>`: 指定 MCP 服务器 URL

### 交互命令

在聊天界面中，你可以使用以下命令：
- `stats`: 查看 Token 使用统计
- `quit` 或 `exit`: 退出程序

## 扩展 MCP 工具

要向 MCP 服务器添加新工具，只需在 `fastmcp_server_streamhttp.py` 中添加新的工具函数：

```python
@mcp.tool()
def new_tool_function(param1: type, param2: type) -> return_type:
    """工具描述...
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
    """
    # 工具实现...
    return result
```

重启 MCP 服务器后，新工具将自动对 Agent 可用。

## 故障排除

### MCP 连接问题

如果 Agent 无法连接到 MCP 服务器，请检查：

1. MCP 服务器是否正在运行
2. URL 是否正确
3. 网络连接是否正常
4. 代理设置是否需要调整

### 工具调用失败

如果工具调用失败，可能的原因包括：

1. 参数类型不匹配
2. 网络连接问题
3. 工具内部错误

查看控制台输出以获取详细错误信息。

## 进一步开发

- 添加更多 MCP 工具
- 实现异步工具调用
- 添加工具结果缓存
- 改进错误处理和重试机制
- 添加用户界面

## 许可证

MIT

## 作者

[Jay Wei]

## 致谢

- LangChain 团队
- LangGraph 项目
- FastMCP 项目