# AI智能聊天机器人项目：LangGraph Agent 与 MCP Server 集成

## 项目简介

这是一个基于LangGraph和LangChain构建的智能聊天机器人项目，具备工具调用能力和ReAct推理模式。机器人可以通过调用各种工具来获取实时信息，如天气查询、历史事件查询、网络搜索等，为用户提供准确和及时的回答。项目特别展示了如何将Model Context Protocol (MCP) 服务器集成到基于LangGraph构建的ReAct Agent中，使大语言模型能够自主调用MCP工具。

## 主要特性

- 🤖 **智能对话**: 基于大语言模型的自然语言理解和生成
- 🔧 **工具调用**: 支持多种外部工具集成，包括天气查询、搜索引擎、历史事件查询等
- 🧠 **ReAct推理**: 采用推理-行动-观察的循环模式，提供结构化的思考过程
- 📊 **Token统计**: 实时监控和统计API调用的Token使用情况
- 💬 **对话记忆**: 支持多轮对话，保持上下文连贯性
- 🌐 **API服务**: 包含FastAPI后端服务，支持用户信息查询
- 🔌 **MCP集成**: 支持Model Context Protocol服务器工具的无缝集成
- ⚡ **同步/异步**: 同时提供同步和异步实现，满足不同性能需求

## 项目架构

本项目包含以下核心组件：

1. **LangGraph ReAct Agent** - 基于LangGraph构建的智能代理，支持工具调用和推理
2. **MCP Server** - 提供各种工具功能的服务器，如BMI计算、天气查询等
3. **MCP工具适配器** - 将MCP工具转换为LangChain工具格式的适配层
4. **FastAPI服务** - 提供用户信息查询等API服务

## 项目结构

```
├── sync/                          # 同步实现目录
│   ├── agent_chatbot_LCEL.py      # 基于LCEL的Agent实现(同步)
│   ├── agent_chatbot_systemmessage.py # 基于系统消息的Agent实现(同步)
│   ├── fastmcp_server_streamhttp.py # MCP服务器实现(同步)
│   ├── mcp_tools_adapter.py       # MCP工具适配器(同步)
│   └── user_api.py                # 用户信息API服务(同步)
│
├── async/                         # 异步实现目录
│   ├── agent_chatbot_async.py     # 异步版本的Agent实现
│   ├── fastmcp_server_async.py    # 异步版本的MCP服务器
│   ├── mcp_tools_adapter_async.py # 异步版本的MCP工具适配器
│   └── user_api_async.py          # 异步版本的用户信息API服务
│
├── .gitignore                     # Git忽略文件配置
├── requirements.txt               # 依赖包列表
└── README.md                   # 项目总体说明文档(中文)
```

## 核心理念

整个项目的核心理念是**工具标准化**：

- 所有工具（本地函数、API服务、MCP服务）必须提供标准化的接口
- 每个工具必须有明确的`name`（工具名称）
- 每个工具必须有详细的`description`（工具描述）
- 每个工具必须定义清晰的参数结构（`args`）

这种标准化使得大模型能够理解工具功能并正确调用，同时也使不同来源的工具可以无缝集成。## 环境要求


项目包含两个主要环境：

### MCP Server 环境

- Python 3.10+ (推荐 3.10.18)
- conda环境: `mcp_env`

### LangGraph Agent 环境

- Python 3.13+ (推荐 3.13.5)
- conda环境: `langgraph_env`

## 依赖包安装

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

### 异步环境额外依赖

```bash
# 在 LangGraph 环境中安装异步支持
pip install aiohttp
```

## 环境变量配置

创建`.env`文件并配置必要的API密钥：

```
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENAI_API_KEY=your_openai_api_key  # 可选
KIMI_API_KEY=your_kimi_api_key      # 可选，用于异步版本
```

## 使用方法

### 同步版本

#### 步骤 1: 启动 MCP Server

首先，在 MCP 环境中启动 MCP 服务器：

```bash
conda activate mcp_env
python sync/fastmcp_server_streamhttp.py
```

服务器将在 `http://127.0.0.1:8083/my-custom-path` 上运行，提供以下工具：
- `calculate_bmi` - 计算 BMI 指数
- `get_current_time` - 获取当前时间
- `get_weather` - 获取指定城市的天气情况

#### 步骤 2: 启动用户API服务（可选）

如果需要使用用户信息查询功能：

```bash
conda activate langgraph_env
python sync/user_api.py
```

服务将在 `http://127.0.0.1:8000` 运行。

#### 步骤 3: 启动 LangGraph Agent

在 LangGraph 环境中启动 Agent，并通过命令行参数指定 MCP 服务器 URL：

```bash
conda activate langgraph_env
python sync/agent_chatbot_LCEL.py --mcp http://127.0.0.1:8083/my-custom-path
```

Agent 将自动连接到 MCP 服务器，获取可用工具，并将它们添加到工具列表中。

### 异步版本

#### 步骤 1: 启动异步 MCP Server

```bash
conda activate mcp_env
python async/fastmcp_server_async.py
```

服务器将在 `http://127.0.0.1:8084/my-custom-path` 上运行。

#### 步骤 2: 启动异步用户API服务（可选）

```bash
conda activate langgraph_env
python async/user_api_async.py
```

服务将在 `http://127.0.0.1:8001` 运行。

#### 步骤 3: 启动异步 LangGraph Agent

```bash
conda activate langgraph_env
python async/agent_chatbot_async.py
```

#### 步骤 4: 运行性能测试（可选）

比较同步和异步版本的性能差异：

```bash
python async/agent_chatbot_async.py --perf-test
```

### 命令行参数

#### 同步版本参数

`agent_chatbot_LCEL.py` 支持以下命令行参数：

- `--mcp <url>`: 主MCP服务器URL
- `--mcp2 <url>`: 第二个MCP服务器URL
- `--no-mcp`: 禁用所有MCP工具

#### 异步版本参数

`agent_chatbot_async.py` 支持以下命令行参数：

- `--mcp <url>`: MCP服务器URL
- `--no-mcp`: 禁用MCP工具
- `--perf-test`: 运行性能测试

### 交互命令

在聊天界面中，你可以使用以下命令：
- `stats`: 查看 Token 使用统计
- `quit` 或 `exit`: 退出程序## 
工作原理

### LangGraph与MCP的统合

本项目展示了两种不同技术的完美结合：

1. **LangGraph** - 提供了强大的工作流控制和状态管理能力，使Agent能够进行复杂的推理和决策
2. **MCP (Model Context Protocol)** - 提供了标准化的工具接口，使外部工具能够被大模型无缝调用

这两种技术看似独立，但通过**MCP工具适配器**实现了统一：

```
LangGraph Agent <---> MCP工具适配器 <---> MCP Server
```

### MCP工具适配器

`mcp_tools_adapter.py` 文件实现了一个适配器，将MCP工具转换为LangChain工具格式：

1. 连接到MCP服务器并获取工具列表
2. 为每个MCP工具创建对应的LangChain工具
3. 处理工具调用和结果转换

### 同步与异步的对比

项目同时提供了同步和异步实现，两者各有优势：

**同步实现**:
- 代码简单直观，易于理解
- 适合简单场景和学习使用
- 按顺序执行，便于调试

**异步实现**:
- 性能更高，特别是在I/O密集型操作中
- 可以同时处理多个请求
- 减少等待时间，提高吞吐量

在性能测试中，异步版本通常能够显示出明显的性能优势，特别是在涉及多个网络请求的情况下。

## 可用工具详解

### 本地工具

1. **网络搜索** (`internet_search_engine`)
   - 使用Tavily搜索引擎获取最新网络信息
   - 最多返回2个搜索结果

2. **日期查询** (`get_today`/`get_today_async`)
   - 获取当前系统日期
   - 格式：YYYY-MM-DD

3. **历史事件查询** (`get_historical_events_on_date`/`get_historical_events_on_date_async`)
   - 查询指定日期的历史事件
   - 需要提供月份和日期参数

4. **用户信息查询** (`get_user_info`/`get_user_info_async`)
   - 从内部API查询用户信息
   - 需要启动FastAPI服务

### MCP服务器工具

1. **BMI计算** (`calculate_bmi`)
   - 计算体重指数
   - 参数：体重(kg)和身高(m)

2. **当前时间** (`get_current_time`)
   - 获取当前系统时间
   - 无需参数

3. **天气查询** (`get_weather`)
   - 获取指定城市的天气情况
   - 参数：城市名称和日期

## ReAct推理模式

机器人采用结构化的推理过程：

1. **关键概念** - 识别任务中的核心概念
2. **概念拆解** - 将复杂概念分解为可查询的子要素
3. **反思** - 分析已获得的信息和缺失的部分
4. **思考** - 分析要素间的依赖关系和查询优先级
5. **推理** - 制定获取信息的策略
6. **计划** - 制定具体的执行步骤
7. **计划校验** - 验证计划的可行性
8. **计划改进** - 优化和调整执行方案

## 示例对话

### BMI计算示例

```
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

### 天气查询示例

```
你: 今天北京的天气怎么样？

ReAct Agent: **关键概念**: 今天北京的天气状况

**概念拆解**: 
- 今天的日期
- 北京的天气信息

**反思**: 
A. 需要获取今天的日期和北京的天气信息
B. 目前还没有获得任何信息
C. 缺少：当前日期、北京天气数据

**思考**: 
A. 可以直接查询北京天气，不需要先获取日期
B. 天气查询工具可以直接提供实时信息

**推理**: 直接调用天气查询工具获取北京的实时天气

**计划**: 调用get_weather工具，参数为"Beijing"

北京当前天气：晴，气温25°C，体感温度27°C，湿度45%。今天是个不错的天气呢！☀️
```
## 扩展开发

### 添加新的MCP工具

要向MCP服务器添加新工具，只需在`fastmcp_server_streamhttp.py`或`fastmcp_server_async.py`中添加新的工具函数：

```python
# 同步版本
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

# 异步版本
@mcp.tool()
async def new_tool_function_async(param1: type, param2: type) -> return_type:
    """工具描述...
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
        
    Returns:
        返回值描述
    """
    # 异步工具实现...
    await asyncio.sleep(0.01)  # 模拟异步操作
    return result
```

在MCP服务器中添加新工具并重启后，需要同时重启Agent，新工具才会对Agent可用。

### 多MCP服务器支持

同步版本的Agent支持连接多个MCP服务器：

```bash
python sync/agent_chatbot_LCEL.py --mcp http://server1:8083/path --mcp2 http://server2:8086/path
```

如果两个服务器提供了同名工具，可以考虑在加载工具时添加前缀，避免冲突。

### 性能优化

1. **缓存机制**
   - 缓存频繁调用的工具结果
   - 设置缓存过期时间
   - 实现缓存失效策略

2. **并发控制**
   - 限制并发请求数量
   - 实现请求队列和优先级
   - 添加超时和重试机制

3. **负载均衡**
   - 在多个提供相同功能的MCP服务器之间分配请求
   - 实现健康检查和故障转移

## 故障排除

### MCP连接问题

如果Agent无法连接到MCP服务器，请检查：

1. MCP服务器是否正在运行
2. URL是否正确
3. 网络连接是否正常
4. 代理设置是否需要调整

### 工具调用失败

如果工具调用失败，可能的原因包括：

1. 参数类型不匹配
   - 大模型可能返回字符串形式的数字，而工具需要数值类型
   - 解决：在工具适配器中添加类型转换逻辑

2. 工具描述不清晰
   - 如果工具描述不够详细，大模型可能无法正确使用
   - 解决：提供详细的参数说明、示例和使用场景

3. 网络连接问题
   - 外部API可能暂时不可用
   - 解决：添加错误处理和重试机制

### 性能问题

如果遇到性能问题，可以考虑：

1. 切换到异步版本
2. 减少不必要的网络请求
3. 优化工具实现
4. 添加缓存机制

## 技术架构

- **LangGraph**: 用于构建复杂的AI工作流图
- **LangChain**: 提供LLM集成和工具管理
- **FastMCP**: 提供标准化的工具接口
- **DeepSeek/OpenAI/Kimi**: 大语言模型提供商
- **FastAPI**: 轻量级Web框架，用于API服务
- **Asyncio/Aiohttp**: 异步编程支持

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。在贡献代码前，请确保：

1. 代码符合项目的编码规范
2. 添加必要的注释和文档
3. 测试新功能的正确性
4. 更新相关文档

## 许可证

本项目采用MIT许可证，详情请参阅LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue  url:https://github.com/mr-jay-wei/my-langgraph-agent
- 发送邮件至项目维护者 Jay Wei email:xiaofeng.0209@gmail.com

---

*最后更新时间：2025年7月*