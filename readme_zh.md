# AI智能聊天机器人项目

## 项目简介

这是一个基于LangGraph和LangChain构建的智能聊天机器人项目，具备工具调用能力和ReAct推理模式。机器人可以通过调用各种工具来获取实时信息，如天气查询、历史事件查询、网络搜索等，为用户提供准确和及时的回答。

## 主要特性

- 🤖 **智能对话**: 基于大语言模型的自然语言理解和生成
- 🔧 **工具调用**: 支持多种外部工具集成，包括天气查询、搜索引擎、历史事件查询等
- 🧠 **ReAct推理**: 采用推理-行动-观察的循环模式，提供结构化的思考过程
- 📊 **Token统计**: 实时监控和统计API调用的Token使用情况
- 💬 **对话记忆**: 支持多轮对话，保持上下文连贯性
- 🌐 **API服务**: 包含FastAPI后端服务，支持用户信息查询

## 项目结构

```
├── agent_chatbot_systemmessage.py  # 基于系统消息的聊天机器人实现
├── agent_chatbot_LCEL.py          # 基于LCEL(LangChain Expression Language)的实现
├── user_api.py                    # FastAPI用户信息查询服务
├── .gitignore                     # Git忽略文件配置
└── readme_zh.md                   # 项目说明文档(中文)
```

## 环境要求

- Python 3.13.5
- conda环境: `langgraph_env`

## 依赖包

主要依赖包包括：
- `langchain` - LangChain核心库
- `langchain-openai` - OpenAI集成
- `langchain-deepseek` - DeepSeek模型集成
- `langchain-community` - 社区工具集
- `langgraph` - 图形化工作流框架
- `fastapi` - Web API框架
- `uvicorn` - ASGI服务器
- `requests` - HTTP请求库
- `python-dotenv` - 环境变量管理

## 安装和配置

1. **创建conda环境**
   ```bash
   conda create -n langgraph_env python=3.13.5
   conda activate langgraph_env
   ```

2. **安装依赖包**
   ```bash
   pip install langchain langchain-openai langchain-deepseek langchain-community
   pip install langgraph fastapi uvicorn requests python-dotenv
   ```

3. **环境变量配置**
   创建`.env`文件并配置必要的API密钥：
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key
   TAVILY_API_KEY=your_tavily_api_key
   OPENAI_API_KEY=your_openai_api_key  # 可选
   ```

## 使用方法

### 启动FastAPI服务（可选）

如果需要使用用户信息查询功能：

1. 关闭代理设置
2. 启动API服务：
   ```bash
   python user_api.py
   ```
   服务将在 `http://127.0.0.1:8000` 运行

### 运行聊天机器人

有两种实现方式可选择：

**方式一：基于系统消息的实现**
```bash
python agent_chatbot_systemmessage.py
```

**方式二：基于LCEL的实现**
```bash
python agent_chatbot_LCEL.py
```

### 交互命令

在聊天界面中，你可以使用以下命令：
- 正常对话：直接输入问题
- `stats` - 查看Token使用统计
- `quit` 或 `exit` - 退出程序

## 功能特性详解

### 可用工具

1. **网络搜索** (`internet_search_engine`)
   - 使用Tavily搜索引擎获取最新网络信息
   - 最多返回2个搜索结果

2. **天气查询** (`get_weather`)
   - 查询全球主要城市的实时天气
   - 返回温度、体感温度、天气状况和湿度

3. **日期查询** (`get_today`)
   - 获取当前系统日期
   - 格式：YYYY-MM-DD

4. **历史事件查询** (`get_historical_events_on_date`)
   - 查询指定日期的历史事件
   - 需要提供月份和日期参数

5. **用户信息查询** (`get_user_info`) - 可选
   - 从内部API查询用户信息
   - 需要启动FastAPI服务

### ReAct推理模式

机器人采用结构化的推理过程：

1. **关键概念** - 识别任务中的核心概念
2. **概念拆解** - 将复杂概念分解为可查询的子要素
3. **反思** - 分析已获得的信息和缺失的部分
4. **思考** - 分析要素间的依赖关系和查询优先级
5. **推理** - 制定获取信息的策略
6. **计划** - 制定具体的执行步骤
7. **计划校验** - 验证计划的可行性
8. **计划改进** - 优化和调整执行方案

### Token统计功能

- 实时监控输入/输出Token使用量
- 缓存Token统计和效率分析
- 会话级别的使用统计报告
- 成本控制和优化建议

## 示例对话

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

## 注意事项

1. **网络连接**: 某些工具需要访问外部API，请确保网络连接正常
2. **代理设置**: 如果使用FastAPI服务，建议关闭代理
3. **API密钥**: 确保在`.env`文件中正确配置所需的API密钥
4. **错误处理**: 程序包含完善的错误处理机制，网络问题会有相应提示

## 技术架构

- **LangGraph**: 用于构建复杂的AI工作流图
- **LangChain**: 提供LLM集成和工具管理
- **DeepSeek**: 主要使用的大语言模型
- **FastAPI**: 轻量级Web框架，用于API服务
- **Pydantic**: 数据验证和序列化

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
- 提交GitHub Issue
- 发送邮件至项目维护者

---

*最后更新时间：2024年*