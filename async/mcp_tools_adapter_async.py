# conda env mcp_env ,Python版本 3.10.18
#如果要测试mcp server,先把prox关闭，fastmcp_server_async.py跑起来：python fastmcp_server_async.py
#使用python mcp_tools_adapter_async.py http://127.0.0.1:8084/my-custom-path

"""
MCP工具适配器 (异步版本) - 将MCP Server工具转换为LangChain工具
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Union
from fastmcp import Client
from langchain.tools import BaseTool
from pydantic import BaseModel, create_model
from functools import wraps

class MCPToolAdapter:
    """
    将MCP Server工具转换为LangChain工具的适配器类 (异步版本)
    """
    def __init__(self, mcp_server_url: str):
        """
        初始化MCP工具适配器
        
        Args:
            mcp_server_url: MCP服务器URL，例如 "http://127.0.0.1:8084/my-custom-path"
        """
        self.mcp_server_url = mcp_server_url
        self.mcp_tools_cache = None  # 缓存MCP工具列表
        
    async def _get_mcp_tools_async(self):
        """异步获取MCP工具列表"""
        if self.mcp_tools_cache is not None:
            return self.mcp_tools_cache
            
        async with Client(self.mcp_server_url) as client:
            tool_list = await client.list_tools()
            self.mcp_tools_cache = tool_list
            return tool_list
    
    def get_mcp_tools(self):
        """同步获取MCP工具列表"""
        # 创建一个新的事件循环来运行异步函数
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._get_mcp_tools_async())
        finally:
            loop.close()
    
    async def call_mcp_tool_async(self, tool_name: str, **kwargs):
        """异步调用MCP工具"""
        async with Client(self.mcp_server_url) as client:
            result = await client.call_tool(tool_name, kwargs)
            return result.content[0].text
    
    def call_mcp_tool(self, tool_name: str, **kwargs):
        """同步调用MCP工具"""
        # 创建一个新的事件循环来运行异步函数
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.call_mcp_tool_async(tool_name, **kwargs))
        finally:
            loop.close()
    
    # 定义一个通用的MCP工具类
    class MCPToolWrapper(BaseTool):
        """通用MCP工具包装类"""
        # 声明额外的字段，使用正确的类型提示
        # 这些是Pydantic模型字段，用于类型验证
        call_mcp_tool: Callable[[str, Any], Any]
        call_mcp_tool_async: Callable[[str, Any], Any]
        
        def __init__(self, name, description, call_func, call_async_func):
            """初始化MCP工具包装器"""
            # 创建包含所有字段的字典
            # 这些将成为实例属性
            kwargs = {
                "name": name,  # 将成为self.name
                "description": description,  # 将成为self.description
                "call_mcp_tool": call_func,  # 将成为self.call_mcp_tool
                "call_mcp_tool_async": call_async_func  # 将成为self.call_mcp_tool_async
            }
            # 调用父类初始化方法，传入所有字段
            # 这会设置所有实例属性
            super().__init__(**kwargs)
        
        def _run(self, tool_input=None, **kwargs):
            """
            同步运行工具
            
            Args:
                tool_input: 工具输入，可以是字典、字符串或其他类型，默认为None
                **kwargs: 额外参数，可能由LangChain框架传入
            """
            # 打印工具名称，验证self.name的存在
            print(f"调用工具: {self.name}")
            
            # 处理tool_input为None的情况
            if tool_input is None:
                return self.call_mcp_tool(self.name)
            # 如果tool_input是字典，合并它和kwargs
            elif isinstance(tool_input, dict):
                combined_kwargs = {**tool_input, **kwargs}
                return self.call_mcp_tool(self.name, **combined_kwargs)
            # 如果tool_input是字符串但为空，只传递kwargs
            elif isinstance(tool_input, str) and not tool_input:
                if kwargs:
                    return self.call_mcp_tool(self.name, **kwargs)
                else:
                    return self.call_mcp_tool(self.name)
            # 其他情况，将tool_input作为input参数，并合并kwargs
            else:
                return self.call_mcp_tool(self.name, input=tool_input, **kwargs)
            
        async def _arun(self, **kwargs):
            """
            异步运行工具 - 重写为不需要tool_input参数的版本
            
            Args:
                **kwargs: 额外参数，可能由LangChain框架传入
            """
            # 直接调用MCP工具，传递kwargs
            if kwargs:
                return await self.call_mcp_tool_async(self.name, **kwargs)
            else:
                return await self.call_mcp_tool_async(self.name)
    
    def create_langchain_tool(self, mcp_tool):
        """
        将单个MCP工具转换为LangChain工具
        
        Args:
            mcp_tool: MCP工具对象
            
        Returns:
            BaseTool: LangChain工具实例
        """
        # 创建工具实例
        tool_instance = self.MCPToolWrapper(
            name=mcp_tool.name,
            description=mcp_tool.description,
            call_func=self.call_mcp_tool,
            call_async_func=self.call_mcp_tool_async
        )
        
        return tool_instance
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """
        获取所有MCP工具转换后的LangChain工具列表
        
        Returns:
            List[BaseTool]: LangChain工具列表
        """
        mcp_tools = self.get_mcp_tools()
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            langchain_tool = self.create_langchain_tool(mcp_tool)
            langchain_tools.append(langchain_tool)
            
        return langchain_tools

    async def get_langchain_tools_async(self) -> List[BaseTool]:
        """
        异步获取所有MCP工具转换后的LangChain工具列表
        
        Returns:
            List[BaseTool]: LangChain工具列表
        """
        mcp_tools = await self._get_mcp_tools_async()
        langchain_tools = []
        
        for mcp_tool in mcp_tools:
            langchain_tool = self.create_langchain_tool(mcp_tool)
            langchain_tools.append(langchain_tool)
            
        return langchain_tools


# 便捷函数，用于直接获取LangChain工具列表
def get_mcp_tools_as_langchain(mcp_server_url: str) -> List[BaseTool]:
    """
    从MCP服务器获取工具并转换为LangChain工具列表
    
    Args:
        mcp_server_url: MCP服务器URL
        
    Returns:
        List[BaseTool]: LangChain工具列表
    """
    adapter = MCPToolAdapter(mcp_server_url)
    return adapter.get_langchain_tools()

# 便捷函数，用于异步获取LangChain工具列表
async def get_mcp_tools_as_langchain_async(mcp_server_url: str) -> List[BaseTool]:
    """
    异步从MCP服务器获取工具并转换为LangChain工具列表
    
    Args:
        mcp_server_url: MCP服务器URL
        
    Returns:
        List[BaseTool]: LangChain工具列表
    """
    adapter = MCPToolAdapter(mcp_server_url)
    return await adapter.get_langchain_tools_async()


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python mcp_tools_adapter_async.py <mcp_server_url>")
        print("例如: python mcp_tools_adapter_async.py http://127.0.0.1:8084/my-custom-path")
        sys.exit(1)
    
    mcp_server_url = sys.argv[1]
    
    # 性能测试函数
    async def performance_test():
        print("\n===== 性能测试: 同步 vs 异步 =====")
        adapter = MCPToolAdapter(mcp_server_url)
        
        # 直接使用异步方法获取工具列表
        mcp_tools = await adapter._get_mcp_tools_async()
        
        # 手动创建工具列表，避免使用同步方法
        sync_tools = []
        async_tools = []
        for mcp_tool in mcp_tools:
            tool_instance = adapter.create_langchain_tool(mcp_tool)
            sync_tools.append(tool_instance)
            async_tools.append(tool_instance)
        
        print(f"成功加载 {len(sync_tools)} 个工具")
        
        # 测试工具调用性能
        if len(sync_tools) > 0:
            print("\n===== 测试工具调用性能 =====")
            
            # 测试同步调用 - 使用直接调用adapter的方法，避免事件循环嵌套
            start_time = time.time()
            results = []
            for i in range(5):  # 调用5次
                for tool in sync_tools:
                    if tool.name == "get_current_time":
                        # 直接使用异步方法，但按顺序执行
                        # 注意：call_mcp_tool_async 使用 **kwargs，所以不能传递位置参数
                        result = await adapter.call_mcp_tool_async(tool.name)
                        results.append(result)
            sync_call_time = time.time() - start_time
            print(f"同步调用工具耗时: {sync_call_time:.4f} 秒")
            
            # 测试异步调用
            start_time = time.time()
            tasks = []
            for i in range(5):  # 调用5次
                for tool in async_tools:
                    if tool.name == "get_current_time":
                        tasks.append(tool._arun())
            results = await asyncio.gather(*tasks)
            async_call_time = time.time() - start_time
            print(f"异步调用工具耗时: {async_call_time:.4f} 秒")
            print(f"性能提升: {(sync_call_time - async_call_time) / sync_call_time * 100:.2f}%")
    
    # 运行性能测试
    asyncio.run(performance_test())