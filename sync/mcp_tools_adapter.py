"""
MCP工具适配器 - 将MCP Server工具转换为LangChain工具
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Union
from fastmcp import Client
from langchain.tools import BaseTool
from pydantic import BaseModel, create_model
from functools import wraps

class MCPToolAdapter:
    """
    将MCP Server工具转换为LangChain工具的适配器类
    """
    def __init__(self, mcp_server_url: str):
        """
        初始化MCP工具适配器
        
        Args:
            mcp_server_url: MCP服务器URL，例如 "http://127.0.0.1:8083/my-custom-path"
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
        return asyncio.run(self._get_mcp_tools_async())
    
    async def call_mcp_tool_async(self, tool_name: str, **kwargs):
        """异步调用MCP工具"""
        async with Client(self.mcp_server_url) as client:
            result = await client.call_tool(tool_name, kwargs)
            return result.content[0].text
    
    def call_mcp_tool(self, tool_name: str, **kwargs):
        """同步调用MCP工具"""
        return asyncio.run(self.call_mcp_tool_async(tool_name, **kwargs))
    
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
            
        def _run(self, tool_input, **kwargs):
            """
            同步运行工具
            
            Args:
                tool_input: 工具输入，可以是字典、字符串或其他类型
                **kwargs: 额外参数，可能由LangChain框架传入
            """
            # 打印工具名称，验证self.name的存在
            print(f"调用工具: {self.name}")
            
            # 如果tool_input是字典，合并它和kwargs
            if isinstance(tool_input, dict):
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
            
        async def _arun(self, tool_input, **kwargs):
            """
            异步运行工具
            
            Args:
                tool_input: 工具输入，可以是字典、字符串或其他类型
                **kwargs: 额外参数，可能由LangChain框架传入
            """
            # 如果tool_input是字典，合并它和kwargs
            if isinstance(tool_input, dict):
                combined_kwargs = {**tool_input, **kwargs}
                return await self.call_mcp_tool_async(self.name, **combined_kwargs)
            # 如果tool_input是字符串但为空，只传递kwargs
            elif isinstance(tool_input, str) and not tool_input:
                if kwargs:
                    return await self.call_mcp_tool_async(self.name, **kwargs)
                else:
                    return await self.call_mcp_tool_async(self.name)
            # 其他情况，将tool_input作为input参数，并合并kwargs
            else:
                return await self.call_mcp_tool_async(self.name, input=tool_input, **kwargs)
    
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


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python mcp_tools_adapter.py <mcp_server_url>")
        print("例如: python mcp_tools_adapter.py http://127.0.0.1:8083/my-custom-path")
        sys.exit(1)
    
    mcp_server_url = sys.argv[1]
    adapter = MCPToolAdapter(mcp_server_url)
    
    # 获取并打印工具列表
    tools = adapter.get_langchain_tools()
    print(f'tools:\n {tools}')
    print(f"成功从MCP服务器获取了 {len(tools)} 个工具:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # 测试调用工具
    if len(tools) > 0:
        print("\n===== 测试工具调用 =====")
        
        # 测试 get_current_time 工具
        try:
            time_tool = next((t for t in tools if t.name == "get_current_time"), None)
            if time_tool:
                print(f"测试 'get_current_time' 工具...")
                # 直接调用 _run 方法
                result = time_tool._run("")
                print(f"结果: {result}")
        except Exception as e:
            print(f"get_current_time 调用失败: {e}")
            
        # 测试 calculate_bmi 工具
        try:
            bmi_tool = next((t for t in tools if t.name == "calculate_bmi"), None)
            if bmi_tool:
                print(f"\n测试 'calculate_bmi' 工具...")
                # 直接调用 _run 方法
                result = bmi_tool._run({"weight_kg": 70, "height_m": 1.75})
                print(f"结果: {result}")
        except Exception as e:
            print(f"calculate_bmi 调用失败: {e}")
            
        # 测试 get_weather 工具
        try:
            weather_tool = next((t for t in tools if t.name == "get_weather"), None)
            if weather_tool:
                print(f"\n测试 'get_weather' 工具...")
                # 直接调用 _run 方法
                result = weather_tool._run({"city": "Beijing", "date": "今天"})
                print(f"结果: {result[:200]}...")
        except Exception as e:
            print(f"get_weather 调用失败: {e}")