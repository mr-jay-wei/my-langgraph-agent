# conda env mcp_env ,Python版本 3.10.18
# 关闭 proxy
# 使用方法：用python 把server启动

import asyncio
from datetime import datetime
import aiohttp
from fastmcp import FastMCP
mcp = FastMCP(debug = True)

@mcp.tool()
async def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """通过给定的体重和身高计算BMI指数。

    Args:
        weight_kg (float): 用户的体重，单位为公斤(kg)。
        height_m (float): 用户的身高，单位为米(m)。

    Returns:
        float: 计算得出的BMI指数值。
    """
    # 模拟一些异步处理
    await asyncio.sleep(0.01)  # 模拟轻微延迟
    return weight_kg / (height_m ** 2)

@mcp.tool()
async def get_current_time() -> str:
    """获取当前时间

    Returns:
        str: 当前时间的字符串表示
    """
    # 模拟一些异步处理
    await asyncio.sleep(0.01)  # 模拟轻微延迟
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
async def get_weather(city: str, date: str):
    """获取指定城市在特定日期的天气情况。

    Args:
        city (str): 需要查询天气的城市名称，例如 "北京" 或 "London"。
        date (str): 需要查询的日期，可以使用 "今天"、"明天" 等相对描述，或 "2023.10.27" 这样的具体日期。

    Returns:
        str: 包含该城市天气情况描述的原始文本。
    """
    endpoint = "https://wttr.in"

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{endpoint}/{city}") as response:
            return await response.text()

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8084,  # 使用不同的端口，避免与同步版本冲突
        path="/my-custom-path",
        log_level="debug",
    )