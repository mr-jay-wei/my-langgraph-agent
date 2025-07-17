# conda env mcp_env ,Python版本 3.10.18
# 关闭 proxy
# 使用方法：用python 把server启动

from datetime import datetime
import requests
from fastmcp import FastMCP
mcp = FastMCP(debug = True)

@mcp.tool()
def calculate_bmi(weight_kg:float,height_m:float)->float:
	"""通过给定的体重和身高计算BMI指数。

    Args:
        weight_kg (float): 用户的体重，单位为公斤(kg)。
        height_m (float): 用户的身高，单位为米(m)。

    Returns:
        float: 计算得出的BMI指数值。
    """
	return weight_kg / (height_m ** 2)

@mcp.tool()
def get_current_time() -> str:
    """获取当前时间

    Returns:
        str: 当前时间的字符串表示
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def get_weather(city: str, date: str):
    """获取指定城市在特定日期的天气情况。

    Args:
        city (str): 需要查询天气的城市名称，例如 "北京" 或 "London"。
        date (str): 需要查询的日期，可以使用 "今天"、"明天" 等相对描述，或 "2023.10.27" 这样的具体日期。

    Returns:
        str: 包含该城市天气情况描述的原始文本。
    """
    endpoint = "https://wttr.in"

    response = requests.get(f"{endpoint}/{city}")
    return response.text

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8083,
        path="/my-custom-path",
        log_level="debug",
    )