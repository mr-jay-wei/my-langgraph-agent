from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
import time

app = FastAPI(
    title="User Info API (Async)",
    description="一个简单的异步API，用于根据用户ID查询模拟的用户信息。",
    version="1.0.0",
)

# 模拟的数据库
fake_user_db = {
    "user_101": {"username": "Alice", "email": "alice@example.com", "membership_level": "Gold"},
    "user_102": {"username": "Bob", "email": "bob@example.com", "membership_level": "Silver"},
    "user_103": {"username": "Charlie", "email": "charlie@example.com", "membership_level": "Bronze"},
}

class UserInfo(BaseModel):
    username: str
    email: str
    membership_level: str

@app.get("/users/{user_id}", response_model=UserInfo, tags=["Users"])
async def get_user_by_id(user_id: str):
    """
    根据用户ID获取用户信息。
    """
    # 模拟异步数据库查询延迟
    await asyncio.sleep(0.05)
    
    if user_id in fake_user_db:
        return fake_user_db[user_id]
    raise HTTPException(status_code=404, detail="User not found")

@app.get("/performance-test", tags=["Performance"])
async def performance_test():
    """
    性能测试端点，模拟多个并发请求。
    """
    start_time = time.time()
    
    # 创建多个并发任务
    tasks = []
    for i in range(10):
        user_id = f"user_10{i % 3 + 1}"  # 循环使用user_101, user_102, user_103
        tasks.append(get_user_by_id(user_id))
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        "execution_time": execution_time,
        "requests_count": len(tasks),
        "average_time_per_request": execution_time / len(tasks),
        "results": results
    }

if __name__ == "__main__":
    # 运行这个服务。它会监听在 http://127.0.0.1:8001
    uvicorn.run("user_api_async:app", host="127.0.0.1", port=8001, reload=True)