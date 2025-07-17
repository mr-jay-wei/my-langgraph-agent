from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="User Info API",
    description="一个简单的 API，用于根据用户ID查询模拟的用户信息。",
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
def get_user_by_id(user_id: str):
    """
    根据用户ID获取用户信息。
    """
    if user_id in fake_user_db:
        return fake_user_db[user_id]
    raise HTTPException(status_code=404, detail="User not found")

if __name__ == "__main__":
    # 运行这个服务。它会监听在 http://127.0.0.1:8000
    uvicorn.run("user_api:app", host="127.0.0.1", port=8000, reload=True)