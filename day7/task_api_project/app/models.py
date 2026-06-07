from pydantic import BaseModel, Field
from typing import Optional

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, description="The title of the task")
    status: str = Field(default="pending", description="Task status (e.g., pending, completed)")
    priority: str = Field(default="medium", description="Task priority (low, medium, high)")

class Task(BaseModel):
    id: int
    title: str
    status: str
    priority: str

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None