from fastapi import FastAPI, HTTPException, status, Response, Query
from typing import List, Optional
from app.models import Task, TaskCreate, TaskUpdate
import app.database as db

app = FastAPI(title="CRUD Task API", description="Structured In-memory REST API")

# Task 1: GET /tasks (list all with ?priority= filter)
@app.get("/tasks", response_model=List[Task])
def get_tasks(priority: Optional[str] = Query(None, description="Filter tasks by priority")):
    tasks = db.get_all_tasks()
    if priority:
        return [t for t in tasks if t["priority"].lower() == priority.lower()]
    return tasks

# Task 1: GET /tasks/{id}
@app.get("/tasks/{id}", response_model=Task)
def get_task_by_id(id: int):
    task = db.get_task(id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID {id} not found")
    return task

# Task 2: POST /tasks
@app.post("/tasks", response_model=Task, status_code=status.HTTP_201_CREATED)
def create_task(task_input: TaskCreate):
    return db.add_task(task_input.title, task_input.status, task_input.priority)

# Task 3: PUT /tasks/{id}
@app.put("/tasks/{id}", response_model=Task)
def update_task(id: int, task_update: TaskUpdate):
    update_data = task_update.dict(exclude_unset=True)
    updated_task = db.update_task_data(id, update_data)
    if not updated_task:
        raise HTTPException(status_code=404, detail=f"Task with ID {id} not found")
    return updated_task

# Task 4: DELETE /tasks/{id}
@app.delete("/tasks/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(id: int):
    success = db.delete_task_data(id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task with ID {id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# Task 5: GET /tasks/stats
@app.get("/tasks/stats")
def get_task_stats():
    stats = {}
    for task in db.get_all_tasks():
        status_name = task["status"]
        stats[status_name] = stats.get(status_name, 0) + 1
    return stats