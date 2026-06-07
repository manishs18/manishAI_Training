from typing import Dict

# In-Memory Database Simulation
tasks_db: Dict[int, dict] = {}
_current_id = 0

def get_all_tasks():
    return list(tasks_db.values())

def get_task(task_id: int):
    return tasks_db.get(task_id)

def add_task(title: str, status: str, priority: str):
    global _current_id
    _current_id += 1
    new_task = {
        "id": _current_id,
        "title": title,
        "status": status,
        "priority": priority
    }
    tasks_db[_current_id] = new_task
    return new_task

def update_task_data(task_id: int, update_data: dict):
    if task_id in tasks_db:
        tasks_db[task_id].update(update_data)
        return tasks_db[task_id]
    return None

def delete_task_data(task_id: int):
    if task_id in tasks_db:
        del tasks_db[task_id]
        return True
    return False