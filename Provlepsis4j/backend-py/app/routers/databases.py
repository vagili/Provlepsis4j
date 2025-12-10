# app/routers/databases.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..db import list_databases, set_database, current_database, run, run_data

router = APIRouter()

# ---------- request model ----------
class DropDbBody(BaseModel):
    name: str

class UseDbBody(BaseModel):
    name: str

@router.get("/databases")
def get_databases():
    rows = list_databases()
    return {"current": current_database(), "databases": rows}

@router.post("/databases/use")
def use_database(body: UseDbBody):
    rows = list_databases()
    names = {r["name"] for r in rows}
    if body.name not in names:
        raise HTTPException(status_code=404, detail=f"Database '{body.name}' not found.")
    status = next(r["currentStatus"] for r in rows if r["name"] == body.name)
    if (status or "").lower() != "online":
        raise HTTPException(status_code=409, detail=f"Database '{body.name}' is not online (status={status}).")
    set_database(body.name)
    return {"ok": True, "current": body.name}

@router.post("/databases/drop")
def drop_database(body: DropDbBody):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Database name required.")
    if name.lower() in {"system", "neo4j"}:
        raise HTTPException(status_code=400, detail=f'Cannot drop reserved database "{name}".')

    # Get the current database name (works across Neo4j 5.x)
    try:
        cur_rows = run_data("CALL db.info() YIELD name RETURN name")
        current = cur_rows[0]["name"] if cur_rows else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to determine current database: {e}")

    if current == name:
        raise HTTPException(
            status_code=400,
            detail=f'Cannot drop the currently selected database "{name}". Switch to a different DB first.'
        )

    # Ensure the database exists (portable way)
    rows = run_data(
        "SHOW DATABASES YIELD name WHERE name = $db RETURN name",
        {"db": name},
        db="system",
    )
    if not rows:
        return {"ok": True, "dropped": False, "message": f'Database "{name}" not found.'}

    try:
        run("DROP DATABASE $db IF EXISTS", {"db": name}, db="system")
        return {"ok": True, "dropped": True, "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to drop database '{name}': {e}")
