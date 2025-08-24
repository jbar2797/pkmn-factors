from fastapi import APIRouter

router = APIRouter()


@router.get("/example")
def example():
    return {"ok": True, "message": "demo endpoint works"}
