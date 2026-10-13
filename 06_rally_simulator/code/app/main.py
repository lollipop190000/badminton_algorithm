from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ExampleDetail, ExampleSummary, RallyRequest, RallyResponse
from app.simulator import RallySimulator


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.simulator = RallySimulator()
        app.state.load_error = None
    except Exception as exc:  # keep health endpoint available when model files are missing
        app.state.simulator = None
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="GRU·CQL 배드민턴 랠리 시뮬레이터",
    description="로컬 PyTorch 체크포인트를 사용하는 연구팀용 배드민턴 랠리 분석 도구",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


def get_simulator(request: Request) -> RallySimulator:
    simulator = request.app.state.simulator
    if simulator is None:
        raise HTTPException(
            status_code=503,
            detail=f"모델을 불러오지 못했습니다: {request.app.state.load_error}",
        )
    return simulator


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    messages = []
    for error in exc.errors():
        location = ".".join(str(value) for value in error.get("loc", []) if value != "body")
        message = str(error.get("msg", "입력값이 올바르지 않습니다."))
        messages.append(f"{location}: {message}" if location else message)
    return JSONResponse(
        status_code=422,
        content={"detail": "랠리 입력을 확인해 주세요.", "errors": messages},
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
async def health(request: Request) -> JSONResponse:
    if request.app.state.simulator is None:
        return JSONResponse(
            status_code=503,
            content={"loaded": False, "error": request.app.state.load_error},
        )
    return JSONResponse(content=request.app.state.simulator.health)


@app.get("/api/examples", response_model=list[ExampleSummary])
async def examples(
    request: Request,
    query: str = Query(default="", max_length=100),
    limit: int = Query(default=500, ge=1, le=1000),
) -> list[ExampleSummary]:
    return get_simulator(request).list_examples(query=query, limit=limit)


@app.get("/api/examples/{rally_id}", response_model=ExampleDetail)
async def example(rally_id: str, request: Request) -> ExampleDetail:
    try:
        return get_simulator(request).get_example(rally_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="해당 예시 랠리를 찾을 수 없습니다.") from exc


@app.post("/api/analyze", response_model=RallyResponse)
async def analyze(payload: RallyRequest, request: Request) -> RallyResponse:
    return get_simulator(request).analyze(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

