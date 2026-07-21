from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    session_id: str = Field(min_length=1)


class CornersRequest(DetectRequest):
    corners: list[list[float]]


class EnhanceRequest(DetectRequest):
    mode: str = Field(default="black_white", pattern="^(black_white|bw_clean)$")
