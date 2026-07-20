from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    session_id: str = Field(min_length=1)


class CornersRequest(DetectRequest):
    corners: list[list[float]]


class EnhanceRequest(DetectRequest):
    mode: str = Field(pattern="^(original|color|grayscale|black_white|high_contrast|magic_color|auto)$")
