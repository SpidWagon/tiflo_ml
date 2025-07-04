from pydantic import BaseModel

class Req(BaseModel):
    image: str
    timestamp: str


class Result(BaseModel):
    comment: str
    timestamp: str
