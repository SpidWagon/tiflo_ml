from pydantic import BaseModel


class Req(BaseModel):
    images: list


class Result(BaseModel):
    comments: list
