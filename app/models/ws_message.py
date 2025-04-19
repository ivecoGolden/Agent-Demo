from pydantic import BaseModel


class StreamMessage(BaseModel):
    uuid: str
    content: str
    status: str
