from fastapi import FastAPI, Body, Query, UploadFile, File
from typing import Union, List
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import io
import tempfile
app = FastAPI()


class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


def validate(func, *args, **kwargs):
    async def wrapper():
        print(args)
        print(kwargs)
        await func(kwargs)
        print("Hecho")
    return wrapper

@app.get("/items/")
async def read_item(skip:int = 0, limit: int =10, opt: Union[int, None] = None):
    return f"skip {skip} limit {limit} opt {opt}" 

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.get("/{id}/")
async def root(id: int):
    return {"message": id}

@app.post("/test/body")
async def test_body(id: int = Body(), other: dict = Body()):
    return f"{id} + {other['papa']}" 

@app.post("/items/")
async def create_item1(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

@app.put("/items/{item_id}")
async def create_item2(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

@app.put("/items/{item_id}")
async def create_item3(item_id: int, item: Item, q: Union[int, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result


@app.get("/items/foo")
async def read_items(q: Union[List[str], None] = Query(default=None)):
    query_items = {"q": q}
    return query_items

@app.post("/upload-file")
async def upload_file(file: UploadFile= File(...)):
    model_file = await file.read()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(model_file)
        model = tf.keras.models.load_model(tmp.name)
        print(model.summary())
    return file.filename

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)