from datetime import datetime
from typing import List, TypedDict


class Info(TypedDict):
    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created: datetime


class Category(TypedDict):
    id: int
    name: str
    supercategory: str
    level: int
    taxonomy_id: str


class Attribute(TypedDict):
    id: int
    name: str
    supercategory: str
    level: int
    taxonomy_id: str


class Image(TypedDict):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    time_captured: str
    original_url: str
    isstatic: int
    kaggle_id: str


class Annotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    attribute_ids: List[int]
    segmentation: List[List[int]]
    bbox: List[int]
    area: int
    iscrowd: int


class License(TypedDict):
    id: int
    name: str
    url: str


class FashionPedia(TypedDict):
    info: Info
    categories: List[Category]
    attributes: List[Attribute]
    images: List[Image]
    annotations: List[Annotation]
    licenses: List[License]
