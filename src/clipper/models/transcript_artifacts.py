"""
Transcript artifacts for clipper pipeline.
"""


from pydantic import BaseModel

from . import Sentence, Word


class WordsArtifact(BaseModel):
    schema_version: int = 1
    words: list[Word]


class SentencesArtifact(BaseModel):
    schema_version: int = 1
    sentences: list[Sentence]
