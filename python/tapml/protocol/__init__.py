"""The protocols for TapML server"""
from . import openai_api_protocol

RequestProtocol = openai_api_protocol.CompletionRequest
