"""
Plugin system for extensible network and community detection methods
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, List, Any
from functools import wraps
import inspect


class MethodRegistry:
    """registry for network and community detection methods"""
    
    def __init__(self):
        self.network_methods = {}
        self.community_methods = {}
        self.default_params = {}
    
    def register_network_method(self, name: str, default_params: Dict = None):
        """decorator to register network adjacency methods"""
        def decorator(func):
            self.network_methods[name] = func
            if default_params:
                self.default_params[name] = default_params
            return func
        return decorator
    
    def register_community_method(self, name: str, default_params: List[Dict] = None):
        """decorator to register community detection methods"""
        def decorator(func):
            self.community_methods[name] = func
            if default_params:
                self.default_params[name] = default_params
            return func
        return decorator
    
    def get_network_methods(self) -> List[str]:
        return list(self.network_methods.keys())
    
    def get_community_methods(self) -> List[str]:
        return list(self.community_methods.keys())
    
    def get_network_method(self, name: str) -> Callable:
        if name not in self.network_methods:
            raise ValueError(f"network method '{name}' not registered")
        return self.network_methods[name]
    
    def get_community_method(self, name: str) -> Callable:
        if name not in self.community_methods:
            raise ValueError(f"community method '{name}' not registered")
        return self.community_methods[name]
    
    def get_default_params(self, method_name: str) -> Any:
        return self.default_params.get(method_name, [{}])


# global registry instance
registry = MethodRegistry()


# decorators for easy method registration
def network_method(name: str, default_params: Dict = None):
    """decorator to register a network adjacency method"""
    return registry.register_network_method(name, default_params)


def community_method(name: str, default_params: List[Dict] = None):
    """decorator to register a community detection method"""
    return registry.register_community_method(name, default_params)


# helper functions to validate method signatures
def validate_network_method(func):
    """validate that network method has correct signature"""
    sig = inspect.signature(func)
    expected_params = ['data']
    
    if list(sig.parameters.keys()) != expected_params:
        raise ValueError(f"network method must have signature: {expected_params}")


def validate_community_method(func):
    """validate that community method has correct signature"""
    sig = inspect.signature(func)
    expected_params = ['adj_matrix', 'params']
    
    if list(sig.parameters.keys()) != expected_params:
        raise ValueError(f"community method must have signature: {expected_params}") 