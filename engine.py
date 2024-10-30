from sqlalchemy import create_engine
import os
from flask import current_app

class ExcelEngine:
    _instances = {}

    def __new__(cls, name):
        if name not in cls._instances:
            instance = super(ExcelEngine, cls).__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name):
        # Initialize only if engine is not already created
        if not hasattr(self, "engine"):
            self.engine = create_engine(f'sqlite:///db/{name}.db')

    def getEngine(self):
        return self.engine

def setup_engine(file_path):
    current_app.config['engines'] = ExcelEngine(os.path.basename(file_path).split('.')[0])

def get_engine():
    if 'engines' not in current_app.config:
        raise ValueError("Engine not setup")
    return current_app.config['engines']