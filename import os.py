import os
print(f"Working dir: {os.getcwd()}")
print(f"Files in dir: {os.listdir()}")
print(f"Config exists: {os.path.isfile('config.yaml')}")