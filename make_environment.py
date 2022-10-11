import subprocess
import os
from os import path

main_dir = os.getcwd()
env_path = main_dir + '/Envir'
user_path = os.path.expanduser('~')
py_path_single_user = user_path + "/AppData/Local/Programs/Python/Python39/python.exe"
py_path_all_user = user_path.split('\\')[0] + "/Program Files/Python39/python.exe"
my_path = "D:/Program Files/Python39/python.exe"

def env_activate(_env_path):
    subprocess.call([_env_path + "/Scripts/activate"])

def make_env_cmd(_py_path, _env_path):
    if len(os.listdir(_env_path)) == 0:
        subprocess.run([_py_path, "-m", "venv", _env_path])
    else:
        print("Environment already exists: %s" % _env_path)
        
def make_env():
    if path.exists(py_path_single_user):
        make_env_cmd(py_path_single_user, env_path)
    elif path.exists(py_path_all_user):
        make_env_cmd(py_path_all_user, env_path)
    elif path.exists(my_path):
        make_env_cmd(my_path, env_path)
        
if not path.exists(env_path):
    os.mkdir(env_path)
    make_env()
else:
    print("Envir folder already exists at %s" % env_path, "Delete it if you want to create new environments")
    make_env()