import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import replace_string_json

print(sys.path)

#from ..utils.utils import replace_string_json



json_file = {
    "field1": "Hola [1] que tal estas?",
    "field2": "[1] es una [2] [3]"
}

json_correct = {
    "field1": "Hola Mario que tal estas?",
    "field2": "Mario es una persona de mierda"
}


json_modified = replace_string_json(json_file, ("Mario", "persona", "de mierda"))
def foo (*args):
    print(type(args))

if json_modified == json_correct:
    print("Ok")
else:
    print("MAL")
    print(json_modified)