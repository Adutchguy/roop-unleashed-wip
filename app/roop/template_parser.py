import re
from datetime import datetime

template_functions = {
    "timestamp": lambda data: str(int(datetime.now().timestamp())),
    "i": lambda data: data.get("index", False),
    "file": lambda data: data.get("file", False),
    "date": lambda data: datetime.now().strftime("%Y-%m-%d"),
    "time": lambda data: datetime.now().strftime("%H-%M-%S"),
}


def parse(text: str, data: dict):
    pattern = r"\{([^}]+)\}"

    matches = re.findall(pattern, text)

    for match in matches:
        replacement = template_functions[match](data)
        if replacement is not False:
            text = text.replace(f"{{{match}}}", replacement)

    return text
