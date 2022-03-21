import re


def regex_decision(rep, str):
    return re.search(rep, str) is not None