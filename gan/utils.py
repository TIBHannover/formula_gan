import logging


def unflat_dict(data_dict):
    result_map = {}
    for k, v in data_dict.items():
        path = k.split(".")
        prev = result_map
        for p in path[:-1]:
            if p not in prev:
                prev[p] = {}
            prev = prev[p]
        prev[path[-1]] = v
    return result_map


def flat_dict(data_dict):
    result_map = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            embedded = flat_dict(v)
            for s_k, s_v in embedded.items():
                s_k = f"{k}.{s_k}"
                if s_k in result_map:
                    logging.error(f"flat_dict: {s_k} alread exist in output dict")

                result_map[s_k] = s_v
            continue

        if k not in result_map:
            result_map[k] = []
        result_map[k] = v
    return result_map


def get_element(data_dict: dict, path: str, split_element="."):
    elem = data_dict
    try:
        for x in path.strip(split_element).split(split_element):
            try:
                x = int(x)
                elem = elem[x]
            except ValueError:
                elem = elem.get(x)
    except:
        pass

    return elem
