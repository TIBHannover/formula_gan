import torch


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


def concat_batch(samples):
    result_map = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("DataLoaderWrapper only support dict datasets")

        flatted_dict = flat_dict(sample)

        for k, v in flatted_dict.items():
            if k not in result_map:
                result_map[k] = []
            result_map[k].append(v)

    final_result = {}
    for k, v in result_map.items():

        try:
            final_result[k] = torch.cat(v, dim=0)
        except AttributeError:
            final_result[k] = [x for n in v for x in n]
        except TypeError:
            final_result[k] = [x for n in v for x in n]

    return unflat_dict(final_result)


class DataLoaderWrapper:
    def __init__(self, datasets, batch_sizes, complete_all=False, **kwargs):
        self.data_loaders = [
            torch.utils.data.DataLoader(datasets[i], batch_size=batch_sizes[i], **kwargs) for i in range(len(datasets))
        ]
        self.complete_all = complete_all

    def __iter__(self):
        iters = [iter(x) for x in self.data_loaders]
        all_done = [False] * len(iters)
        while not all(all_done):
            # get samples from all iters
            samples = []
            for i, it in enumerate(iters):
                try:
                    s = next(it)
                except StopIteration as e:
                    if not self.complete_all:
                        return
                    all_done[i] = True
                    if all(all_done):
                        return
                    iters[i] = iter(self.data_loaders[i])

                    try:
                        s = next(iters[i])
                    except StopIteration as e:
                        return None
                samples.append(s)

            final_result = concat_batch(samples)

            yield final_result
