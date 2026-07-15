import json
from pathlib import Path


def load_data(response_path: str, source_info_path: str):
    """Load response and source info data from JSONL files."""
    response = []
    with open(response_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            response.append(data)

    source_info_dict = {}
    with open(source_info_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            source_info_dict[data['source_id']] = data

    return response, source_info_dict


response_path = r"/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/response_spans.jsonl"
source_info_path = r"/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/source_info_spans.jsonl"
new_path = Path(r"/mnt/internal/sata-ssd/GitHub/SteffenLuminaETC/ReDeEP/dataset/")

responses, source_info = load_data(response_path, source_info_path)

dc = {}
for i in responses:
    source_id = i["source_id"]
    source_inf = source_info[source_id]
    model = i["model"]
    dic = dc.setdefault(model, {})
    val = {**i, **source_inf}
    del val["id"]
    dic[source_id] = val

for k, v in dc.items():
    o_p = new_path / Path(f"response_span_{k}.json")
    with o_p.open("w") as f:
        json.dump(v, f)
