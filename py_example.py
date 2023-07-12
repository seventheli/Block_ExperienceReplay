import torch
import json
from utils import check_path

check_path("/tmp/")
with open ("/tmp/test.json", "w") as f:
    json.dump({"test": torch.cuda.is_available()}, f)
print(torch.tensor([123,]).cuda())