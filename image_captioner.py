from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "List all the objects that can be segmented in one word each"
image_file = "/home/sumanth/ResearchIntern/astronaut.png"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature":0.2,
    "top_p": None,
    "num_beams":1,
    "max_new_tokens":512
})()
