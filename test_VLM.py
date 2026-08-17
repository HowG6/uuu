import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

IMAGE_PATH = "/home/yyan-admin/Documents/udmc_carla/test.jpg"


processor = AutoProcessor.from_pretrained(MODEL_ID)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"file://{IMAGE_PATH}",
            },
            {
                "type": "text",
                "text": (
                    "Analyze this autonomous driving scene. "
                    "Identify the ego lane, nearby vehicles, "
                    "traffic lights, possible lane changes, "
                    "and collision risks. "
                    "Then recommend an appropriate driving action."
                ),
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 这里只处理图片，不获取 video_kwargs
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
    )

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids
    in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("\n===== Qwen3-VL Result =====\n")
print(output_text[0])


# import torch
# from transformers import AutoModelForImageTextToText, AutoProcessor
# from qwen_vl_utils import process_vision_info

# MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# processor = AutoProcessor.from_pretrained(MODEL_ID)

# model = AutoModelForImageTextToText.from_pretrained(
#     MODEL_ID,
#     dtype=torch.bfloat16,
#     device_map="auto",
# )

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "file:///home/yyan-admin/Documents/udmc_carla/test.jpg",
#             },
#             {
#                 "type": "text",
#                 "text": (
#                     "Analyze this autonomous driving scene. "
#                     "Identify the ego lane, nearby vehicles, "
#                     "possible lane changes, and collision risks."
#                 ),
#             },
#         ],
#     }
# ]

# text = processor.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )

# images, videos, video_kwargs = process_vision_info(
#     messages,
#     return_video_kwargs=True,
# )

# inputs = processor(
#     text=[text],
#     images=images,
#     videos=videos,
#     padding=True,
#     return_tensors="pt",
#     **video_kwargs,
# ).to(model.device)

# with torch.no_grad():
#     output_ids = model.generate(
#         **inputs,
#         max_new_tokens=256,
#     )

# generated_ids = [
#     output[len(input_ids):]
#     for input_ids, output in zip(inputs.input_ids, output_ids)
# ]

# result = processor.batch_decode(
#     generated_ids,
#     skip_special_tokens=True,
#     clean_up_tokenization_spaces=False,
# )

# print(result[0])