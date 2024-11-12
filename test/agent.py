import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from .test_mathvista.answer import MathVistaAnswer


class Agent:
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    def run(self, batch):
        data = batch.get_task_batch()
        questions, images = data[0], data[1]
        
        conversations = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": questions[i]},
                ],
            } for i in range(len(questions))
        ]
        
        text_prompt = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
        inputs = self.processor(
            text=text_prompt, images=images, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device).to(torch.bfloat16)
        
        output_ids = self.model.generate(**inputs, max_new_tokens=10)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return [[MathVistaAnswer(output_text[i])] for i in range(len(output_text))]
        