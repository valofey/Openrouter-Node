import base64
import json
import requests
from PIL import Image
import io
import torch
import re
from torchvision.transforms import ToPILImage

class OpenrouterNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "multiline": False,
                    "default": "https://openrouter.ai/api/v1/chat/completions"
                }),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "gpt4o"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_round": 2
                }),
                "trim_think": ("BOOLEAN", {
                    "default": True,
                }),
            },
            "optional": {
                "image_input": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_completion"
    CATEGORY = "OpenRouter"

    def remove_think_tags(self, text):
        # Pattern to match <think>...</think> tags and their content
        pattern = r'<think>.*?</think>'
        # Remove the tags and their content using regex
        return re.sub(pattern, '', text, flags=re.DOTALL)

    def get_completion(self, base_url, model, api_key, prompt, system_prompt, temperature, trim_think, image_input=None):
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            # Initialize messages list
            messages = []
            
            # Add system message if system_prompt is provided and not empty
            if system_prompt and system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })

            # Add user message with content as a list
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }

            if image_input is not None:
                if isinstance(image_input, torch.Tensor):
                    # Handle 4D tensors by removing the batch dimension
                    if image_input.dim() == 4:
                        image_input = image_input.squeeze(0)
                    if image_input.dim() != 3:
                        return ("Error: Image tensor must be 3D after squeezing.",)
                    # Rearrange tensor from [H, W, C] to [C, H, W] if necessary
                    if image_input.shape[-1] in [1, 3, 4]:
                        image_input = image_input.permute(2, 0, 1)
                    else:
                        return (f"Error: Invalid number of channels: {image_input.shape[-1]}.",)
                    to_pil = ToPILImage()
                    pil_image = to_pil(image_input)
                elif isinstance(image_input, Image.Image):
                    pil_image = image_input
                else:
                    return ("Error: Unsupported image_input type.",)

                # Save the image without resizing or cropping
                buffered = io.BytesIO()
                # Handle different Pillow versions for resampling filters
                resample_filter = getattr(Image, "Resampling", None)
                if resample_filter and hasattr(resample_filter, "LANCZOS"):
                    pil_image.save(buffered, format="PNG", resample=Image.Resampling.LANCZOS)
                else:
                    pil_image.save(buffered, format="PNG", resample=Image.LANCZOS)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_data = f"data:image/png;base64,{img_str}"

                image_message = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
                
                # Append the image message to the content list
                user_message["content"].append(image_message)

            # Add the user message to messages list
            messages.append(user_message)

            body = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }

            response = requests.post(base_url, headers=headers, data=json.dumps(body), timeout=120)
            response.raise_for_status()

            response_json = response.json()

            if "choices" in response_json and len(response_json["choices"]) > 0:
                assistant_message = response_json["choices"][0].get("message", {}).get("content", "")
                
                # If trim_think is enabled, remove think tags
                if trim_think:
                    assistant_message = self.remove_think_tags(assistant_message)
                
                return (assistant_message,)
            else:
                return ("No response from the model.",)

        except requests.exceptions.RequestException as req_err:
            return (f"Request Error: {str(req_err)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "OpenrouterNode": OpenrouterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenrouterNode": "OpenRouter Node"
}