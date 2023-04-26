from diffusers import DiffusionPipeline
from PIL import Image
import os

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

prompt = "a dog riding a bicycle , high quality"

# First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
image = pipe(prompt).images[0]

# Save the image to a file
output_folder = "output"
output_filename = "output3.jpg"
output_path = os.path.join(output_folder, output_filename)

image.save(output_path)

print(f"Image saved to {output_path}")