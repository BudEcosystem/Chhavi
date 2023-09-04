from argparse import ArgumentParser

import gradio as gr

from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc


model_id = "budecosystem/Chhavi"

# Generate how many images by default
default_num_images = max(1, int(os.getenv("NUM_IMAGES_PER_PROMPT", "4")))


print(
    f"[INFO] Loading model from {model_id}",
)
pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True
)
pipe.to("cuda")


def infer(prompt, negative, scale, samples=default_num_images, steps=50, seed=-1):
    prompt, negative = [prompt] * samples, [negative] * samples

    g = torch.Generator(device="cuda")
    if seed != -1:
        g.manual_seed(seed)
    else:
        g.seed()

    images_b64_list = []

    images = pipe(
        prompt=prompt,
        negative_prompt=negative,
        guidance_scale=scale,
        num_inference_steps=steps,
        # num_images_per_prompt=samples,
        generator=g,
    ).images

    gc.collect()
    torch.cuda.empty_cache()

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        image_b64 = f"data:image/jpeg;base64,{img_str}"
        images_b64_list.append(image_b64)

    return images_b64_list


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(
    css=css,
    title="Chhavi",
    description="A Latent Diffusion Model (LDM)",
)

negative_prompt = "tiling, out of frame, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur"

examples = [
    [
        "Vector anime chibi style logo featuring a single piece of flaming piece of popcorn with a smiling face, with mirrorshades sunglasses, popcorn as morpheus, clean composition, symmetrical",
        negative_prompt,
        7.5,
    ],
    [
        "Clean, sharp, vectorized logo of an AI organization named Bud. Company logo, icon, trending, modern and minimalist",
        negative_prompt,
        7.5,
    ],
    [
        "Minimalistic mountains design logo from word parlatur, experiential tourism, banksy, bold font, black font, nature, travel, sharp, white background, illustration",
        negative_prompt,
        7.5,
    ],
    ["a modern logo with kingfisher", negative_prompt, 7.5],
    [
        "Coffee logo, featuring a mushroom cloud coming out of a cup, the cloud looks like brains, by mcbess, full colour print, vintage colours, 1960s",
        negative_prompt,
        7.5,
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Chhavi Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                Chhavi is a Latent Diffusion Model (LDM) that is first in the release cycle of Bud Ecosystem's finetune series of Diffusion Models.
                <a style="text-decoration: underline;" href="https://huggingface.co/budecosystem/Chhavi">Access Chhavi Model</a>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(equal_height=True):
                with gr.Column(variant="box"):
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        container=False,
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        container=False,
                    )
                btn = gr.Button("Generate image").style(
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(columns=[4], height="auto")

        with gr.Accordion("Advanced settings", open=False):
            #    gr.Markdown("Advanced settings are temporarily unavailable")
            samples = gr.Slider(
                label="Images",
                minimum=1,
                maximum=max(4, default_num_images),
                value=default_num_images,
                step=1,
            )
            steps = gr.Slider(label="Steps", minimum=1, maximum=250, value=50, step=1)
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=9, step=0.1
            )

            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[text, negative, guidance_scale],
            outputs=[gallery],
            cache_examples=False,
        )
        ex.dataset.headers = [""]
        negative.submit(
            infer,
            inputs=[
                text,
                negative,
                guidance_scale,
                samples,
                steps,
                seed,
            ],
            outputs=[gallery],
            postprocess=False,
        )
        text.submit(
            infer,
            inputs=[
                text,
                negative,
                guidance_scale,
                samples,
                steps,
                seed,
            ],
            outputs=[gallery],
            postprocess=False,
        )
        btn.click(
            infer,
            inputs=[
                text,
                negative,
                guidance_scale,
                samples,
                steps,
                seed,
            ],
            outputs=[gallery],
            postprocess=False,
        )

        gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/budecosystem" style="text-decoration: underline;" target="_blank">BudEcosystem</a> - Gradio Demo by 🤗 Hugging Face and <a style="text-decoration: underline;" href="https://github.com/BudEcosystem">BudEcosystem</a>
                    </p>
                </div>
           """
        )
        with gr.Accordion(label="License", open=False):
            gr.HTML(
                """<div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/budecosystem/Chhavi/blob/main/LICENSE.md" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL++-M License</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
               </div>
                """
            )

block.queue().launch(server_name="0.0.0.0", share=True)
