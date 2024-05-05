import modules.scripts as scripts
import gradio as gr
import os
import re
import torch
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm  
from modules import script_callbacks
import torchvision.transforms as transforms
#from lavis.models import load_model_and_preprocess
from transformers import BlipProcessor, BlipForConditionalGeneration

global_model = None
global_vis_processors = None

#### Promptの処理 ####
def prompt_to_caption(folder_path, check=False):
    print("Folder Path:", folder_path)
    print("Enable BLIP:", check)
    # BLIPの初期化
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files, desc='Processing images', unit='image'):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)

        # メタデータを取得
        metadata = image.info
        parameters_info = metadata.get('parameters', '').split('Negative prompt:')[0]
        
        # メタデータが存在しない場合BLIPで生成
        if not parameters_info and check:
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            parameters_info = processor.decode(out[0], skip_special_tokens=True)

        # lora情報削除
        prompt = re.sub(r'<lora:.*?>', '', parameters_info)

        text_file_path = os.path.splitext(image_path)[0] + '.txt'

        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

    print("All caption files have been created.")
    return "All caption files have been created."

#### BLIPの処理 ####
def caption_with_blip(selection, folder_path=None, image=None):
    # BLIPの初期化
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    if selection == "Single Image" and image is not None:
        # 画像一枚に対してキャプション生成
        img = Image.fromarray(image).convert("RGB")
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    if selection == "Batch process" and folder_path:
        # フォルダ内の画像ファイルを取得
        image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            img = Image.open(image_path)

            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            text_file_path = os.path.splitext(image_path)[0] + '.txt'
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(caption)

        return "All caption files have been created."
    
    return "Please provide either a folder or an image."

# #### BLIP2の処理 #####
# def process_image(img_path, device):
#     raw_image = Image.open(img_path).convert('RGB')
#     image = global_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
#     print(f"Image tensor shape before model processing: {image.shape}")

#     try:
#         result = global_model.generate({"image": image})
#     except RuntimeError as e:
#         print("Error during model generation:", str(e))
#         # Additional debug output to help trace the tensor shapes
#         if hasattr(global_model, 'debug_last_input_shapes'):
#             print("Debug info about last input shapes:", global_model.debug_last_input_shapes())
#         raise e  # Re-raise the exception to keep the error visible
#     return result


# def save_caption(caption, img_path):
#     # save caption file
#     caption_file = os.path.splitext(img_path)[0] + '.txt'
#     with open(caption_file, 'w') as f:
#         f.write('\n'.join(map(str, caption)))

# def load_model(model_choice):
#     global global_model, global_vis_processors
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model_mapping = {
#         "pretrain_opt2.7b": ("blip2_opt", "pretrain_opt2.7b"),
#         "pretrain_opt6.7b": ("blip2_opt", "pretrain_opt6.7b"),
#         "caption_coco_opt2.7b": ("blip2_opt", "caption_coco_opt2.7b"),
#         "caption_coco_opt6.7b": ("blip2_opt", "caption_coco_opt6.7b"),
#         "pretrain_flant5xl": ("blip2_t5", "pretrain_flant5xl"),
#         "caption_coco_flant5xl": ("blip2_t5", "caption_coco_flant5xl"),
#         "pretrain_flant5xxl": ("blip2_t5", "pretrain_flant5xxl")
#     }

#     model_name, model_type = model_mapping.get(model_choice, (None, None))
#     if not model_name:
#         return "Invalid model selection."

#     print("Loading model...")
#     global_model, global_vis_processors, _ = load_model_and_preprocess(
#         name=model_name, model_type=model_type, is_eval=True, device=device
#     )
#     print("Model loaded successfully.")
#     return "Model loaded successfully."

# def unload_model():
#     print("Unload model...")
#     global global_model, global_vis_processors
#     global_model = None
#     global_vis_processors = None
#     print("Model unloaded successfully.")
#     return "Model unloaded successfully."

# def generate_captions(folder_path):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     #エラー処理
#     if not os.path.exists(folder_path):
#         return [("Error", f"Folder '{folder_path}' not found.")]
#     if not global_model or not global_vis_processors:
#         return [("Error", "Model is not loaded. Please load a model first.")]

#     captions = []
#     file_list = os.listdir(folder_path)
#     image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     for img_file in tqdm(image_files, desc="Processing images", unit="image"):     
#         img_path = os.path.join(folder_path, img_file)
#         caption = process_image(img_path, device)
#         save_caption(caption, img_path)
#         captions.append((img_file, caption))
#     return captions

def on_ui_tabs():
    # Blocksの作成
    with gr.Blocks() as demo:
        # UI
        with gr.Tabs():
            # Promptタブ
            with gr.TabItem("Prompt"):
                with gr.Row():
                    prompt_fpath = gr.Textbox(label="image folder path")
                enable_blip = gr.Checkbox(label="Enable BLIP", info="Using BLIP if metadata is empty")
                prompt_gen_btn = gr.Button("Gen", variant="primary")
                prompt_status = gr.Label()
            # BLIPタブ
            with gr.TabItem("BLIP"):
                with gr.Row():
                    with gr.Column():
                        selection = gr.Radio(["Batch process", "Single Image"], label="Choose Operation Mode")
                        blip_fpath = gr.Textbox(label="image folder path")
                        blip_gen_btn = gr.Button("Gen", variant="primary")
                    with gr.Column():
                        blip_singleimage = gr.Image(label="Image to Caption")                 
                blip_status = gr.Label()
            # BLIP2タブ
            # with gr.TabItem("BLIP2"):
            #     with gr.Row():
            #         blip2_fpath = gr.Textbox(label="image folder path")
            #         BLIP2_model = gr.Dropdown(label="Model", choices=[
            #             "pretrain_opt2.7b", "pretrain_opt6.7b",
            #             "caption_coco_opt2.7b", "caption_coco_opt6.7b",
            #             "pretrain_flant5xl", "caption_coco_flant5xl",
            #             "pretrain_flant5xxl"
            #         ])
            #     with gr.Row():
            #         load_button = gr.Button("Load Model")
            #         unload_button = gr.Button("Unload Model")
            #     model_status = gr.Label()
            #     generate_button = gr.UploadButton("Gen", variant="primary")
            #     blip2_output = gr.Dataframe()
        
        # イベントリスナー 
        # # BLIP2
        # load_button.click(
        #     load_model, inputs=BLIP2_model, outputs=model_status
        # )
        # unload_button.click(
        #     unload_model, outputs=model_status
        # )
        # generate_button.click(
        #     generate_captions, inputs=blip2_fpath, outputs=blip2_output
        # )
        # Prompt
        prompt_gen_btn.click(
            prompt_to_caption, inputs=[prompt_fpath, enable_blip], outputs=prompt_status
        )
        # BLIP
        blip_gen_btn.click(
            caption_with_blip, inputs=[selection, blip_fpath, blip_singleimage], outputs=blip_status
        )

    return [(demo, "PromptToCaption", "PromptToCaption_tab")]
    
        

# 作成したコンポーネントをwebuiに登録
script_callbacks.on_ui_tabs(on_ui_tabs)
