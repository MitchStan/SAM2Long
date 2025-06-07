import subprocess
import re
from typing import List, Tuple, Optional
# import spaces # Removed: Not needed for Colab

# Define the command to be executed
command = ["python", "setup.py", "build_ext", "--inplace"]

# Execute the command
result = subprocess.run(command, capture_output=True, text=True)

# Print the output and error (if any)
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)

# Check if the command was successful
if result.returncode == 0:
    print("Command executed successfully.")
else:
    print("Command failed with return code:", result.returncode)

import gradio as gr
from datetime import datetime
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2_video_predictor
from moviepy.editor import ImageSequenceClip

# --- FIX 1: Auto-detect GPU and set performance flags ---
# Set device to CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Enable performance optimizations for CUDA
if device == "cuda":
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    if torch.cuda.get_device_properties(0).major >= 8:
        # Turn on TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

css="""
div#component-18, div#component-25, div#component-35, div#component-41{
    align-items: stretch!important;
}
"""

def sparse_sampling(jpeg_images, original_fps, target_fps=6):
    # Calculate the frame interval for sampling based on the target fps
    frame_interval = int(original_fps // target_fps)
    
    # Sparse sample the jpeg_images by selecting every 'frame_interval' frame
    sampled_images = [jpeg_images[i] for i in range(0, len(jpeg_images), frame_interval)]
    
    return sampled_images

def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    return fps

def clear_points(image):
    # we clean all
    return [
        image,   # first_frame_path
        [],      # tracking_points
        [],      # trackings_input_label
        image,   # points_map
        #gr.State()     # stored_inference_state
    ]

def preprocess_video_in(video_path):

    # Generate a unique ID based on the current date and time
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Set directory with this ID to store video frames 
    extracted_frames_output_dir = f'frames_{unique_id}'
    
    # Create the output directory
    os.makedirs(extracted_frames_output_dir, exist_ok=True)

    ### Process video frames ###
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the frames per second (FPS) of the video
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Calculate the number of frames to process (60 seconds of video)
    max_frames = int(fps * 60)
    
    frame_number = 0
    first_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break
        if frame_number % 6 == 0:
            # Format the frame filename as '00000.jpg'
            frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
            
            # Save the frame as a JPEG file
            cv2.imwrite(frame_filename, frame)
        
        # Store the first frame
        if frame_number == 0:
            first_frame = frame_filename
        
        frame_number += 1
    
    # Release the video capture object
    cap.release()
    
    # scan all the JPEG frame names in this directory
    scanned_frames = [
        p for p in os.listdir(extracted_frames_output_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    return [
        first_frame,           # first_frame_path
        [],          # tracking_points
        [],          # trackings_input_label
        first_frame,           # input_first_frame_image
        first_frame,           # points_map
        extracted_frames_output_dir,            # video_frames_dir
        scanned_frames,        # scanned_frames
        None,                  # stored_inference_state
        None,                  # stored_frame_names
        gr.update(open=False)  # video_in_drawer
    ]

def get_point(point_type, tracking_points, trackings_input_label, input_first_frame_image, evt: gr.SelectData):
    print(f"You selected {evt.value} at {evt.index} from {evt.target}")

    tracking_points.append(evt.index)
    print(f"TRACKING POINT: {tracking_points}")

    if point_type == "include":
        trackings_input_label.append(1)
    elif point_type == "exclude":
        trackings_input_label.append(0)
    print(f"TRACKING INPUT LABEL: {trackings_input_label}")
    
    transparent_background = Image.open(input_first_frame_image).convert('RGBA')
    w, h = transparent_background.size
    
    fraction = 0.02
    radius = int(fraction * min(w, h))
    
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    for index, track in enumerate(tracking_points):
        if trackings_input_label[index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)
    
    return tracking_points, trackings_input_label, selected_point_map
    
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def load_model(checkpoint):
    # Load model accordingly to user's choice
    if checkpoint == "tiny":
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        return [sam2_checkpoint, model_cfg]
    elif checkpoint == "samll":
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        return [sam2_checkpoint, model_cfg]
    elif checkpoint == "base-plus":
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        return [sam2_checkpoint, model_cfg]

def get_mask_sam_process(
    stored_inference_state,
    input_first_frame_image, 
    checkpoint, 
    tracking_points, 
    trackings_input_label, 
    video_frames_dir, 
    scanned_frames, 
    working_frame: str = None, 
    available_frames_to_check: List[str] = [],
):
    
    # --- FIX 2: Use the auto-detected GPU device, not CPU ---
    print(f"USER CHOSEN CHECKPOINT: {checkpoint}")
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    print("MODEL LOADED")

    # Set predictor on the correct device
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("PREDICTOR READY")

    video_dir = video_frames_dir
    frame_names = scanned_frames

    if stored_inference_state is None:
        inference_state = predictor.init_state(video_path=video_dir)
        inference_state['num_pathway'] = 3
        inference_state['iou_thre'] = 0.3
        inference_state['uncertainty'] = 2
        print("NEW INFERENCE_STATE INITIATED")
    else:
        inference_state = stored_inference_state

    # Ensure the inference state is also set to the correct device
    inference_state["device"] = device
        
    if working_frame is None:
        ann_frame_idx = 0
        working_frame = "00000.jpg"
    else:
        match = re.search(r'frame_(\d+)', working_frame)
        if match:
            frame_number = int(match.group(1))
            ann_frame_idx = frame_number
            
    print(f"NEW_WORKING_FRAME PATH: {working_frame}")
    
    ann_obj_id = 1
    points = np.array(tracking_points, dtype=np.float32)
    labels = np.array(trackings_input_label, np.int32)

    # Use autocast for better performance on GPU
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    
    first_frame_output_filename = "output_first_frame.jpg"
    plt.savefig(first_frame_output_filename, format='jpg')
    plt.close()
    if device == "cuda":
        torch.cuda.empty_cache()

    if working_frame not in available_frames_to_check:
        available_frames_to_check.append(working_frame)
        print(available_frames_to_check)
    
    return "output_first_frame.jpg", frame_names, predictor, inference_state, gr.update(choices=available_frames_to_check, value=working_frame, visible=False)

# --- FIX 3: Removed @spaces.GPU decorator and simplified device logic ---
def propagate_to_all(video_in, checkpoint, stored_inference_state, stored_frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, working_frame, progress=gr.Progress(track_tqdm=True)):   
    
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    
    # Set predictor and inference state to the correct device
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = stored_inference_state
    inference_state["device"] = device
    
    frame_names = stored_frame_names
    video_dir = video_frames_dir
    
    frames_output_dir = "frames_output_images"
    os.makedirs(frames_output_dir, exist_ok=True)
    
    jpeg_images = []
    video_segments = {}

    # Use autocast for better performance on GPU during propagation
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
        out_obj_ids, out_mask_logits = predictor.propagate_in_video(inference_state, start_frame_idx=0, reverse=False)
    
    print(out_obj_ids)
    for frame_idx in progress.tqdm(range(0, inference_state['num_frames']), desc="Processing Frames"):
        video_segments[frame_idx] = {out_obj_ids[0]: (out_mask_logits[frame_idx] > 0.0).cpu().numpy()}

    if vis_frame_type == "check":
        vis_frame_stride = 15
    elif vis_frame_type == "render":
        vis_frame_stride = 1
    
    plt.close("all")
    for out_frame_idx in progress.tqdm(range(0, len(frame_names), vis_frame_stride), desc="Rendering Visualization"):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx}.jpg")
        plt.savefig(output_filename, format='jpg')
        plt.close()
        jpeg_images.append(output_filename)

        if f"frame_{out_frame_idx}.jpg" not in available_frames_to_check:
            available_frames_to_check.append(f"frame_{out_frame_idx}.jpg")

    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"JPEG_IMAGES: {jpeg_images}")

    if vis_frame_type == "check":
        return gr.update(value=jpeg_images), gr.update(value=None), gr.update(choices=available_frames_to_check, value=working_frame, visible=True), available_frames_to_check, gr.update(visible=True)
    elif vis_frame_type == "render":
        original_fps = get_video_fps(video_in)
        clip = ImageSequenceClip(jpeg_images, fps=original_fps//6)
        final_vid_output_path = "output_video.mp4"
        clip.write_videofile(final_vid_output_path, codec='libx264', logger='bar')
        
        return gr.update(value=None), gr.update(value=final_vid_output_path), working_frame, available_frames_to_check, gr.update(visible=True)

def update_ui(vis_frame_type):
    if vis_frame_type == "check":
        return gr.update(visible=True), gr.update(visible=False)
    elif vis_frame_type == "render":
        return gr.update(visible=False), gr.update(visible=True)

def switch_working_frame(working_frame, scanned_frames, video_frames_dir):
    new_working_frame = None
    if working_frame == None:
        new_working_frame = os.path.join(video_frames_dir, scanned_frames[0])
        
    else:
        match = re.search(r'frame_(\d+)', working_frame)
        if match:
            frame_number = int(match.group(1))
            ann_frame_idx = frame_number
            new_working_frame = os.path.join(video_frames_dir, scanned_frames[ann_frame_idx])
    return gr.State([]), gr.State([]), new_working_frame, new_working_frame

def reset_propagation(first_frame_path, predictor, stored_inference_state):
    
    predictor.reset_state(stored_inference_state)
    return first_frame_path, [], [], gr.update(value=None, visible=False), stored_inference_state, None, ["frame_0.jpg"], first_frame_path, "frame_0.jpg", gr.update(visible=False)


with gr.Blocks(css=css) as demo:
    # ... (Gradio UI layout remains the same)
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    trackings_input_label = gr.State([])
    video_frames_dir = gr.State()
    scanned_frames = gr.State()
    loaded_predictor = gr.State()
    stored_inference_state = gr.State()
    stored_frame_names = gr.State()
    available_frames_to_check = gr.State([])
    with gr.Column():
        gr.Markdown(
            """
            <h1 style="text-align: center;">🔥 SAM2Long Demo 🔥</h1>
            """
        )
        gr.Markdown(
            """
            This is a simple demo for video segmentation with [SAM2Long](https://github.com/Mark12Ding/SAM2Long).
            """
        )
        gr.Markdown(
            """
            ### 📋 Instructions:
            It is largely built on the [SAM2-Video-Predictor](https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor).
            1. **Upload your video** [MP4-24fps]
            2. With **'include' point type** selected, click on the object to mask on the first frame
            3. Switch to **'exclude' point type** if you want to specify an area to avoid
            4. **Get Mask!**
            5. **Check Propagation** every 15 frames
            6. **Propagate with "render"** to render the final masked video
            7. **Hit Reset** button if you want to refresh and start again
            
            *Note: Input video will be processed for up to 60 seconds only for demo purposes.*
            """
        )
        with gr.Row():
            
            with gr.Column():
                with gr.Group():
                    with gr.Group():
                        with gr.Row():
                            point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include", scale=2)
                            clear_points_btn = gr.Button("Clear Points", scale=1)
                    
                    input_first_frame_image = gr.Image(label="input image", interactive=False, type="filepath", visible=False)                 
                    
                    points_map = gr.Image(
                        label="Point n Click map", 
                        type="filepath",
                        interactive=False
                    )
    
                    with gr.Group():
                        with gr.Row():
                            checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus"], value="tiny")
                            submit_btn = gr.Button("Get Mask", size="lg")

                with gr.Accordion("Your video IN", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Video IN", format="mp4")

                # The duplicate button HTML won't work in Colab, but leaving it as it doesn't hurt
                gr.HTML("""
                <a href="https://huggingface.co/spaces/your_space_id?duplicate=true">
                    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg-dark.svg" alt="Duplicate this Space" />
                </a> to skip queue and avoid OOM errors from heavy public load
                """)
            
            with gr.Column():
                with gr.Group():
                    working_frame = gr.Dropdown(label="working frame ID", choices=["frame_0.jpg"], value="frame_0.jpg", visible=False, allow_custom_value=False, interactive=True)
                    output_result = gr.Image(label="current working mask ref")
                with gr.Group():
                    with gr.Row():
                        vis_frame_type = gr.Radio(label="Propagation level", choices=["check", "render"], value="check", scale=2)
                        propagate_btn = gr.Button("Propagate", scale=2)

                reset_prpgt_brn = gr.Button("Reset", visible=False)
                output_propagated = gr.Gallery(label="Propagated Mask samples gallery", columns=4, visible=False)
                output_video = gr.Video(visible=False)
    
    video_in.upload(
        fn = preprocess_video_in, 
        inputs = [video_in], 
        outputs = [
            first_frame_path, 
            tracking_points,
            trackings_input_label,
            input_first_frame_image,
            points_map,
            video_frames_dir,
            scanned_frames,
            stored_inference_state,
            stored_frame_names,
            video_in_drawer,
        ],
        queue = False
    )
    
    points_map.select(
        fn = get_point, 
        inputs = [
            point_type,
            tracking_points,
            trackings_input_label,
            input_first_frame_image,
        ], 
        outputs = [
            tracking_points,
            trackings_input_label,
            points_map,
        ], 
        queue = False
    )

    clear_points_btn.click(
        fn = clear_points,
        inputs = input_first_frame_image,
        outputs = [
            first_frame_path, 
            tracking_points, 
            trackings_input_label, 
            points_map,
        ],
        queue=False
    )
    
    submit_btn.click(
        fn = get_mask_sam_process,
        inputs = [
            stored_inference_state,
            input_first_frame_image, 
            checkpoint, 
            tracking_points, 
            trackings_input_label, 
            video_frames_dir, 
            scanned_frames, 
            working_frame,
            available_frames_to_check,
        ],
        outputs = [
            output_result, 
            stored_frame_names, 
            loaded_predictor,
            stored_inference_state,
            working_frame,
        ]
    )

    reset_prpgt_brn.click(
        fn = reset_propagation,
        inputs = [first_frame_path, loaded_predictor, stored_inference_state],
        outputs = [points_map, tracking_points, trackings_input_label, output_propagated, stored_inference_state, output_result, available_frames_to_check, input_first_frame_image, working_frame, reset_prpgt_brn],
        queue=False
    )

    propagate_btn.click(
        fn = update_ui,
        inputs = [vis_frame_type],
        outputs = [output_propagated, output_video],
        queue=False
    ).then(
        fn = propagate_to_all,
        inputs = [video_in, checkpoint, stored_inference_state, stored_frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, working_frame],
        outputs = [output_propagated, output_video, working_frame, available_frames_to_check, reset_prpgt_brn]
    )

demo.launch(share=True, debug=True)
