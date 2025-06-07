import subprocess
import re
from typing import List, Tuple, Optional

# Define the command to be executed
command = ["python", "setup.py", "build_ext", "--inplace"]
result = subprocess.run(command, capture_output=True, text=True)
print("Output:\n", result.stdout)
print("Errors:\n", result.stderr)
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
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from moviepy.editor import ImageSequenceClip

# Auto-detect GPU and set performance flags
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

css="""
div#component-18, div#component-25, div#component-35, div#component-41{
    align-items: stretch!important;
}
"""

def draw_mask_and_points_on_image(image_path, mask, points=None, labels=None):
    image = cv2.imread(image_path)
    color = np.array([0, 255, 0], dtype=np.uint8)
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    mask_overlay[mask] = color
    blended_image = cv2.addWeighted(image, 1.0, mask_overlay, 0.6, 0)

    if points is not None and labels is not None:
        h, w, _ = image.shape
        radius = int(0.01 * min(h, w))
        for (x, y), label in zip(points, labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(blended_image, (int(x), int(y)), radius, color, -1)
            cv2.circle(blended_image, (int(x), int(y)), radius, (255,255,255), 2)

    return blended_image

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def clear_points(image):
    return [image, [], [], image]

def preprocess_video_in(video_path):
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S')
    extracted_frames_output_dir = f'frames_{unique_id}'
    os.makedirs(extracted_frames_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 60)
    frame_number = 0
    first_frame = None
    
    # --- FIX: Set JPEG quality parameter ---
    jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]

    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break
        if frame_number % 6 == 0:
            # --- FIX: Save as high-quality JPG, not PNG ---
            frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
            cv2.imwrite(frame_filename, frame, jpeg_quality)
        
        if frame_number == 0:
            first_frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
            if not os.path.exists(first_frame_filename):
                 cv2.imwrite(first_frame_filename, frame, jpeg_quality)
            first_frame = first_frame_filename
        
        frame_number += 1
    
    cap.release()
    
    # --- FIX: Scan for JPG files ---
    scanned_frames = [
        p for p in os.listdir(extracted_frames_output_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg"]
    ]
    scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    return [
        first_frame, [], [], first_frame, first_frame,
        extracted_frames_output_dir, scanned_frames, None, None, gr.update(open=False)
    ]

def get_point(point_type, tracking_points, trackings_input_label, input_first_frame_image, evt: gr.SelectData):
    tracking_points.append(evt.index)
    label = 1 if point_type == "include" else 0
    trackings_input_label.append(label)

    points_to_draw = [tuple(map(int, p)) for p in tracking_points]
    img_with_points = cv2.imread(input_first_frame_image)
    h, w, _ = img_with_points.shape
    radius = int(0.01 * min(h, w))

    for point, lbl in zip(points_to_draw, trackings_input_label):
        color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
        cv2.circle(img_with_points, point, radius, color, -1)
        cv2.circle(img_with_points, point, radius, (255, 255, 255), 2)

    selected_point_map = Image.fromarray(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB))
    
    return tracking_points, trackings_input_label, selected_point_map
    
def load_model(checkpoint):
    if checkpoint == "tiny":
        return ["./checkpoints/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"]
    elif checkpoint == "small":
        return ["./checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"]
    elif checkpoint == "base-plus":
        return ["./checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"]

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
    print(f"USER CHOSEN CHECKPOINT: {checkpoint}")
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("PREDICTOR READY")

    video_dir = video_frames_dir
    frame_names = scanned_frames

    if stored_inference_state is None:
        # This is where the error happened. It needs to find JPGs.
        inference_state = predictor.init_state(video_path=video_dir)
        inference_state.update({'num_pathway': 3, 'iou_thre': 0.3, 'uncertainty': 2})
        print("NEW INFERENCE_STATE INITIATED")
    else:
        inference_state = stored_inference_state
    inference_state["device"] = device
        
    ann_frame_idx = 0
    if working_frame is None:
        # --- FIX: Default filename is JPG ---
        working_frame = "00000.jpg"
    else:
        match = re.search(r'frame_(\d+)', working_frame)
        if match:
            ann_frame_idx = int(match.group(1))

    points = np.array(tracking_points, dtype=np.float32)
    labels = np.array(trackings_input_label, np.int32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=1,
            points=points, labels=labels,
        )

    mask_np = (out_mask_logits[0] > 0.0).cpu().numpy()
    current_frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    final_image = draw_mask_and_points_on_image(current_frame_path, mask_np, points, labels)
    
    # --- FIX: Save preview as high-quality JPG ---
    first_frame_output_filename = "output_first_frame.jpg"
    cv2.imwrite(first_frame_output_filename, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    if device == "cuda":
        torch.cuda.empty_cache()

    if working_frame not in available_frames_to_check:
        available_frames_to_check.append(working_frame)
    
    return first_frame_output_filename, frame_names, predictor, inference_state, gr.update(choices=available_frames_to_check, value=working_frame, visible=False)

def propagate_to_all(video_in, checkpoint, stored_inference_state, stored_frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, working_frame, progress=gr.Progress(track_tqdm=True)):   
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = stored_inference_state
    inference_state["device"] = device
    
    frame_names = stored_frame_names
    video_dir = video_frames_dir
    
    frames_output_dir = "frames_output_images_hq"
    os.makedirs(frames_output_dir, exist_ok=True)
    
    output_image_paths = []
    video_segments = {}
    jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device=="cuda")):
        out_obj_ids, out_mask_logits = predictor.propagate_in_video(inference_state, start_frame_idx=0, reverse=False)
    
    for frame_idx in progress.tqdm(range(0, inference_state['num_frames']), desc="Processing Frames"):
        video_segments[frame_idx] = {out_obj_ids[0]: (out_mask_logits[frame_idx] > 0.0).cpu().numpy()}

    vis_frame_stride = 15 if vis_frame_type == "check" else 1
    
    for out_frame_idx in progress.tqdm(range(0, len(frame_names), vis_frame_stride), desc="Rendering High-Quality Frames"):
        original_frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
        obj_id = list(video_segments[out_frame_idx].keys())[0]
        mask_np = video_segments[out_frame_idx][obj_id]
        final_frame = draw_mask_and_points_on_image(original_frame_path, mask_np)

        # --- FIX: Save each frame as a high-quality JPG ---
        output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx:05d}.jpg")
        cv2.imwrite(output_filename, final_frame, jpeg_quality)
        output_image_paths.append(output_filename)

        # --- FIX: Track JPG filenames ---
        if f"frame_{out_frame_idx}.jpg" not in available_frames_to_check:
            available_frames_to_check.append(f"frame_{out_frame_idx}.jpg")

    if device == "cuda":
        torch.cuda.empty_cache()

    if vis_frame_type == "check":
        return gr.update(value=output_image_paths), gr.update(value=None), gr.update(choices=available_frames_to_check, value=working_frame, visible=True), available_frames_to_check, gr.update(visible=True)
    
    elif vis_frame_type == "render":
        original_fps = get_video_fps(video_in)
        all_rendered_frames = sorted([os.path.join(frames_output_dir, f) for f in os.listdir(frames_output_dir)])
        clip = ImageSequenceClip(all_rendered_frames, fps=original_fps//6)
        final_vid_output_path = "output_video_hq.mp4"
        
        clip.write_videofile(
            final_vid_output_path, codec='libx264', bitrate='8000k', logger='bar'
        )
        
        return gr.update(value=None), gr.update(value=final_vid_output_path), working_frame, available_frames_to_check, gr.update(visible=True)

def update_ui(vis_frame_type):
    return (gr.update(visible=True), gr.update(visible=False)) if vis_frame_type == "check" else (gr.update(visible=False), gr.update(visible=True))

def reset_propagation(first_frame_path, predictor, stored_inference_state):
    if predictor and stored_inference_state:
        predictor.reset_state(stored_inference_state)
    # --- FIX: Default filenames are JPG ---
    return first_frame_path, [], [], gr.update(value=None, visible=False), None, None, ["frame_0.jpg"], first_frame_path, "frame_0.jpg", gr.update(visible=False)


with gr.Blocks(css=css) as demo:
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
        gr.Markdown("<h1 style='text-align: center;'>🔥 SAM2Long HQ Demo 🔥</h1>")
        # Instructions...
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        point_type = gr.Radio(label="Point Type", choices=["include", "exclude"], value="include", scale=2)
                        clear_points_btn = gr.Button("Clear Points", scale=1)
                    input_first_frame_image = gr.Image(label="Input Image", interactive=False, type="filepath", visible=False)
                    points_map = gr.Image(label="Point & Click Map", type="pil", interactive=True)
                    with gr.Row():
                        checkpoint = gr.Dropdown(label="Checkpoint", choices=["tiny", "small", "base-plus"], value="tiny")
                        submit_btn = gr.Button("Get Mask", size="lg")
                with gr.Accordion("Your Video IN", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Video IN", format="mp4")
            with gr.Column():
                # --- FIX: Default dropdown choice is JPG ---
                working_frame = gr.Dropdown(label="Working Frame ID", choices=["frame_0.jpg"], value="frame_0.jpg", visible=False, interactive=True)
                output_result = gr.Image(label="Current Working Mask")
                with gr.Group():
                    with gr.Row():
                        vis_frame_type = gr.Radio(label="Propagation Level", choices=["check", "render"], value="check", scale=2)
                        propagate_btn = gr.Button("Propagate", scale=2)
                reset_prpgt_brn = gr.Button("Reset", visible=False)
                output_propagated = gr.Gallery(label="Propagated Mask Samples", columns=4, visible=False)
                output_video = gr.Video(label="Final High-Quality Video", visible=False)

    video_in.upload(
        preprocess_video_in, [video_in], 
        [first_frame_path, tracking_points, trackings_input_label, input_first_frame_image, points_map, video_frames_dir, scanned_frames, stored_inference_state, stored_frame_names, video_in_drawer],
        queue=False
    )
    
    points_map.select(
        get_point, [point_type, tracking_points, trackings_input_label, input_first_frame_image], 
        [tracking_points, trackings_input_label, points_map],
        queue=False
    )

    clear_points_btn.click(
        clear_points, input_first_frame_image,
        [first_frame_path, tracking_points, trackings_input_label, points_map],
        queue=False
    )
    
    submit_btn.click(
        get_mask_sam_process,
        [stored_inference_state, input_first_frame_image, checkpoint, tracking_points, trackings_input_label, video_frames_dir, scanned_frames, working_frame, available_frames_to_check],
        [output_result, stored_frame_names, loaded_predictor, stored_inference_state, working_frame]
    )

    reset_prpgt_brn.click(
        reset_propagation,
        [first_frame_path, loaded_predictor, stored_inference_state],
        [points_map, tracking_points, trackings_input_label, output_propagated, stored_inference_state, output_result, available_frames_to_check, input_first_frame_image, working_frame, reset_prpgt_brn],
        queue=False
    )

    propagate_btn.click(
        update_ui, vis_frame_type, [output_propagated, output_video], queue=False
    ).then(
        propagate_to_all,
        [video_in, checkpoint, stored_inference_state, stored_frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, working_frame],
        [output_propagated, output_video, working_frame, available_frames_to_check, reset_prpgt_brn]
    )

demo.launch(share=True, debug=True)
