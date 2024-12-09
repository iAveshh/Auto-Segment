import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import torch
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dicttoxml import dicttoxml
import json
import os
from datetime import datetime
import base64
from pycocotools import mask as mask_util
import random

# Set page config
st.set_page_config(page_title="Advanced Object Segmentation App", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: #ecf0f1;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Advanced Object Detection & Segmentation")

# Initialize YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n-seg.pt')

model = load_model()

# Create a session state to store results
if 'results_data' not in st.session_state:
    st.session_state.results_data = []

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    # Progress bar
    progress_bar = st.progress(0)
    
    # Process each image
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform prediction
        results = model.predict(image_rgb, conf=0.3)
        result = results[0]
        
        # Get detections
        detections = sv.Detections(
            xyxy=result.boxes.xyxy.cpu().numpy(),
            confidence=result.boxes.conf.cpu().numpy(),
            class_id=result.boxes.cls.cpu().numpy().astype(int)
        )
        
        # Initialize annotator
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        
        # Process masks and create visualization
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            segmented_image = image_rgb.copy()
            
            # Store detection data
            image_data = {
                'filename': uploaded_file.name,
                'objects': [],
                'masks': [],
                'original_image': image_rgb,
                'processed_image': None
            }
            
            for mask_idx, mask in enumerate(masks):
                # Create a random color for each instance
                color = np.random.randint(0, 255, size=3).tolist()
                
                # Resize mask to match image dimensions
                mask_resized = cv2.resize(
                    mask.astype(float),
                    (image_rgb.shape[1], image_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                mask_bool = mask_resized.astype(bool)
                segmented_image[mask_bool] = segmented_image[mask_bool] * 0.5 + np.array(color) * 0.5
                
                # Store mask data
                mask_rle = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
                mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
                
                # Store object data
                if mask_idx < len(detections.class_id):
                    obj_data = {
                        'class': model.model.names[detections.class_id[mask_idx]],
                        'confidence': float(detections.confidence[mask_idx]),
                        'bbox': detections.xyxy[mask_idx].tolist(),
                        'mask': mask_rle
                    }
                    image_data['objects'].append(obj_data)
            
            # Store the final processed image
            annotated_image = box_annotator.annotate(
                scene=segmented_image.copy(),
                detections=detections,
                labels=[
                    f"{model.model.names[class_id]} {conf:0.2f}"
                    for conf, class_id in zip(detections.confidence, detections.class_id)
                ]
            )
            image_data['processed_image'] = annotated_image
            
            # Add to session state
            st.session_state.results_data.append(image_data)
    
    # After processing all images, show random samples and analytics
    if st.session_state.results_data:
        st.header("🖼️ Sample Results")
        
        # Create a 4x4 grid layout
        grid = st.container()
        all_results = list(enumerate(st.session_state.results_data))
        
        # Randomly select 8 images if we have more than 8
        if len(all_results) > 8:
            selected_results = random.sample(all_results, 8)
        else:
            selected_results = all_results
            
        # Pad the results to 16 (4x4 grid) with None values
        while len(selected_results) < 16:
            selected_results.append(None)
            
        # Display images in 4x4 grid
        for row in range(4):
            cols = st.columns(4)
            for col in range(4):
                idx = row * 4 + col
                with cols[col]:
                    if selected_results[idx] is not None:
                        orig_idx, result = selected_results[idx]
                        # Display images
                        try:
                            # Create a tab layout for original and processed
                            tabs = st.tabs(["Original", "Processed"])
                            
                            with tabs[0]:
                                st.image(
                                    result['original_image'],
                                    caption=f"Original {orig_idx + 1}",
                                    use_column_width=True
                                )
                            
                            with tabs[1]:
                                st.image(
                                    result['processed_image'],
                                    caption=f"Processed {orig_idx + 1}",
                                    use_column_width=True
                                )
                                if 'objects' in result:
                                    objects_text = ", ".join([obj['class'] for obj in result['objects']])
                                    st.caption(f"Detected: {objects_text}")
                        except Exception as e:
                            st.error(f"Error displaying image {orig_idx + 1}")
                    else:
                        # Empty space for padding
                        st.empty()
        
        st.header("📊 Analytics Dashboard")
        
        # Collect all objects and their counts
        all_objects = []
        for result in st.session_state.results_data:
            for obj in result['objects']:
                all_objects.append(obj['class'])
        
        # Create DataFrame for visualization
        df = pd.DataFrame({'Object': all_objects})
        object_counts = df['Object'].value_counts()
        
        # Create bar chart
        fig = px.bar(
            x=object_counts.index,
            y=object_counts.values,
            title="Object Distribution",
            labels={'x': 'Object Class', 'y': 'Count'},
            color=object_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence Distribution
        st.subheader("📈 Confidence Score Distribution")
        all_confidences = []
        confidence_labels = []
        for result in st.session_state.results_data:
            for obj in result['objects']:
                all_confidences.append(obj['confidence'])
                confidence_labels.append(obj['class'])
        
        fig_box = px.box(
            y=all_confidences,
            points="all",
            title="Confidence Score Distribution"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Export Options
        st.header("💾 Export Options")
        col1, col2, col3 = st.columns(3)
        
        # JSON Export
        if col1.button("Export as JSON"):
            json_str = json.dumps(st.session_state.results_data, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            st.markdown(
                f'<a href="data:file/json;base64,{b64}" download="detection_results.json" style="display:inline-block; padding:10px 20px; margin:10px 0; background-color:#4CAF50; color:white; text-align:center; text-decoration:none; border-radius:5px;">Download JSON File</a>',
                unsafe_allow_html=True
            )

        # XML Export
        if col2.button("Export as XML"):
            xml = dicttoxml(st.session_state.results_data, custom_root='detection_results', attr_type=False)
            b64 = base64.b64encode(xml).decode()
            st.markdown(
                f'<a href="data:file/xml;base64,{b64}" download="detection_results.xml" style="display:inline-block; padding:10px 20px; margin:10px 0; background-color:#2196F3; color:white; text-align:center; text-decoration:none; border-radius:5px;">Download XML File</a>',
                unsafe_allow_html=True
            )
            
        st.markdown("### Example JSON Format")
        st.code(json.dumps({
            "filename": "image1.jpg",
            "objects": [
                {"class": "cat", "confidence": 0.95, "bbox": [100, 150, 200, 250]},
                {"class": "dog", "confidence": 0.89, "bbox": [300, 350, 400, 450]}
            ]
        }, indent=2), language='json')
        
        st.markdown("### Example XML Format")
        st.code("""
<detection_results>
    <item>
        <filename>image1.jpg</filename>
        <objects>
            <item>
                <class>cat</class>
                <confidence>0.95</confidence>
                <bbox>[100, 150, 200, 250]</bbox>
            </item>
            <item>
                <class>dog</class>
                <confidence>0.89</confidence>
                <bbox>[300, 350, 400, 450]</bbox>
            </item>
        </objects>
    </item>
</detection_results>
""", language='xml')
        
        # COCO Format Export
        if col3.button("Export as COCO"):
            coco_format = {
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Create category mapping
            categories = {}
            cat_id = 1
            ann_id = 1
            
            for img_id, result in enumerate(st.session_state.results_data, 1):
                # Add image info
                coco_format["images"].append({
                    "id": img_id,
                    "file_name": result["filename"],
                    "width": 0,  # Add actual dimensions if needed
                    "height": 0
                })
                
                # Add annotations
                for obj in result["objects"]:
                    if obj["class"] not in categories:
                        categories[obj["class"]] = cat_id
                        coco_format["categories"].append({
                            "id": cat_id,
                            "name": obj["class"],
                            "supercategory": "object"
                        })
                        cat_id += 1
                    
                    # Convert bbox from xyxy to xywh
                    bbox = obj["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    coco_format["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": categories[obj["class"]],
                        "bbox": [bbox[0], bbox[1], width, height],
                        "area": width * height,
                        "segmentation": obj["mask"],
                        "iscrowd": 0
                    })
                    ann_id += 1
            
            json_str = json.dumps(coco_format, indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="detection_results_coco.json" style="display:inline-block; padding:10px 20px; margin:10px 0; background-color:#FF5722; color:white; text-align:center; text-decoration:none; border-radius:5px;">Download COCO Format File</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.write("Please upload images to begin...")