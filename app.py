import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import matplotlib.cm as cm

# Load model
autoencoder = load_model('model/screw_autoencoder_final.keras')

def apply_colormap(img):
    """Apply a colormap to a grayscale image."""
    norm_img = img / np.max(img)  # Normalize to range [0, 1]
    colormap = cm.get_cmap('hot')  # Get the 'hot' colormap
    color_img = colormap(norm_img)  # Apply colormap
    color_img = (color_img[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB
    return color_img

def img_to_base64(img):
    """Convert numpy array to base64-encoded PNG."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    return buf.getvalue()

def draw_anomaly_boxes(original_img, anomaly_map, box_size=40):
    """Draw bounding boxes on the original image based on the anomaly map."""
    anomaly_map = anomaly_map.squeeze()  # Remove extra dimensions
    original_img = original_img.squeeze()  # Remove extra dimensions

    # Find the coordinates of the highest difference
    max_coords = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
    
    # Calculate top-left and bottom-right coordinates of the bounding box
    start_x = max(0, max_coords[1] - box_size // 2)
    start_y = max(0, max_coords[0] - box_size // 2)
    end_x = min(anomaly_map.shape[1], max_coords[1] + box_size // 2)
    end_y = min(anomaly_map.shape[0], max_coords[0] + box_size // 2)
    
    # Draw bounding box on the original image
    image_with_box = cv2.rectangle(
        np.copy(original_img), 
        (start_x, start_y), 
        (end_x, end_y), 
        (255, 0, 0),  # Color (red) in BGR format
        1  # Thickness of the rectangle
    )
    return image_with_box

def main():
    st.title("Screw Anomaly Detection with Autoencoder")
    
    uploaded_file = st.file_uploader("Choose an image...", type="png")
    
    if uploaded_file is not None:
        # Read image
        img = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

        # Process image
        img = cv2.resize(img, (256, 256))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predict (reconstruct) the image
        reconstructed_img = autoencoder.predict(img)
        
        # Calculate anomaly map
        anomaly_map = np.abs(img - reconstructed_img)
        anomaly_map = anomaly_map[0, :, :, 0]

        # Convert images to base64
        original_img = (img[0, :, :, 0] * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        reconstructed_img_pil = Image.fromarray((reconstructed_img[0, :, :, 0] * 255).astype(np.uint8))
        anomaly_map_colored = apply_colormap(anomaly_map)
        anomaly_map_pil = Image.fromarray(anomaly_map_colored)
        
        # Draw bounding box
        original_img_with_box = draw_anomaly_boxes(original_img_bgr, anomaly_map)

        # Display images
        st.image(Image.fromarray(original_img_with_box), caption='Original with Anomaly Box', use_column_width=True)
        st.image(reconstructed_img_pil, caption='Reconstructed Image', use_column_width=True)
        st.image(anomaly_map_pil, caption='Anomaly Map', use_column_width=True)

if __name__ == "__main__":
    main()
