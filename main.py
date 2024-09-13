
import streamlit as st
import rawpy
import cv2
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from skimage import filters
import pandas as pd

# Title and description
st.title("Enhanced Coffee Roast Degree Analyzer")
st.write("""
Upload your RAW DNG images of coffee grounds to analyze the roast degree.

This app automatically computes color statistics, and provides visualizations.
""")


def remove_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# Sidebar for calibration and settings
st.sidebar.title("Calibration and Settings")

# Option to upload a calibration image
calibration_file = st.sidebar.file_uploader("Upload Calibration Image (optional)", type=['dng'])
apply_calibration = st.sidebar.checkbox("Apply Color Calibration", value=False)

# Instructions for calibration image
if apply_calibration:
    st.sidebar.write("""
    **Instructions for Calibration Image:**
    - Take a photo of a standard gray card under the same lighting conditions.
    - Ensure the gray card fills most of the image.
    """)
else:
    st.sidebar.write("Color calibration is not applied.")

# Upload multiple DNG files
uploaded_files = st.file_uploader("Upload DNG files", type=["dng"], accept_multiple_files=True)

# Process the calibration image if provided
if calibration_file is not None and apply_calibration:
    with rawpy.imread(BytesIO(calibration_file.read())) as raw:
        # Extract calibration data (white balance gains)
        calibration_wb = raw.daylight_whitebalance
    st.sidebar.success("Calibration image uploaded and will be applied.")
else:
    calibration_wb = None

# Function to apply calibration (simple white balance correction)
def apply_calibration_function(image, calibration_wb):
    if calibration_wb is not None:
        # Apply the calibration white balance gains
        # Normalize the gains
        gains = calibration_wb / np.mean(calibration_wb)
        image = image.astype(np.float32)
        image[..., 0] *= gains[0]  # Red channel
        image[..., 1] *= gains[1]  # Green channel
        image[..., 2] *= gains[2]  # Blue channel
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# Initialize lists to store results
mean_lightness_values = []
std_lightness_values = []
image_names = []
histogram_data_list = []


st.sidebar.title("ROI Settings")
roi_width = st.sidebar.slider("ROI Width", min_value=50, max_value=2000, value=1000, step=10)
roi_height = st.sidebar.slider("ROI Height", min_value=50, max_value=2000, value=1000, step=10)

# These sliders allow the user to move the ROI around the image
roi_x_offset = st.sidebar.slider("ROI X Offset", min_value=0, max_value=100, value=50)
roi_y_offset = st.sidebar.slider("ROI Y Offset", min_value=0, max_value=100, value=50)

if uploaded_files is not None and len(uploaded_files) > 0:
    for uploaded_file in uploaded_files:
        with rawpy.imread(BytesIO(uploaded_file.read())) as raw:
            # Process the RAW image into an RGB image
            rgb_image = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
            
            # Apply calibration if selected
            if apply_calibration and calibration_wb is not None:
                rgb_image = apply_calibration_function(rgb_image, calibration_wb)
            
            # Convert to LAB color space
            lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            
            # Automatic coffee grounds detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY_INV, 51, 10)
            # Get the dimensions of the image
            height, width, _ = rgb_image.shape

            # Calculate the center position for the ROI
            center_x = width // 2 + int((roi_x_offset - 50) / 100 * width)
            center_y = height // 2 + int((roi_y_offset - 50) / 100 * height)

            # Ensure the ROI doesn't go out of bounds
            x1 = max(0, center_x - roi_width // 2)
            y1 = max(0, center_y - roi_height // 2)
            x2 = min(width, center_x + roi_width // 2)
            y2 = min(height, center_y + roi_height // 2)

            # Extract the ROI from the image
            roi = rgb_image[y1:y2, x1:x2]

            # Convert the ROI to the Lab color space
            roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
                
            # Assume the largest contour is the coffee grounds
            # Create a mask
            
            # Compute statistics within the mask
            l_channel = roi_lab[:,:,0]
            mean_lightness = cv2.mean(l_channel)[0]
            std_lightness = cv2.meanStdDev(l_channel)[1][0][0]
            # Extract histogram data
            hist_data = l_channel.ravel()
            
            # Save results
            mean_lightness_values.append(mean_lightness)
            std_lightness_values.append(std_lightness)
            image_names.append(uploaded_file.name)
            histogram_data_list.append(hist_data)
            
            # Visualize detected area
            rgb_image_with_roi = rgb_image.copy()
            cv2.rectangle(rgb_image_with_roi, (x1, y1), (x2, y2), (255, 0, 0), 3)
            st.image(rgb_image_with_roi, caption=f"Detected Coffee Area in {uploaded_file.name}", use_column_width=True)
            
            st.write(f"**Mean Lightness:** {mean_lightness:.2f}")
            st.write(f"**Standard Deviation of Lightness:** {std_lightness:.2f}")
            
            # Plot histogram
            fig, ax = plt.subplots()
            ax.hist(hist_data, bins=256, range=(0, 255), color='black', alpha=0.7)
            ax.set_title(f'Lightness Histogram for {uploaded_file.name}')
            ax.set_xlabel('Lightness Value')
            ax.set_ylabel('Pixel Count')
            st.pyplot(fig)
    
    # Display summary if more than one image processed
    if len(mean_lightness_values) > 1:
        fig, ax = plt.subplots()
        ax.errorbar(image_names, mean_lightness_values, yerr=std_lightness_values, fmt='o', ecolor='r', capthick=2)
        ax.set_title('Mean Lightness Across Samples')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Mean Lightness')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Present results in a table
    results_df = pd.DataFrame({
        'Image': image_names,
        'Mean Lightness': mean_lightness_values,
        'Std Dev Lightness': std_lightness_values
    })
    
    st.write("### Summary of Analysis")
    st.dataframe(results_df)
    
    # Option to download the results as CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        mime='text/csv',
        file_name="roast_degree_analysis.csv",
    )
else:
    st.info("Please upload one or more DNG files to begin analysis.")