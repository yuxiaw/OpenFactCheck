import streamlit as st
from PIL import Image, ImageOps

def pad_image(image_path, size=(800, 800)):
    """Load an image, pad it to square, and resize to the specified size."""
    try:
        with Image.open(image_path) as img:
            # Ensure the image is RGB
            img = img.convert("RGB")

            # Calculate padding to make the image square
            max_dim = max(img.size)
            left = int((max_dim - img.width) / 2)
            top = int((max_dim - img.height) / 2)
            right = max_dim - img.width - left
            bottom = max_dim - img.height - top

            # Pad the image to make it square
            img = ImageOps.expand(img, (left, top, right, bottom), fill=(255, 255, 255))

            # Resize to the target size using LANCZOS (high-quality resampling)
            img = img.resize(size, Image.Resampling.LANCZOS)
            return img
    except IOError:
        st.error("Unable to load and process image")
        return None
    
def evaluate_factchecker_input_formatcheck(df):
    """
    Check the format of the DataFrame containing the fact-checker input.
    """
    # Check if the DataFrame has the correct number of columns
    if len(df.columns) != 3:
        st.error("The DataFrame must contain 3 columns")
        return
    
    # Check the columns in the DataFrame
    if "verification" not in df.columns:
        st.error("The DataFrame must contain a 'verification' column")
        return
    
    if "time (ms)" not in df.columns:
        st.error("The DataFrame must contain a 'time (ms)' column")
        return
    
    if "cost" not in df.columns:
        st.error("The DataFrame must contain a 'cost' column")
        return
    
    # Check the data types of the columns
    if df["verification"].dtype != bool:
        st.error("The 'verification' column must contain boolean values")
        return
    
    if df["time (ms)"].dtype != float:
        st.error("The 'time (ms)' column must contain float values")
        return
    
    if df["cost"].dtype != float:
        st.error("The 'cost' column must contain float values")
        return
    
    return 