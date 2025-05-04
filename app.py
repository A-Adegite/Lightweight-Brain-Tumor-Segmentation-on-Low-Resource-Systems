import streamlit as st
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import gdown  # For downloading files from Google Drive
import zipfile  # To handle folder uploads
import tempfile  # To handle temporary files
from tensorflow.keras.utils import to_categorical

# Title of the app
st.title("Brain Tumor Segmentation using 3D U-Net - (Lightweight Architecture on Normal CPUs)")

# Function to download the default model from Google Drive
def download_default_model():
    file_id = "1lV1SgafomQKwgv1NW2cjlpyb4LwZXFwX"
    output_path = "default_model.keras"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return output_path

# Load the default model
@st.cache_resource
def load_default_model():
    model_path = download_default_model()
    model = load_model(model_path, compile=False)
    return model

default_model = load_default_model()

# Function to preprocess a NIfTI file
def preprocess_nifti(file_path):
    image = nib.load(file_path).get_fdata()
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

# Function to combine 4 channels
def combine_channels(t1n, t1c, t2f, t2w):
    combined_image = np.stack([t1n, t1c, t2f, t2w], axis=3)
    combined_image = combined_image[56:184, 56:184, 13:141]
    return combined_image

# Function to run segmentation
def run_segmentation(model, input_image):
    input_image = np.expand_dims(input_image, axis=0)
    if len(input_image.shape) != 5:
        st.error(f"Unexpected shape for input_image: {input_image.shape}. Expected: (batch, H, W, D, channels)")
        return None
    prediction = model.predict(input_image)
    return np.argmax(prediction, axis=4)[0, :, :, :]

# Sidebar for custom model
st.sidebar.header("Upload Your Own Model")
uploaded_model = st.sidebar.file_uploader("Upload a Keras model (.keras)", type=["keras"])

if uploaded_model is not None:
    with open("temp_model.keras", "wb") as f:
        f.write(uploaded_model.getbuffer())
    try:
        model = load_model("temp_model.keras", compile=False)
        st.sidebar.success("Custom model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom model: {e}")
        st.sidebar.info("Using the default model instead.")
        model = default_model
else:
    model = default_model
    st.sidebar.info("Using the default model.")

# Upload NIfTI ZIP folder
st.header("Upload a Folder Containing NIfTI Files")
uploaded_folder = st.file_uploader("Upload a zip containing T1n, T1c, T2f, T2w NIfTI files", type=["zip"])

if uploaded_folder is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded_folder.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_folder.getbuffer())
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # Search for required files
        paths = {'t1n': None, 't1c': None, 't2f': None, 't2w': None, 'seg': None}
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith("t1n.nii.gz"):
                    paths['t1n'] = os.path.join(root, file)
                elif file.endswith("t1c.nii.gz"):
                    paths['t1c'] = os.path.join(root, file)
                elif file.endswith("t2f.nii.gz"):
                    paths['t2f'] = os.path.join(root, file)
                elif file.endswith("t2w.nii.gz"):
                    paths['t2w'] = os.path.join(root, file)
                elif file.endswith("seg.nii.gz"):
                    paths['seg'] = os.path.join(root, file)

        if all(paths[key] for key in ['t1n', 't1c', 't2f', 't2w']):
            # Preprocess each
            t1n = preprocess_nifti(paths['t1n'])
            t1c = preprocess_nifti(paths['t1c'])
            t2f = preprocess_nifti(paths['t2f'])
            t2w = preprocess_nifti(paths['t2w'])
            combined_image = combine_channels(t1n, t1c, t2f, t2w)
            st.write(f"Shape of combined_image: {combined_image.shape}")

            if len(combined_image.shape) != 4:
                st.error("Combined image shape incorrect.")
            else:
                st.write("Running segmentation...")
                segmentation_result = run_segmentation(model, combined_image)
                st.write("Segmentation completed!")

                if paths['seg']:
                    mask = nib.load(paths['seg']).get_fdata().astype(np.uint8)
                    mask[mask == 4] = 3
                    mask_argmax = np.argmax(to_categorical(mask, num_classes=4), axis=3)
                else:
                    mask_argmax = None

                # Visualize
                slice_indices = [75, 90, 100]
                fig, ax = plt.subplots(3, 4, figsize=(18, 12))
                for i, idx in enumerate(slice_indices):
                    img = np.rot90(combined_image[:, :, idx, 0])
                    pred = np.rot90(segmentation_result[:, :, idx])
                    ax[i, 0].imshow(img, cmap='gray')
                    ax[i, 0].set_title(f"Image - Slice {idx}")
                    if mask_argmax is not None:
                        mask_vis = np.rot90(mask_argmax[:, :, idx])
                        ax[i, 1].imshow(mask_vis)
                        ax[i, 1].set_title(f"Mask - Slice {idx}")
                    else:
                        ax[i, 1].axis('off')
                    ax[i, 2].imshow(pred)
                    ax[i, 2].set_title(f"Prediction - Slice {idx}")
                    ax[i, 3].imshow(img, cmap='gray')
                    ax[i, 3].imshow(pred, alpha=0.5)
                    ax[i, 3].set_title(f"Overlay - Slice {idx}")
                plt.tight_layout()
                st.pyplot(fig)

                # Save + download
                output_file = "segmentation_result.nii.gz"
                nib.save(nib.Nifti1Image(segmentation_result.astype(np.float32), np.eye(4)), output_file)
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="Download Segmentation Result",
                        data=f,
                        file_name=output_file,
                        mime="application/octet-stream"
                    )
                os.remove(output_file)
        else:
            st.error("Missing one or more required NIfTI files (T1n, T1c, T2f, T2w).")
