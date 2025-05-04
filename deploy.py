import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
 
# Function to preprocess a new NIfTI file
def preprocess_nifti(file_path, patch_size=(64, 64, 64)):
    # Load the NIfTI file
    image = nib.load(file_path).get_fdata()
    
    # Normalize the image
    scaler = MinMaxScaler()
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    
    # Pad or crop the image to match the patch size
    if image.shape != patch_size:
      # Example: Center cropping
      start_x = (image.shape[0] - patch_size[0]) // 2
      start_y = (image.shape[1] - patch_size[1]) // 2
      start_z = (image.shape[2] - patch_size[2]) // 2
      image = image[start_x:start_x + patch_size[0],  start_y:start_y +
patch_size[1], start_z:start_z + patch_size[2]]
 
    # Expand dimensions for model input
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image
 
# Function to run segmentation and save results
def run_segmentation(model, input_file, output_folder):
    # Preprocess the input file
    input_image = preprocess_nifti(input_file)
  
    # Run segmentation
    prediction = model.predict(input_image)
    prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]
  
    # Save the segmentation result
    output_file = os.path.join(output_folder, os.path.basename(input_file).replace('.nii.gz', '_segmentation.nii.gz'))
    nib.save(nib.Nifti1Image(prediction_argmax, np.eye(4)), output_file)
    print(f"Segmentation saved to {output_file}")
 
# Example usage
input_file = "path/to/new_image.nii.gz"  #replace with the correct path to your image
output_folder = "path/to/output_folder" #replace with the correct part to where you want to save the output
run_segmentation(my_model, input_file, output_folder)