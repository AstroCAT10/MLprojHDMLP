import os
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")

fname = 'heart.csv'

# Define the source file path and the destination directory path
source = f'{path}/{fname}'
destination = os.getcwd()

# Move the file
shutil.move(source, destination)

print("Path to dataset files:", destination)
