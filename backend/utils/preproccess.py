import torch
import PIL
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import io

class utility():
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor()])
    def transform_image(self, image: torch.Tensor, device: str = "cpu", batch: bool = False):
        if not batch:
            image = image.squeeze(0)
            image = image.squeeze(0)
            image = image.squeeze(0) # PARA ESTAR SEGUROS QUE NO QUEDAN DIMENSIONES RARAS
            image = image.unsqueeze(0)
        image = image.float()
        image = image.to(device)
        image = F.interpolate(image, size=(128, 128), mode='bilinear')
        return image
      
    def extract_frames(self, video_path, start=-1, end=-1, every=30):
        """
        Extract frames from a video using OpenCV and return them as a list of NumPy arrays (RGB).
        :param video_path: path of the video
        :param start: start frame
        :param end: end frame
        :param every: frame spacing
        :return: list of frames as NumPy arrays (RGB)
        """

        # Open the video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Failed to open video file."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start < 0:  # if start isn't specified, assume 0
            start = 0
        if end < 0 or end > total_frames:  # if end isn't specified, assume the end of the video
            end = total_frames

        frames_list = list(range(start, end, every))
        extracted_frames = []

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_index >= end:  # Stop if no more frames or end is reached
                break

            if frame_index in frames_list:  # Check if the current frame is in the list
                # Convert BGR (OpenCV default) to RGB (PIL default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames.append(frame_rgb)  # Append the RGB frame

            frame_index += 1

        cap.release()  # Release the video capture object
        return extracted_frames  # Return the list of frames as NumPy arrays


    # Example usage
    # Display the first frame

    def preprocess_image(self, image, target_size=(640, 640)):
        """
        Resizes or crops an image to the target size. Returns RGB NumPy array.

        Args:
            image (str or np.ndarray): Path to the image file or a NumPy array representing the image (assumed RGB).
            target_size (tuple): The desired (width, height) of the output image.

        Returns:
            np.ndarray: The resized or cropped image as an RGB NumPy array.
        """
        if isinstance(image, str):
            img = PIL.Image.open(image).convert('RGB') # Ensure RGB
        elif isinstance(image, np.ndarray):
            # Assume input numpy array is RGB (as returned by extract_frames)
            img = PIL.Image.fromarray(image)
        else:
            raise ValueError("Input must be a file path or a NumPy array.")

        img = img.resize(target_size, PIL.Image.Resampling.BILINEAR) # Use Resampling enum
        return np.array(img)

    def count_frames(self, video_path):
        """
        Counts the total number of frames in a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            int: Total number of frames in the video.
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Failed to open video file."
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()  # Release the video capture object
        return total_frames

   

    def convert_to_tensor(self, video_frames):
        """
        Converts a list of NumPy arrays (frames) into a PyTorch tensor with the shape
        [batch_size, channels, depth, height, width].

        Args:
            video_frames (list of np.ndarray): List of frames with shape (H, W, 3), assumed RGB.

        Returns:
            torch.Tensor: Tensor with shape [1, 3, depth, H, W].
        """
        # Stack frames along a new axis to create a 4D NumPy array (depth, height, width, channels)
        video_np = np.stack(video_frames, axis=0)  # Shape: (depth, H, W, 3)

        # Convert to PyTorch tensor and permute dimensions to [1, 3, depth, height, width]
        video_tensor = torch.tensor(video_np, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0)

        return video_tensor

