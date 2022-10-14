import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer


class EmbeddingModel:
    
    def __init__(self, model_path=None):
        if not model_path:
            model_path = "openai/clip-vit-base-patch32"
            
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        
    def get_text_templates(self, text_search):
        text_templates = ['A photo of a {}.',
                    'a photo of the {}.',
                    'a bad photo of a {}.',
                    'a photo of many {}.',
                    'a low resolution photo of the {}.',
                    'a photo of my {}.',
                    'a close-up photo of a {}.',
                    'a cropped photo of a {}.',
                    'a photo of the {}.',
                    'a good photo of the {}.',
                    'a photo of one {}.',
                    'a close-up photo of the {}.',
                    'a photo of a {}.',
                    'the {} in a video game.',
                    'a origami {}.',
                    'a low resolution photo of a {}.',
                    'a photo of a large {}.',
                    'a blurry photo of a {}.',
                    'a sketch of the {}.',
                    'a pixelated photo of a {}.',
                    'a good photo of a {}.',
                    'a drawing of the {}.',
                    'a photo of a small {}.',
                    ]
            
        text_inputs = [template.format(text_search) for template in text_templates] # format with class
        return text_inputs


    def encode_text(self, text_input, apply_templates=True):
        # If apply_templates is True, apply text templates to the text input
        if apply_templates:
            text_input = self.get_text_templates(text_input)
        
        # Get text embeddings 
        with torch.no_grad():
            # Encode and normalize the description using CLIP
            inputs = self.processor(text_input, return_tensors="pt", padding=True)
            text_embeddings =  self.model.get_text_features(**inputs).detach().cpu().numpy()
            
            # Normalize text embeddings
            text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        return text_embeddings


    def encode_images(self, img_paths_list, normalize=True, nobg=False):
        if not isinstance(img_paths_list, list):
            img_paths_list = [img_paths_list]

        imgs = [Image.open(img_path) for img_path in img_paths_list]    
        inputs = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        
        # Get image embeddings
        with torch.no_grad():
            img_embeddings = self.model.get_image_features(pixel_values=inputs)
            img_embeddings = img_embeddings.detach().cpu().numpy()
            
            # Normalize image embeddings
            if normalize:
                img_embeddings /= np.linalg.norm(img_embeddings, axis=1, keepdims=True)
            
        return img_embeddings


    def get_similar_images_indexes(self, img_embeddings_np, text_search, n=5, apply_templates=True):
        # Get text embeddings
        text_embeddings = self.encode_text(text_search, apply_templates=apply_templates)
        
        # Compute cosine similarity between image and text embeddings
        similarity = np.dot(text_embeddings, img_embeddings_np.T)
        
        # Sort results by similarity and reverse
        results = (-similarity).argsort()[0]
        
        # Return indexes of the most similar images
        return results[:n]