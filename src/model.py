import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tqdm.notebook import tqdm
import pickle

import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer



class EmbeddingModel:
    ''' Class for generate image and text embeddings'''
       
    def __init__(self, model_path=None):
        if not model_path:
            model_path = "openai/clip-vit-base-patch32"
            
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        
    def get_text_templates(self, text_search):
        ''' Apply text templates to the text input '''
        
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
        ''' Encode text input 
        Args:
        - text_input: str or list of str
        - apply_templates: bool, if True, apply text templates to the text input
        '''
        
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


    def encode_images(self, img_paths_list, normalize=True, from_url=False):
        ''' Encode images
        Args:
        - img_paths_list: list of str, list of image paths
        - normalize: bool, if True, normalize image embeddings
        - from_url: bool, if True, load images from url
        '''
        
        if not isinstance(img_paths_list, list):
            img_paths_list = [img_paths_list]

        if not from_url:
            imgs = [Image.open(img_path) for img_path in img_paths_list]    
        else:
            imgs = [Image.open(BytesIO(requests.get(url).content)) for url in img_paths_list]
            
        inputs = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        
        # Get image embeddings
        with torch.no_grad():
            img_embeddings = self.model.get_image_features(pixel_values=inputs)
            img_embeddings = img_embeddings.detach().cpu().numpy()
            
            # Normalize image embeddings
            if normalize:
                img_embeddings /= np.linalg.norm(img_embeddings, axis=1, keepdims=True)
            
        return img_embeddings


    def get_similar_images_indexes(self, img_embeddings_np, input, n=5, input_mode='text', apply_templates=True, from_url=False):
        ''' Get similar images indexes
        Args:
        - img_embeddings_np: numpy array, image embeddings
        - input: str or list of str, text input or list of image paths
        - n: int, number of similar images to return
        - input_mode: str, 'text' or 'image'
        - apply_templates: bool, if True, apply text templates to the text input
        - from_url: bool, if True, load images from url
        '''
        
        if input_mode == 'text':
            # Get text embeddings
            input_embedding = self.encode_text(input, apply_templates=apply_templates)
        
        elif input_mode == 'image':            
            # Get image embeddings
            input_embedding = self.encode_images(input, normalize=True, from_url=from_url)

        # Compute similarity between image and text embeddings
        similarity = np.dot(input_embedding, img_embeddings_np.T)
            
        # Sort results by similarity and reverse
        results = (-similarity).argsort()[0]
        
        # Return indexes of the most similar images
        return results[:n]
    
    
    def generate_img_embeddings(self, img_paths_list, batch_size, save_path=None):
        ''' Generate image embeddings
        Args:
        - img_paths_list: list of str, list of image paths
        - batch_size: int, batch size
        - save_path: str, path to save image embeddings
        '''
        
        # This list could be a generator, but then we would need to provide tqdm with the number of batches as total
        batch_list = [img_paths_list[i:i+batch_size] for i in range(0, len(img_paths_list), batch_size)]
        batch_embeddings = [self.encode_images(batch, normalize=False) for batch in tqdm(batch_list, unit='batch')]
        img_embeddings_np = np.concatenate(batch_embeddings, axis=0)
        
        # Normalize image embeddings
        img_embeddings_np /= np.linalg.norm(img_embeddings_np, ord=2, axis=-1, keepdims=True)

        # Save image embeddings
        if save_path:
            embed_dict = {img_path.name: embedding_np for img_path, embedding_np in zip(img_paths_list, img_embeddings_np)}

            with open(save_path, 'wb') as handle:
                pickle.dump(embed_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return img_embeddings_np
    
    
    def load_embeddings_dict(self, path,):
        ''' Load embeddings dict
        Args:
        - path: str, path to embeddings dict
        '''
        
        with open(path, 'rb') as handle:
            embed_dict = pickle.load(handle)
            
        return embed_dict