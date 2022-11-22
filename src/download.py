from pathlib import Path
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import flickr_api



class FlickrDownloader:
    ''' Class for download images from Flickr '''
    
    def __init__(self, api_key, api_secret):
        self.api = flickr_api.set_keys(api_key=api_key, api_secret=api_secret)
        
    def get_all_photos(self):
        ''' Get all photos from a user 
        Flickr API only allows you to retrieve photos on pages of up to 500 photos.
        This generator abstracts from that limitation so all photos are iterated.
        '''

        photos = self.user.getPublicPhotos()
        page = 1
        while photos:
            yield from photos
            # Advance to the next page
            page += 1
            photos = self.user.getPublicPhotos(page=page)

    def download_photo(self, photo, dest_folder):
        ''' Download a photo from Flickr 
        Args:
        - photo: Photo object from Flickr
        - dest_folder: Destination folder to save the image
        '''
        
        # Using private _getOutputFilename as a quick hack to retrieve file extension
        extension = photo._getOutputFilename('', None)
        
        # Get image biggest size
        try:
            filename = f'{photo.title.replace("/", "-")}_{photo.id}{extension}'
            sizes = photo.getSizes()
            biggest_size = list(sizes.keys())[-1]
        except Exception as e:
            print(e, 'Error getting image metadata')
            
        # Get original image url
        try:
            url = sizes[biggest_size]['source']
        except Exception as e:
            print(e, 'Error getting image url')
        
        # Save image in local
        try:
            dest_path = dest_folder / filename
            if not dest_path.exists():
                photo.save(str(dest_path))
        except Exception as e:
            print(e, 'Error saving image ', filename)
        
    def download_public_photos_from_user(self, user_name, dest_folder):
        ''' Download all public photos from a user 
        Args:
        - user_name: Flickr user name
        - dest_folder: Destination folder to save the images
        '''
        
        # Find user by user name
        try:
            self.user = flickr_api.Person.findByUserName(user_name)
            print('User found: ' + self.user.username)
        except Exception as e:
            self.user = None
            print(e, 'User not found')
            return None
        
        # Total of images to download
        total_imgs = self.user.getPublicPhotos().info.total
        
        # Download user public photos using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            for _ in tqdm(
                executor.map(
                    partial(self.download_photo, dest_folder=dest_folder),
                    self.get_all_photos()),
                total=total_imgs,
                unit='imgs'
            ):
                pass
        
            
