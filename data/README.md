Contains the [ImageCLEF/IAPR TC 12 Photo Collection](https://www.imageclef.org/photodata) as well as the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset. When running the code on the server, these can be accessed there (and the paths are in the code). 

Also contains the pre-trained (Flickr8k) model and the corresponding wordmap, sourced from the course folder on the mltgpu server using the following commands (run from command line when in the data folder):
+ MODEL: scp -P62266 XYZ@mltgpu.flov.gu.se:/srv/data/aics/03-image-captioning/data/BEST_checkpoint_flickr8k_5_10.pth.tar YOUR/FILE/PATH
+ WORDMAP: scp -P62266 XYZ@mltgpu.flov.gu.se:/srv/data/aics/03-image-captioning/data/out/wordmap_flickr8k_5_10.json YOUR/FILE/PATH
