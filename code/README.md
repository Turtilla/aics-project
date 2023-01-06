In this folder you can find both our code and code taken from the [course repository](https://github.com/sdobnik/aics/tree/master/tutorials/03-image-captioning/2022)  

The origin of the code is marked in the respective files. The following files can be found here:
+ ``captioning_XYZ.ipynb``: our evaluation notebooks; they are split into multiple parts, with the 2k_5k including the evaluation of the original model too. This is because the .ipynb files would be too large if they had not been split. Evaluation is based on Nikolai's code, but adapted by Maria.
+ ``dataset.py``: dataset and collate classes/functions suited for the datasets we are working with. By Dominik.
+ ``filters.py``: two different relation filters; one originally by Maria, one from [CLASP](https://github.com/GU-CLASP/spatial_relations_vectors_sltc2018). Both of them transformed into classes by Dominik.
+ ``main.py``: fine-tuning loop; this can (should) be run from the command line using appropriate arguments. It will create a training dataset of the desired size and unknown filter, and fine-tune Nikolai's model on that. Parts of the code here are by Maria, parts by Dominik, parts taken from Nikolai's code.
+ ``models.py``: taken straight from Nikolai's code, specifies the encoder and decoder models used in the project.
+ ``preproc.py``: taken straight from Nikolai's code, includes some of the preprocessing code that we wanted to make use of.
+ ``test.py``: predominantly Nikolai's code from his testing notebook that allows for the generation and visualization of captions and attention. Turned into a class by Dominik, minor changes by Dominik and Maria to make the code work and do what we needed it to do.
+ ``train.py``: predominantly Nikolai's code, adapted by Dominik, with small changes marked in the code.
