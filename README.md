# AICS project 
### by Dominik Künkele and Maria Irena Szawerna

### (i) REPOSITORY STRUCTURE:  
Each folder in this repository contains its own README. In case it should contain some datasets or other large files, the instructions on how to obtain them or where to obtain them from are in the README for that folder. All of those files are also available on MLTGPU and when running the code there, they should be accessible.

+ ``code/``: contains all of the code for this project, in .py and .ipynb files. A detailed description of the files is available in the folder, along with who created what (which is also specified in detail in file documentation).
+ ``data/``: contains a README file specifying how and where to get the datasets used in the project. Does **NOT** contain saved models (see: ``saved/``).
+ ``library/``: contains the requirements for running the project and instructions on how to install those, along with advice on virtual environments.
+ ``notes/``: contains a rough timeline of our work on the project and division of labor.
+ ``paper/``: contains the project paper files and the final paper.
+ ``saved/``: contains the outputs of our file training and instructions on how to download the fine-tuned models.  

### (ii) INSTRUCTIONS:
+ This project makes use of the [ImageCLEF/IAPR TC 12 Photo Collection](https://www.imageclef.org/photodata) and [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) datasets. Please download them and store them in the ``data/`` folder if running the code locally.
+ We recommend using a virtual environment to install all the necessary dependencies (see ``library/requirements.txt``). 
+ When running the code locally you will also need to download the original model and wordmap (as per the README in ``data/`` as well as (potentially) the fine-tuned models (see ``saved/``).  

While more options are hidden in the code (different relation filters, for example), you will mainly need to run the ``main.py`` script from the command line to fine-tune the original model. Only do this if you want to fine-tune models with different data size, otherwise, download them from MLTGPU (as specified in ``saved/``).
+ For running on the server: ``python main.py --number_of_images IJK --unknown_filter X --on_server true``, where IJK is the number of images before the train/val/test split, X is a fraction representation of the maximum of UNK tokens per caption (e.g. 0.1 = max 10% UNK tokens).
+ For running locally: ``python main.py --number_of_images IJK --unknown_filter X``

The "evaluation" of the success of fine-tuning can be explored in captioning_IJK.ipynb files, where IJK stands for the data size. You should be able to run the code in the notebooks without issues if you have downloaded the necessary datasets, models, and wordmaps, as specified in the relevant READMEs (and earlier stages of these instructions). NOTE: in cell 2, change ``on_server`` to ``True`` if running the notebook on the server.

### (iii) WORKPLAN, TEACHER COMMENTS:
A more detailed timeline can be found in ``notes/``. The general idea of this project is testing how training data size influences the quality of captions when finetuning a CNN+LSTM-based image captioning model. Dominik's research question is focused on where the generation (and the model) fall apart when fine-tuning, and Maria's on how the generated captions are evaluated by humans.   

The general workplan:
+ test and adapt the original code (by Nikolai) if needed
+ develop code needed to complete our goals (loading a different dataset, fine-tuning, evaluation)
+ construct a questionnaire, gather responses
+ analyse the results, write the paper  

Teacher comments:
