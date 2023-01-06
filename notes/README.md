While in the project instructions it was recommended to keep a lab log, we resorted more to [GitHub projects](https://github.com/users/Turtilla/projects/2/views/1) for task management, and we kept track of the issues that arose in direct messages to each other. However, we will try to approximate the timeline of what we did here.  

+ early November: deciding to work together on a project, discussing options, trying to get access to different datasets (MS, DK).
+ Nov 18: getting started on the repository, documenting ideas (MS).
+ Nov 18-Dec 12: discussing and "fine-tuning" the ideas with the teachers (MS, DK).
+ Dec 12: adding some file structure, developing the initial dataloader, optimizing, adding pip requirements (DK).
+ Dec 13: adding multiple caption loading from imageCLEF, adding tqdm loading bars, creating a rudimentary relation filter, adding some error-handling, adding dataset splitting (MS)
+ Dec 14: adding ``models.py`` from the original code, updating readmes with information on how to access/download original data, fixing minor issues with Nikolai's code (filling in start and end tokens) (MS).
+ Dec 15: trying to make the training/fine-tuning loop work, importing ``preproc.py`` (MS).
+ Dec 17: optimization, fixing issues with BLEU calculation and checkpoint saving (DK).
+ Dec 19: adding the RuleBasedRelationFilter, splitting the code into separate files to avoid the notebook crashing, changing many pieces of code into classes to avoid other errors (DK).
+ Dec 20: trying to finetune and evaluate the models, running into the issue of the models not being able to generate full sequences (generating many UNK tokens) (MS).
+ Dec 21: exploring the UNK token issue (DK).
+ Dec 22: brainstoring caption generation solutions and discussing it with the teachers (DK, MS). Excluding the UNK token from caption generation (DK).
+ Dec 23: creating the UNK benchmark filter, updating how naming saved models works (MS).
+ Dec 25: adding code to access Flickr8k, moving code from notebook to separate files, cleaning the code, fixing image loading issues (DK).
+ Dec 26: fixing a number of errors when working on the server, fixing file strucure, preliminary model training (DK).
+ Dec 27: working on caption generation and evaluation, writing the bulk of doscstrings (MS).
+ Jan 2: training more models, starting to work on the paper, splitting the notebook for different models, creating the first version of the questionnaire (MS). Updating documentation (DK).
+ Jan 3: writing the abstract and introduction, updating bibliography, finishing and sending out the questionnaire (MS).
+ Jan 4: preliminary assignment of parts of the paper between Maria and Dominik (MS).
+ Jan 5: processing questionnaire results, writing about methods (questionnaire, imageCLEF), results (questionnaire), conclusions (MS).
+ Jan 6: updating the repository file structure, updating/adding readme files, updating the ``requirements.txt file``, adding statement of contribution to the paper (MS).
