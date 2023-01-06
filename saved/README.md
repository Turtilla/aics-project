This folder contains saved models, checkpoints and other files that were output running the code. While all of them are accessible in Dominik's MLTGPU directory, some of them were trained by him (2k and 5k) and some by Maria (the rest). 

Naturally, the models are too large to be included here. The fine-tuned models are available on the server and should be accessible when running the notebook on the server. For accessing them locally, use the following command (if you have access to the server):
+ ``scp -P62266 XYZ@mltgpu.flov.gu.se:/srv/data/guskunkdo/saved/FILENAME YOUR/FILE/PATH``  

The filenames are constructed the following way:
+ (optional) ``BEST`` to denote that it exceeded the performance of the original model on the BLEU score.
+ ``checkpoint`` to denote it is, well, a checkpoint.
+ ``clef`` or ``imageCLEF`` to denote it has been finetuned on that dataset.
+ ``100`` to ``5000`` to denote how many images were sourced to finetune that model.
+ ``0.1`` to ``1.0`` to denote the % of UNK tokens allowed in the captions.

Choose from the following filenames:
+ BEST_checkpoint_clef_2000_0.1.pth.tar   
+ BEST_checkpoint_clef_2000_0.15.pth.tar  
+ BEST_checkpoint_clef_2000_0.2.pth.tar 
+ checkpoint_clef_2000_1.0.pth.tar
+ BEST_checkpoint_clef_5000_0.1.pth.tar   
+ BEST_checkpoint_clef_5000_0.15.pth.tar  
+ BEST_checkpoint_clef_5000_0.2.pth.tar 
+ BEST_checkpoint_clef_5000_1.0.pth.tar  
+ checkpoint_clef_1000_0.1.pth.tar   
+ checkpoint_clef_1000_0.15.pth.tar  
+ checkpoint_clef_1000_0.2.pth.tar 
+ checkpoint_clef_1000_1.0.pth.tar 
+ checkpoint_clef_500_0.1.pth.tar   
+ checkpoint_clef_500_0.15.pth.tar  
+ checkpoint_clef_500_0.2.pth.tar 
+ checkpoint_clef_500_1.0.pth.tar  
+ checkpoint_clef_200_0.1.pth.tar   
+ checkpoint_clef_200_0.15.pth.tar  
+ checkpoint_clef_200_0.2.pth.tar 
+ checkpoint_clef_200_1.0.pth.tar 
+ checkpoint_clef_100_0.1.pth.tar   
+ checkpoint_clef_100_0.15.pth.tar  
+ checkpoint_clef_100_0.2.pth.tar 
+ checkpoint_clef_100_1.0.pth.tar 

