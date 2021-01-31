# GEDet

This document is for the project "GEDet: Adversarially Learned Few-shot Detection of Erroneous Nodes in Graphs", which is published on IEEE BigData 2020 as a regular paper.

To run the classification modular, simply run train.py
train.py needs the following input:
(1) A pre-trained graph embedding result from graph auto-encoder that stands for the (X'_L, A_L) in the paper. To balance the class labels in the training examples, X'_L may contain embeddings for instances that are generated by node augmentation (optional)
(2) A pre-trained graph embedding result from graph auto-encoder that stands for the (X'_S, A_L) in the paper. X'_S is the embeddings results after node augmentation
In the above two load functions, an augmented graph adjacency matrix A_L is saved and directly loaded.
The pre-trained results are saved in GEDet/processed/

GEDet now has a web interface availabel. 
The web interface is：
https://gdet.hcma.repl.co/

Related Repo:  https://github.com/CWRU-DB-Group/GDet

There are related videos to introduce GEDet:

Youtube:
https://www.youtube.com/watch?v=GGSs7wcKaBU

https://youtube.com/playlist?list=PL0YKREc7vLdWfX3lXplYvpiklA8zAFEfS

Bilibili:

https://www.bilibili.com/s/video/BV1ET4y1M7F1

GEDet consists of several components and this repository will be updated and maintained by Sheng Guan.
Please feel free to send emails to sxg967@case.edu if you have any questions.
Any contribution is highly appreciated!
