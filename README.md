# GraphRfi
This is an implementation for the paper:

Shijie Zhang, Hongzhi Yin*, Tong Chen, Nguyen Quoc Viet Hung, Zi Huang and Lizhen Cui. "GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Identification". In 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval(SIGIR'2020), Xian, China. July, 2020. Preprint[https://arxiv.org/pdf/2005.10150.pdf]
# Abstract
In recent years, recommender system has become an indispensable function in all e-commerce platforms. The review rating data for a recommender system typically comes from open platforms, which may attract a group of malicious users to deliberately insert fake feedback in an attempt to bias the recommender system to their favour. The presence of such attacks may violate modeling assumptions that high-quality data is always available and these data truly reflect users’ interests and preferences. Therefore, it is of great practical significance to construct a robust recommender system that is able to generate stable recommendations even in the presence of shilling attacks. In this paper, we propose GraphRfi - a GCN-based user representation learning framework to perform robust recommendation and fraudster detection in a unified way. In its end-to-end learning process, the probability of a user being identified as a fraudster in the fraudster detection component automatically determines the contribution of this user’s rating data in the recommendation component; while the prediction error outputted in the recommendation component acts as an important feature in the fraudster detection component. Thus, these two components can mutually enhance each other. Extensive experiments have been conducted and the experimental results show the superiority of our GraphRfi in the two tasks - robust rating prediction and fraudster detection. Furthermore, the proposed GraphRfi is validated to be more robust to the various types of shilling attacks over the state-of-the-art recommender systems.
# The framework of our model
The GraphRfi framework consists of two key components to perform robust rating prediction and fraudster detection respectively. In these two components, we adopt the
graph convolutional network (GCN) and neural random forest (NRF) as the building blocks, because the GCN is capable of fully exploiting local structure information of the rating graph G and the user side information xu to capture both user preferences and reliability in a unified way, and the NRF has achieved outstanding performance in a variety of classification tasks. Note that these two components are closely coupled and mutually enhanced with each other, making GraphRfi an end-to-end learning framework. First, these two components share the same user embeddings. Second, the probability of a user being classified as a fraudster by the NRF component is taken as a weight to control the contribution of this user’s generated rating data in the GCN component, while the aggregated prediction error of the user’s ratings is taken as an important feature for the fraudster detection component (i.e., the NRF).
# Code
If you use this code, please cite our paper:
@article{zhang2020gcn,
  title={GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection},
  author={Zhang, Shijie and Yin, Hongzhi and Chen, Tong and Hung, Quoc Viet Nguyen and Huang, Zi and Cui, Lizhen},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2020}
}
run_GraphRfi_example.py only for recommendation robust test. 

To generate shilling attack, the fake users and correspondingly fake ratings can be manipulated based on the rule (e.g. hate, random and average) introduced in the experiment part.

To test the auxiliary task (i.e. fraudster detection), the user set should be split in two sets: training set and testing set. And both sets include fraudster and genuine users. The testing users don't need to be trained in the NRF component by simply setting $L_fraudster$ as $0$, or set a variable $flag \in {0,1}$ to control training or not and then setting the loss as $flag*L_frudster$. 

# Dataset
Yelp: [Google Drive](https://drive.google.com/drive/folders/0B8JIKvhJUvRdfk8yS1E4T0lXUm1uOGtJUmN2cExMTXRmVUpsSGE2OHRzNkdUT0RyMzA4WDA)
# Acknolwedgement
The original version of this code base was from GraphRec and NRF. Many thanks to Dr Wenqi Fan for making his code available.











