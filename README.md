# MFAMatch
Multi-scale Feature Aggregation Match: Balancing shallow and deep features for multimodal image matching.

Multimodal image matching (MMIM), as a fundamental problem in computer vision, refers to identifying and then corresponding the same content from images of different modalities. Although deep learning-based MMIM methods show great potential in handling modality variance, they still confront a dilemma: coarse matching requires deep semantic information for initial correspondence, while fine matching requires shallow fine-grained features for precise pixel alignment. To tackle this challenge, we introduce the Multi-scale Feature Aggregation Match (MFAMatch) network. In MFAMatch, we begin by constructing consistent structural feature pyramids with the fully convolutional layers. To leverage both shallow and deep information while maintaining a certain degree of mutual independence, we aggregate the multi-scale features without lateral connection. Gradual fusion is then implemented on the independent features to encourage the interaction of multi-scale information. Finally, we utilize both the independent and fused features for similarity measurement. Moreover, a quintuplet soft-margin loss function is proposed to improve the explainability of the training process and the final matching accuracy. Extensive experiments show that our method achieves state-of-the-art performance on typical MMIM tasks such as SAR-visible image matching.

The test set for our experiment is in the link below: 
https://pan.baidu.com/s/1jgG01AhoqNZLjGV8mp5rSg?pwd=gg4c
Fetch code：gg4c 
