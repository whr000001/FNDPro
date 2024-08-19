# FNDPro Repository
Repository for DASFAA 2024: FNDPro: Evaluating the Importance of Propagations during Fake News Spread

We plan to upload the completed code before August 20th.

# Basic Usage
FNDPro aims to address two main challenges in fake news detection: the **disguised content** and the **echo chamber phenomenon**. Thus FNDPro proposes to model the news propagation as a heterogeneous graph named **news propagation network** with multiple media and employs a **propagation transformer** module to evaluate the importance of propagation during news spread.
## Dataset & Preprocess
We use a modified [Fakenewsnet dataset](https://github.com/KaiDMML/FakeNewsNet?tab=readme-ov-file). Based on this original dataset, we collected more media related to the news spread: news, source, tweet, and user. Besides, we also collect five relations: *publish* relation between source and news; *comment* relation between comment and news; *post* relation between user and comment; *reply* relation between comment and comment; and *retweet* relation between comment and comment.

To ensure reproducibility, we make the dataset public on [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing). 

To protect privacy and copyright, we only provide the data after feature extraction, and can only be used for research purposes.



# Updating
### 20340818
- We uploaded the modified code of FNDPro, and now it can work with the data from the [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing)
- We uploaded the presentation of our paper.
- We plan to update the readme and provide a short guideline on how to train FNDPro.
- We plan to update the baseline code.
### 20240817
- We uploaded our processed data on the [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing)
### Before
- Our paper has been accepted to the DASFAA
- We uploaded the core code of FNDPro, but it's missing a lot of details.
