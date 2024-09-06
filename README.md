# FNDPro Repository
Repository for DASFAA 2024: [FNDPro: Evaluating the Importance of Propagations during Fake News Spread](https://link.springer.com/chapter/10.1007/978-981-97-5572-1_4)

# Basic Usage
FNDPro aims to address two main challenges in fake news detection: the **disguised content** and the **echo chamber phenomenon**. Thus FNDPro proposes to model the news propagation as a heterogeneous graph named **news propagation network** with multiple media and employs a **propagation transformer** module to evaluate the importance of propagation during news spread.
## Dataset & Preprocess
We use a modified [Fakenewsnet dataset](https://github.com/KaiDMML/FakeNewsNet?tab=readme-ov-file). Based on this original dataset, we collected more media related to the news spread: news, source, tweet, and user. Besides, we also collect five relations: *publish* relation between source and news; *comment* relation between comment and news; *post* relation between user and comment; *reply* relation between comment and comment; and *retweet* relation between comment and comment.

To ensure reproducibility, we make the dataset public on [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing). You could directly download the processed dataset and the dataset split.

To protect privacy and copyright, we only provide the data after feature extraction, and can only be used for research purposes.

## How to train FNDPro?
You could directly run train.py
```
python train.py --dataset politifact --mode GAT
```
Here we employ three settings of the dataset including PolitiFact, gossipcop, and mixed. for --mode, we provide four GNNs including RGCN, GCN, GAT, and HGT.

Besides vanilla FNDPro, we also provide some variants of FNDPro, where the propagation transformer module is replaced with other fusion methods including last, mean, max, mlp, rnn, and first:
```
python train.py --dataset politifact --ablation rnn
```

We also provide the ablation study of media, you could modify the parameters of function *obtain_hetero_data*, for example
```
data = obtain_hetero_data(medias=['news', 'tweet'], data_dir=data_dir)
```

# Citation
If you find our work interesting/helpful, please consider citing:
```
@inproceedings{wan2024fndpro,
  title={FNDPro: Evaluating the Importance of Propagations during Fake News Spread},
  author={Wan, Herun and Wang, Ningnan and Zhao, Xiang and Li, Rui and Yang, Hui and Luo, Minnan},
  booktitle={International Conference on Database Systems for Advanced Applications},
  pages={52--67},
  year={2024},
  organization={Springer}
}
```
# Questions?
Feel free to open issues in this repository! Instead of emails, Github issues are much better at facilitating a conversation between you and our team to address your needs. You can also contact Herun Wan through `wanherun@stu.xjtu.edu.cn`.


# Updating
### 20240819
- We uploaded the readme, providing a guideline to use FNDPro
### 20240818
- We uploaded the modified code of FNDPro, and now it can work with the data from the [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing)
- We uploaded the presentation of our paper.
- We plan to update the readme and provide a short guideline on how to train FNDPro.
- We plan to update the baseline code.
### 20240817
- We uploaded our processed data on the [Google Drive](https://drive.google.com/drive/folders/1TVcYw93JwesL6O5Z1RuDDasJ2-BfU8L1?usp=sharing)
### Before
- Our paper has been accepted to the DASFAA
- We uploaded the core code of FNDPro, but it's missing a lot of details.
