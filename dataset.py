import torch
import json
from torch_geometric.data import HeteroData


# this function is to load the data,
# where medias denote the medias you want to use, including news, source, tweet, and user
# if medias is none, means using all related medias
def obtain_hetero_data(medias=None, data_dir='data'):
    # loading news related data
    news_content = torch.load(f'{data_dir}/news_content.pt')
    news_title = torch.load(f'{data_dir}/news_title.pt')
    news_time = torch.load(f'{data_dir}/news_time.pt')
    news_label = torch.load(f'{data_dir}/news_label.pt')

    # loading source related data
    source_description = torch.load(f'{data_dir}/source_description.pt')
    source_time = torch.load(f'{data_dir}/source_time.pt')

    # loading user related data
    user_description = torch.load(f'{data_dir}/user_description.pt')
    user_profile = torch.load(f'{data_dir}/user_profile.pt')
    user_time = torch.load(f'{data_dir}/user_time.pt')

    # loading tweet related data
    tweet_content = torch.load(f'{data_dir}/tweet_content.pt')
    tweet_profile = torch.load(f'{data_dir}/tweet_profile.pt')
    tweet_time = torch.load(f'{data_dir}/tweet_time.pt')

    # loading relation data
    post_edge = json.load(open(f'{data_dir}/post_edge.json'))
    post_edge = torch.tensor(post_edge, dtype=torch.long).T  # user post tweet
    publish_edge = json.load(open(f'{data_dir}/publish_edge.json'))
    publish_edge = torch.tensor(publish_edge, dtype=torch.long).T  # source publish news
    comment_edge = json.load(open(f'{data_dir}/comment_edge.json'))
    comment_edge = torch.tensor(comment_edge, dtype=torch.long).T  # tweet comment news
    reply_edge = json.load(open(f'{data_dir}/reply_edge.json'))
    reply_edge = torch.tensor(reply_edge, dtype=torch.long).T  # tweet reply tweet
    retweet_edge = json.load(open(f'{data_dir}/retweet_edge.json'))
    retweet_edge = torch.tensor(retweet_edge, dtype=torch.long).T  # tweet retweet tweet

    data = HeteroData()
    data['news'].content = news_content
    data['news'].title = news_title
    data['news'].time = news_time
    data['news'].label = news_label

    data['source'].description = source_description
    data['source'].time = source_time

    data['user'].description = user_description
    data['user'].profile = user_profile
    data['user'].time = user_time

    data['tweet'].content = tweet_content
    data['tweet'].profile = tweet_profile
    data['tweet'].time = tweet_time

    data['news'].num_nodes = news_content.shape[0]
    data['source'].num_nodes = source_description.shape[0]
    data['tweet'].num_nodes = tweet_content.shape[0]
    data['user'].num_nodes = user_description.shape[0]

    data['user', 'post', 'tweet'].edge_index = post_edge
    data['source', 'publish', 'news'].edge_index = publish_edge
    data['tweet', 'comment', 'news'].edge_index = comment_edge
    data['tweet', 'reply', 'tweet'].edge_index = reply_edge
    data['tweet', 'retweet', 'tweet'].edge_index = retweet_edge

    # setting for the ablation study
    if medias is not None:
        if 'news' not in medias:
            data['news'].content = news_content.fill_(0)
            data['news'].title = news_title.fill_(0)
            data['news'].time = news_time.fill_(0)

        if 'source' not in medias:
            data['source'].description = source_description.fill_(0)
            data['source'].time = source_time.fill_(0)

        if 'user' not in medias:
            data['user'].description = user_description.fill_(0)
            data['user'].profile = user_profile.fill_(0)
            data['user'].time = user_time.fill_(0)

        if 'tweet' not in medias:
            data['tweet'].content = tweet_content.fill_(0)
            data['tweet'].profile = tweet_profile.fill_(0)
            data['tweet'].time = tweet_time.fill_(0)

    return data


if __name__ == '__main__':
    out = obtain_hetero_data()
    print(out)
