import torch
import json
from torch_geometric.data import HeteroData


def get_data():
    news_content = torch.load('../processed_data/news_content.pt')
    news_title = torch.load('../processed_data/news_title.pt')
    news_time = torch.load('../processed_data/news_time.pt')
    news_label = torch.load('../processed_data/news_label.pt')

    source_description = torch.load('../processed_data/source_description.pt')
    source_time = torch.load('../processed_data/source_time.pt')

    user_description = torch.load('../processed_data/user_description.pt')
    user_profile = torch.load('../processed_data/user_profile.pt')
    user_time = torch.load('../processed_data/user_time.pt')

    tweet_content = torch.load('../processed_data/tweet_content.pt')
    tweet_profile = torch.load('../processed_data/tweet_profile.pt')
    tweet_time = torch.load('../processed_data/tweet_time.pt')

    post_edge = json.load(open('../processed_data/post_edge.json'))
    post_edge = torch.tensor(post_edge, dtype=torch.long).T  # user post tweet
    publish_edge = json.load(open('../processed_data/publish_edge.json'))
    publish_edge = torch.tensor(publish_edge, dtype=torch.long).T  # source publish news
    comment_edge = json.load(open('../processed_data/comment_edge.json'))
    comment_edge = torch.tensor(comment_edge, dtype=torch.long).T  # tweet comment news
    reply_edge = json.load(open('../processed_data/reply_edge.json'))
    reply_edge = torch.tensor(reply_edge, dtype=torch.long).T  # tweet reply tweet
    retweet_edge = json.load(open('../processed_data/retweet_edge.json'))
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

    # data['tweet', 'rpost', 'user'].edge_index = torch.stack([post_edge[1], post_edge[0]])
    # data['news', 'rpublished', 'source'].edge_index = torch.stack([publish_edge[1], publish_edge[0]])
    # data['news', 'rcomment', 'tweet'].edge_index = torch.stack([comment_edge[1], comment_edge[0]])
    # data['tweet', 'rreply', 'tweet'].edge_index = torch.stack([reply_edge[1], reply_edge[0]])
    # data['tweet', 'rretweet', 'tweet'].edge_index = torch.stack([retweet_edge[1], retweet_edge[0]])
    return data


def get_mediums_data(mediums):
    news_content = torch.load('../processed_data/news_content.pt')
    news_title = torch.load('../processed_data/news_title.pt')
    news_time = torch.load('../processed_data/news_time.pt')
    news_label = torch.load('../processed_data/news_label.pt')

    source_description = torch.load('../processed_data/source_description.pt')
    source_time = torch.load('../processed_data/source_time.pt')

    user_description = torch.load('../processed_data/user_description.pt')
    user_profile = torch.load('../processed_data/user_profile.pt')
    user_time = torch.load('../processed_data/user_time.pt')

    tweet_content = torch.load('../processed_data/tweet_content.pt')
    tweet_profile = torch.load('../processed_data/tweet_profile.pt')
    tweet_time = torch.load('../processed_data/tweet_time.pt')

    post_edge = json.load(open('../processed_data/post_edge.json'))
    post_edge = torch.tensor(post_edge, dtype=torch.long).T  # user post tweet
    publish_edge = json.load(open('../processed_data/publish_edge.json'))
    publish_edge = torch.tensor(publish_edge, dtype=torch.long).T  # source publish news
    comment_edge = json.load(open('../processed_data/comment_edge.json'))
    comment_edge = torch.tensor(comment_edge, dtype=torch.long).T  # tweet comment news
    reply_edge = json.load(open('../processed_data/reply_edge.json'))
    reply_edge = torch.tensor(reply_edge, dtype=torch.long).T  # tweet reply tweet
    retweet_edge = json.load(open('../processed_data/retweet_edge.json'))
    retweet_edge = torch.tensor(retweet_edge, dtype=torch.long).T  # tweet retweet tweet

    data = HeteroData()

    data['news'].content = news_content
    data['news'].title = news_title
    data['news'].time = news_time
    data['news'].label = news_label
    if 'news' not in mediums:
        data['news'].content = news_content.fill_(0)
        data['news'].title = news_title.fill_(0)
        data['news'].time = news_time.fill_(0)

    data['source'].description = source_description
    data['source'].time = source_time

    if 'source' not in mediums:
        data['source'].description = source_description.fill_(0)
        data['source'].time = source_time.fill_(0)

    data['user'].description = user_description
    data['user'].profile = user_profile
    data['user'].time = user_time

    if 'user' not in mediums:
        data['user'].description = user_description.fill_(0)
        data['user'].profile = user_profile.fill_(0)
        data['user'].time = user_time.fill_(0)

    data['tweet'].content = tweet_content
    data['tweet'].profile = tweet_profile
    data['tweet'].time = tweet_time

    if 'tweet' not in mediums:
        data['tweet'].content = tweet_content.fill_(0)
        data['tweet'].profile = tweet_profile.fill_(0)
        data['tweet'].time = tweet_time.fill_(0)

    data['news'].num_nodes = news_content.shape[0]
    data['source'].num_nodes = source_description.shape[0]
    data['tweet'].num_nodes = tweet_content.shape[0]
    data['user'].num_nodes = user_description.shape[0]

    data['user', 'post', 'tweet'].edge_index = post_edge
    data['source', 'publish', 'news'].edge_index = publish_edge
    data['tweet', 'comment', 'news'].edge_index = comment_edge
    data['tweet', 'reply', 'tweet'].edge_index = reply_edge
    data['tweet', 'retweet', 'tweet'].edge_index = retweet_edge
    # data['tweet', 'rpost', 'user'].edge_index = torch.stack([post_edge[1], post_edge[0]])
    # data['news', 'rpublished', 'source'].edge_index = torch.stack([publish_edge[1], publish_edge[0]])
    # data['news', 'rcomment', 'tweet'].edge_index = torch.stack([comment_edge[1], comment_edge[0]])
    # data['tweet', 'rreply', 'tweet'].edge_index = torch.stack([reply_edge[1], reply_edge[0]])
    # data['tweet', 'rretweet', 'tweet'].edge_index = torch.stack([retweet_edge[1], retweet_edge[0]])
    return data


def tmp(mediums):
    news_content = torch.load('../processed_data/news_content.pt')
    news_title = torch.load('../processed_data/news_title.pt')
    news_time = torch.load('../processed_data/news_time.pt')
    news_label = torch.load('../processed_data/news_label.pt')

    source_description = torch.load('../processed_data/source_description.pt')
    source_time = torch.load('../processed_data/source_time.pt')

    user_description = torch.load('../processed_data/user_description.pt')
    user_profile = torch.load('../processed_data/user_profile.pt')
    user_time = torch.load('../processed_data/user_time.pt')

    tweet_content = torch.load('../processed_data/tweet_content.pt')
    tweet_profile = torch.load('../processed_data/tweet_profile.pt')
    tweet_time = torch.load('../processed_data/tweet_time.pt')

    post_edge = json.load(open('../processed_data/post_edge.json'))
    post_edge = torch.tensor(post_edge, dtype=torch.long).T  # user post tweet
    publish_edge = json.load(open('../processed_data/publish_edge.json'))
    publish_edge = torch.tensor(publish_edge, dtype=torch.long).T  # source publish news
    comment_edge = json.load(open('../processed_data/comment_edge.json'))
    comment_edge = torch.tensor(comment_edge, dtype=torch.long).T  # tweet comment news
    reply_edge = json.load(open('../processed_data/reply_edge.json'))
    reply_edge = torch.tensor(reply_edge, dtype=torch.long).T  # tweet reply tweet
    retweet_edge = json.load(open('../processed_data/retweet_edge.json'))
    retweet_edge = torch.tensor(retweet_edge, dtype=torch.long).T  # tweet retweet tweet

    idx = torch.arange(comment_edge.shape[1], dtype=torch.long)
    idx[comment_edge[1]] = comment_edge[0]
    user_comment_news = torch.stack([idx[comment_edge[0]], comment_edge[1]])

    data = HeteroData()

    data['news'].content = news_content
    data['news'].title = news_title
    data['news'].time = news_time
    data['news'].label = news_label
    if 'news' not in mediums:
        data['news'].content = news_content.fill_(0)
        data['news'].title = news_title.fill_(0)
        data['news'].time = news_time.fill_(0)

    data['source'].description = source_description
    data['source'].time = source_time

    # if 'source' not in mediums:
    #     data['source'].description = source_description.fill_(0)
    #     data['source'].time = source_time.fill_(0)

    data['user'].description = user_description
    data['user'].profile = user_profile
    data['user'].time = user_time

    # if 'user' not in mediums:
    #     data['user'].description = user_description.fill_(0)
    #     data['user'].profile = user_profile.fill_(0)
    #     data['user'].time = user_time.fill_(0)

    data['tweet'].content = tweet_content
    data['tweet'].profile = tweet_profile
    data['tweet'].time = tweet_time

    # if 'tweet' not in mediums:
    #     data['tweet'].content = tweet_content.fill_(0)
    #     data['tweet'].profile = tweet_profile.fill_(0)
    #     data['tweet'].time = tweet_time.fill_(0)

    data['news'].num_nodes = news_content.shape[0]
    data['source'].num_nodes = source_description.shape[0]
    data['tweet'].num_nodes = tweet_content.shape[0]
    data['user'].num_nodes = user_description.shape[0]

    data['user', 'post', 'tweet'].edge_index = post_edge
    data['source', 'publish', 'news'].edge_index = publish_edge
    data['tweet', 'comment', 'news'].edge_index = comment_edge
    data['tweet', 'reply', 'tweet'].edge_index = reply_edge
    data['tweet', 'retweet', 'tweet'].edge_index = retweet_edge

    if 'tweet' not in mediums:
        data['user', 'user_comment', 'news'] = user_comment_news
    # data['tweet', 'rpost', 'user'].edge_index = torch.stack([post_edge[1], post_edge[0]])
    # data['news', 'rpublished', 'source'].edge_index = torch.stack([publish_edge[1], publish_edge[0]])
    # data['news', 'rcomment', 'tweet'].edge_index = torch.stack([comment_edge[1], comment_edge[0]])
    # data['tweet', 'rreply', 'tweet'].edge_index = torch.stack([reply_edge[1], reply_edge[0]])
    # data['tweet', 'rretweet', 'tweet'].edge_index = torch.stack([retweet_edge[1], retweet_edge[0]])

    sample = HeteroData({item: data[item] for item in mediums})

    for edge_type, edge_index in data.edge_index_dict.items():
        dst_type, _, src_type = edge_type
        if dst_type in mediums and src_type in mediums:
            sample[edge_type].edge_index = edge_index
    print(sample)
    return data


if __name__ == '__main__':
    data = get_data()
    print(data['news'].label)
    print(len(data['news'].label))
    print(data['news'].label.sum(0))
