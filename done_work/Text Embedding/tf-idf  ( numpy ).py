import numpy as np

videos_data = {
    'video_id': [f'video_{i+1}' for i in range(30)], # video_1, video_2, ..., video_30
    'title': [f'Example Video {i+1}' for i in range(30)],
    'category': [video_categories[i % len(video_categories)] for i in range(30)], # Cycle through categories
    'channel_id': [channel_names[i % len(channel_names)] for i in range(30)], # Assign videos to channels
    'topics': [
        ['Machine Learning', 'Tutorial'], ['Pop Music', 'New Release'], ['Gaming', 'Walkthrough'], ['World News', 'Analysis'],
        ['Data Science', 'Beginner'], ['Rock Music', 'Live Performance'], ['eSports', 'Tournament'], ['Tech News', 'Startups'],
        ['Python Programming', 'Libraries'], ['Classical Music', 'Symphony'], ['Basketball Highlights', 'NBA'], ['Cloud Computing', 'AWS'],
        ['Deep Learning', 'Neural Networks'], ['Indie Music', 'Music Video'], ['Game Reviews', 'Action Games'], ['US Politics', 'Debate'],
        ['Artificial Intelligence', 'Ethics'], ['Pop Music', 'Dance Remix'], ['Soccer Analysis', 'Champions League'], ['Travel Vlogs', 'Europe'],
        ['Space Exploration', 'Mars'], ['Rock Music', 'Classic Rock'], ['Game Reviews', 'Strategy Games'], ['US Politics', 'Election'],
        ['Quantum Physics', 'Theory'], ['Classical Music', 'Concerto'], ['Sports News', 'NFL'], ['DIY Projects', 'Home Improvement'],
        ['Food Recipes', 'Italian'], ['Baking Tips', 'Cakes']
    ]
}

videos_df = pd.DataFrame(videos_data) 
def tf_idf(documents): 
    def tf(t,d):
        d = d.split(" ")
        return d.count(t) / len(d)

    def idfing(t,d): 
        N = len(d)
        have_doc = 0 
        for d_ in d : 
            have_doc += int(t in d_.split(" "))
        return np.log( (N + 1) / (1 + have_doc)) + 1 
    
    temp_dic = {} 
    for doc_index , sub_doc in enumerate(documents): 

        for text_index , text in enumerate(np.unique(sub_doc.split(" "))):
            Tf = tf(text,sub_doc)   
            idf = idfing(text,documents)
            values = Tf * idf
            temp_dic[(doc_index , text_index)] = np.float32(values)  

    return temp_dic 

doc = list(videos_df['title'] + " " + videos_df['category'] + ' ' + videos_df['topics'].apply(' '.join)) 
doc = ["apple banana apple", "banana orange", "apple orange"]
tf_idf_dic = tf_idf(doc)
for key , values in tf_idf_dic.items(): 
    print(key,values)
