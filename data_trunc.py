#!/usr/bin/env python3
import sys

def truncate(data_path: str, n_chunks: int, idx=0):
    """ Create a smaller dataset of the initial dataset, saved in `trunc_data_path`.
    
    arguments:
        data_path -- path to the dataset
        n_chunks -- number of partitions from the dataset. 
        idx -- index of the partition to keep
    """
    lines = open(data_path, 'r').readlines()
    n = len(lines)
    print('{} lines in original dataset'.format(n))
    chunk_size = round(len(lines) / int(n_chunks))
    print('{} lines in truncated dataset'.format(chunk_size))
    # get to idx block (so idx * x)
    start_id = idx * chunk_size
    output = lines[start_id:start_id+chunk_size]
    # write the next x lines in the output file (done)
    with open('trunc{}p_{}'.format(n_chunks, data_path), 'w') as out:
        for l in output:
            out.write(l)

def truncate_gowalla_integrity(data_path, rate=0.5, ustart=0):
    """Create a smaller dataset by truncating `rate` percent of the users. 

    arguments:
        data_path: path to the dataset.
        rate: rate of users to dismiss. 
        ustart: id of starting user
    """
    #count users
    lines = open(data_path, 'r').readlines()
    print('{} lines in original dataset'.format(len(lines)))
    prev_user = 0
    cnt = 0
    user2start_count = {}
    for i, line in enumerate(lines):
        tokens = line.split('\t')
        user = int(tokens[0])
        if user == prev_user:
            cnt += 1
        else: 
            user2start_count[prev_user] = (i-cnt, cnt)
            prev_user = user 
            cnt = 1
    
    # select n users (wrt rate)
    n_users  = round(len(user2start_count) * rate)
    new_size=0
    with open('{}{}.txt'.format(data_path.split('.')[0] + 'intr_urate', str(rate).split('.')[1]), 'w') as out:
        selected_u = 0
        u = ustart
        while selected_u < n_users: 
            try:
                start, cnt = user2start_count[u]
                new_size+=cnt
                for l in lines[start:start+cnt]:
                    out.write(l)
                selected_u += 1
            except KeyError:
                pass
            u += 1
    print('{} lines in new dataset'.format(new_size))


