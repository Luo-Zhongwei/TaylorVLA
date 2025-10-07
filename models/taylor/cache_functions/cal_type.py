# def cal_type(cache_dic, current):
#     '''
#     Determine calculation type for this step
#     '''
#     last_steps = (current['step'] <=2)
#     first_steps = (current['step'] > (current['num_steps'] - cache_dic['first_enhance'] - 1))

#     fresh_interval = cache_dic['interval']

#     if (first_steps) or (cache_dic['cache_counter'] == fresh_interval - 1 ):
#         current['type'] = 'full'
#         cache_dic['cache_counter'] = 0
#         current['activated_steps'].append(current['step'])
#         #current['activated_times'].append(current['t'])
    
#     else:
#         cache_dic['cache_counter'] += 1
#         current['type'] = 'Taylor'


def cal_type(cache_dic, current):
    step = current['step_idx']           # 0~4
    if step == 0 or step == current['num_steps'] - 1:
        current['type'] = 'full'         # F T T T F
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
    else:
        current['type'] = 'Taylor'