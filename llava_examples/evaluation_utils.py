# prepare your result as a list named data
# each element data is a dict containing video_path, task_type, judge_result
# judge result is 1 for correct and 0 for incorrect
# for official VNbench, len(data)==5400
def get_detail_result(data):
    task_result = {}
    res = []
    processed_data = {}
    for d in data:
        if not 'judge_result' in d:
            d['judge_result'] = 1 if d['pred'][0]==d['gt'] else 0
        if d['video_path'] in task_result:
            task_result[d['video_path']] += d['judge_result']  
        else:
            task_result[d['video_path']] = d['judge_result']
        processed_data[d['video_path']] = {'video_path': d['video_path'], 
                                           'task_type': d['task_type']}
    for k, v in task_result.items():
        if v==4:
            processed_data[k]['result'] = 1
        else:
            processed_data[k]['result'] = 0
    result = list(processed_data.values())
    assert 4*len(result) == len(data)
    acc = {}
    sample_num = {}
    for res in result:
        if res['task_type'] in acc:
            acc[res['task_type']] += res['result']  
        else:
            acc[res['task_type']] = res['result']
        if res['task_type'] in sample_num:
            sample_num[res['task_type']] += 1  
        else:
            sample_num[res['task_type']] = 1
    res = {}
    res['ret'] = 0.0
    res['ord'] = 0.0
    res['cnt'] = 0.0
    for k, v in acc.items():
        acc[k] = v/sample_num[k]
        if 'ord' in k:
            res['ord'] += acc[k]
        if 'cnt' in k:
            res['cnt'] += acc[k]
        if 'ret' in k:  
            res['ret'] += acc[k]
    tmp = acc['ret_insert1']
    acc['ret_insert1'] = acc['ret_insert2']
    acc['ret_insert2'] = tmp
    acc = {key: acc[key] for key in sorted(acc.keys())}

    res['ord'] = res['ord']/3
    res['cnt'] = res['cnt']/3
    res['ret'] = res['ret']/3
    res = {key: res[key] for key in sorted(res.keys())}
    res['Overall'] = (res['ord']+res['cnt']+res['ret'])/3
    acc.update(res)
    

    return acc

import argparse
import json
import os
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", help="result path, jsonl format", required=True)
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    # if '.jsonl' in args.result_path:
    #     data = [json.loads(q) for q in open(os.path.expanduser(args.result_path), "r")]
    # else:
    #     data = json.load(open(args.result_path))
    data = [json.loads(q) for q in open(os.path.expanduser(args.result_path), "r")]
    print(get_detail_result(data))
