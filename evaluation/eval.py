import argparse
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="evaluation/example_result.jsonl")
args = parser.parse_args()
annos = [json.loads(q) for q in open(os.path.expanduser(args.path), "r")]
res = {}
for anno in annos:
    name = anno['question_id'][:-2]
    label = anno['type']
    if anno['pred'] is None:
        continue
    if not label in res:
        res[label] = []
    if anno['gt'] in [0, 1, 2, 3]:
        anno['gt'] = chr(ord('A') + anno['gt'])
    anno['pred'] = anno['pred'].split('.')[0]
    dic = {
            'name': name,
            'gt': anno['gt'],
            'pred': anno['pred'],
    }
    if "gpt_judge" in anno:
        dic['judge'] = anno['gpt_judge'][0]
    res[label].append(dic)

RES = {}
result = {}
sorted_items = sorted(res.items(), key=lambda x: x[0])
for k, vv in sorted_items:
    acc = {}
    for v in vv:
        name = v['name']
        if not name in acc:
            acc[name] = 0  
        if 'judge' in v:
            acc[name] += (v['judge']=='1')
        else:
            pred = v['pred']
            if 'A' in pred:
                pred = 'A'
            elif 'B' in pred:
                pred = 'B'
            elif 'C' in pred:
                pred = 'C'
            elif 'D' in pred:
                pred = 'D'
            acc[name] += (v['gt']==pred)
    accuracy = 0
    for n, ac in acc.items():
        if ac==4:
            accuracy += 1
    st = f'true: {accuracy}, total: {len(acc)}, acc: {accuracy/len(acc)}'
    RES[k] = st
    result[k] = accuracy/len(acc)
RES_list = []
for k, v in result.items():
    print(k)
    print(RES[k])
    RES_list.append(result[k])
print('Overall: ', sum(RES_list)/len(RES_list))
