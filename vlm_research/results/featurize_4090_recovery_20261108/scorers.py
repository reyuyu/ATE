from runtime_config import configure_scorers
scorer=configure_scorers()

def score(s,text):
    if s['benchmark']=='GQA':
        from vlm_research.gqa_testdev_c3.common import prediction
        p=prediction(text);return dict(score=float(p==s['answer']),parsed=p)
    if s['benchmark']=='OCRBench-v1':
        answers=s['answer'] if isinstance(s['answer'],list) else [s['answer']]
        def normal(x):
            return x.strip().replace('\n',' ').replace(' ','') if s['dataset_name']=='HME100k' else x.lower().strip().replace('\n',' ')
        p=normal(text);return dict(score=float(any(normal(a) in p for a in answers)),parsed=p,protocol='official-v1-substring')
    return scorer.score(s,text)
