from .common import *
from .official_vqa_eval import VQAEval
from benchmarks.scoring import textvqa_score
class Dummy:
 def getQuesIds(self):return []
VQA=VQAEval(Dummy(),Dummy())
def normalize(text):return VQA.processDigitArticle(VQA.processPunctuation(text.replace('\n',' ').replace('\t',' ').strip()))
def consensus(pred,answers):
 assert len(answers)==10
 return sum(min(1.,sum(pred==x for j,x in enumerate(answers) if j!=i)/3.) for i in range(10))/10
def score(dataset,pred,answers):
 assert len(answers)==10
 if dataset=='textvqa':return float(textvqa_score(pred,answers))
 return consensus(normalize(pred),[normalize(x) for x in answers])
