"""Optional observer-specific institution outside the frozen economic state."""
from copy import deepcopy

class DirectedCredibility:
    def __init__(self,company_ids,records=None):
        self.company_ids=tuple(company_ids);self.records=deepcopy(records or {})
    def belief(self,observer,proposer):
        if observer==proposer or observer not in self.company_ids or proposer not in self.company_ids:raise ValueError('invalid directed pair')
        r=self.records.get(f'{observer}>{proposer}',{'kept':0,'broken':0,'last_breach':-100})
        return dict(**r,credibility=(2+r['kept'])/(3+r['kept']+r['broken']))
    def accepts(self,observer,proposer,round_number):
        r=self.belief(observer,proposer)
        return r['credibility']>=.6 or (r['broken']>0 and round_number-r['last_breach']>=4)
    def verify(self,observer,proposer,promised,actual,round_number,*,accepted):
        if not accepted:return
        if type(promised) is not int or type(actual) is not int or promised<=0 or actual<0:raise ValueError('invalid observed commitment')
        r=self.belief(observer,proposer);r.pop('credibility')
        if actual>=promised:r['kept']+=1
        else:r['broken']+=1;r['last_breach']=round_number
        self.records[f'{observer}>{proposer}']=r
    def export(self):return dict(version='directed-credibility-v1',company_ids=list(self.company_ids),records=deepcopy(self.records))
    @classmethod
    def restore(cls,value):
        if value['version']!='directed-credibility-v1':raise ValueError('unknown institution version')
        return cls(value['company_ids'],value['records'])
