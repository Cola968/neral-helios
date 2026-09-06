"""Live own-model switching plus post-restart persistence and feature API checks."""
import json
from pathlib import Path
from live_check import call
def main():
    report={'initial':call('/api/info'),'checks':[]}
    try:
        for key,count in [('neral40m',40051200),('pilot_02',20587008)]:
            info=call('/api/model',{'checkpoint':key});assert info['parameters']==count and info['active_checkpoint']==key
            response=call('/api/generate',{'prompt':'Hallo Neral','max_new_tokens':32,'temperature':0});assert response['model_generated'] and response['active_checkpoint']==key
            report['checks'].append({'model':key,'parameters':count,'sha256':info['checkpoint_sha256'],'real_generation':response})
        eq=call('/api/generate',{'mode':'solve','prompt':'2*x + 3 = 11'});assert eq['solve']['result']=='x = 4' and not eq['model_generated'];report['checks'].append(eq)
        listing=call('/api/projects')['projects'];demo=next(p for p in listing if p['name']=='Beispiel · Budgetplanung');state=call('/api/projects/'+demo['id'])
        assert state['nodes']['budget']['value']=='1200'
        assert [v['text'] for v in state['nodes']['verfuegbar']['versions']]==['1080','1440','1080']
        impact=call('/api/projects/'+demo['id']+'/ripple',{'node':'budget'});assert len(impact['affected'])==4
        report['checks'].append({'project_id':demo['id'],'persisted_revision':state['revision'],'persisted_live_versions':state['nodes']['verfuegbar']['versions'],'ripple':impact})
    finally:call('/api/model',{'checkpoint':report['initial']['active_checkpoint']})
    Path('reports/live_http_03.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print('40M/20M actual inference, linear SOLVE and project persistence checks passed.')
if __name__=='__main__':main()
