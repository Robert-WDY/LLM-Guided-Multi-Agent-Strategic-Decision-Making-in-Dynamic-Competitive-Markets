"""Restore matched model drafts after bounded format failures; keep superseded evidence."""
from study import RAW,OUT,save,read
from paid import initialize
from paid_confirm import case,report as base_report
from paid_units import job,report as units_report
from prompt_audit import check


if __name__=='__main__':
    initialize()
    before=check()
    archive=RAW/'superseded-model-drafts'
    archive.mkdir(exist_ok=True)
    receipt=archive/'before.json'
    if receipt.exists():
        before=read(receipt)
    else:
        save(receipt,before)
    tasks=set()
    for name in before['affected']:
        path=RAW/name
        family,p,m,s,arm=name.split('-')
        # Route outputs through case/job's normal atomic create path.
        target=archive/name
        if not target.exists():
            path.rename(target)
        elif path.exists():
            continue
        tasks.add((family,p,m,int(s)))
    for family,p,m,s in sorted(tasks):
        (case if family=='real2' else job)((p,m,s))
    base_report()
    units_report()
    after=check()
    assert not after['affected'] and not after['multi_draft_histories']
    save(OUT/'prompt-audit.json',dict(before=before,after=after))

