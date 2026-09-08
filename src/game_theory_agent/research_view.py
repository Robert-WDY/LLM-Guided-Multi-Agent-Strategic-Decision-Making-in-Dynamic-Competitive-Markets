"""Read-only controller projection; no recomputation of past model reasoning."""
def build_research_view(session, round_number=None):
    transitions=session.transitions
    choices=[t.step_result.settled_round for t in transitions]
    selected=round_number if round_number is not None else (choices[-1] if choices else None)
    if selected is not None and selected not in choices: raise ValueError('该回合尚未结算或不存在。')
    timeline=[]
    for transition in transitions:
        s=transition.state_after
        timeline.append({'round':transition.step_result.settled_round,'companies':{c.company_id:{'cash_cents':c.financial.cash_balance_cents,'profit_cents':c.financial.round_profit_cents,'price_cents':c.commercial.price_cents,'sales_orders':c.commercial.sales_orders} for c in s.companies},'state_hash':s.state_hash})
    detail=None
    if selected is not None:
        transition=next(t for t in transitions if t.step_result.settled_round==selected)
        event=None
        for _,payload in session.coordinator_runs.values():
            for item in payload.get('rounds',[]):
                if item['settled_round']==selected: event=item.get('research_event')
        # Old checkpoints lack trace capture. Never manufacture posterior/model text.
        messages=[]
        for key,ledger in session.communication_ledgers.items():
            if key[0]==selected and getattr(ledger,'_closure',None) is not None:
                messages=[m.model_dump(mode='json') for m in ledger._closure.all_messages]
        detail={'round':selected,'state_before':transition.state_before.to_dict(),'state_after':transition.state_after.to_dict(),'final_actions':{k:v.to_dict() for k,v in transition.joint_action},'traces':event.get('traces',[]) if event else [],'messages':messages,'cooperation':event.get('cooperation_round') if event else None,'trace_available':event is not None,'invariants':list(transition.step_result.invariant_results)}
    return {'schema':'controller-research-view-v1','episode_id':session.env.get_state().episode_id,'manifest':session.manifest.to_dict(),'rounds':choices,'selected_round':selected,'timeline':timeline,'detail':detail,'visibility':'controller_full_state; recorded agent observations retain their original scope'}
