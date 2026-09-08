import asyncio
from types import SimpleNamespace
from game_theory_agent.agents.contracts import AgentRequestedAction
from game_theory_agent.model_clients.fixed_action import FixedActionModelClient

def test_human_supplier_choice_and_quantity_survive_fixed_client():
    request=AgentRequestedAction(price_cents=9800,primary_supplier_id='resilient_supplier',backup_supplier_id='economy_supplier',primary_supplier_share_ppm=600000,procurement_quantity_orders=1234)
    bounds={k:{'min':0,'max':100000} for k in ('price_cents','advertising_budget_cents','service_budget_cents','capacity_investment_cents','resilience_budget_cents')}
    bounds['procurement_quantity_orders']={'min':0,'max':3500}
    context=SimpleNamespace(objective='human procurement',action_constraints={'bounds':bounds,'supply_chain_enabled':True,'procurement_quantity_enabled':True})
    result=asyncio.run(FixedActionModelClient(request).generate_decision(context))
    action=result.parsed_output['requested_action']
    assert action['procurement_quantity_orders']==1234
    assert action['primary_supplier_id']=='resilient_supplier'
    assert action['backup_supplier_id']=='economy_supplier'
    assert action['primary_supplier_share_ppm']==600000
    request.procurement_quantity_orders=5000
    result=asyncio.run(FixedActionModelClient(request).generate_decision(context))
    assert result.parsed_output['requested_action']['procurement_quantity_orders']==3500
