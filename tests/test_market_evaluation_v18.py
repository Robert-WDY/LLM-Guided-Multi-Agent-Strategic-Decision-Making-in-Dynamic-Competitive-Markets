from copy import deepcopy
from pathlib import Path
import importlib.util
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evaluation_v18',ROOT/'scripts/evaluate_market_v18.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)

def test_mutual_aid_receipts_audit_and_tampering():
 from game_theory_agent.game_theory.advisor_audit import audit_settlement
 c,e,s=evaluation.initial('recession',2,480001,5);transfers=0
 for _ in range(5):
  aa,extra,_=evaluation.population(c,s,'mutual_aid',{});after=evaluation.step(e,s,aa,extra);audit_settlement(s.to_dict(),after.to_dict())
  if after.to_dict()['strategic_market']['mutual_aid']['last_transfers']:
   transfers+=1;bad=deepcopy(after.to_dict());bad['strategic_market']['mutual_aid']['last_transfers'][0]['total_transfer_fee_cents']+=1
   with pytest.raises(AssertionError,match='mutual aid invoice'):audit_settlement(s.to_dict(),bad)
  s=after
 assert transfers>0

def test_exited_company_portfolios_keep_frozen_action():
 from dataclasses import replace
 from game_theory_agent.market.models import CompanyOperatingStatus
 from game_theory_agent.market.actor_policies import company_action,COMPANY_OPTIONS
 from game_theory_agent.market.protocols import state_hash
 c,e,s=evaluation.initial('normal',2,480003,5)
 lifecycle=tuple(replace(x,status=CompanyOperatingStatus.EXITED,exit_round=1,exit_reason='evaluation') if x.company_id=='company_A' else x for x in s.strategic_market.company_lifecycle)
 strategic=replace(s.strategic_market,company_lifecycle=lifecycle)
 s=replace(s,strategic_market=strategic);s=replace(s,state_hash=state_hash(s.to_dict()))
 baseline=company_action(c,s,'company_A','balanced')
 for option in COMPANY_OPTIONS:assert company_action(c,s,'company_A',option)==baseline

def test_liquidation_is_asset_conversion_not_operating_profit():
 from game_theory_agent.game_theory.advisor_audit import audit_settlement
 c,e,s=evaluation.initial('normal',5,481001,20);exits=0
 for _ in range(20):
  aa,extra,_=evaluation.population(c,s,'coordination',{});after=evaluation.step(e,s,aa,extra)
  audit_settlement(s.to_dict(),after.to_dict(),c.to_dict())
  if after.strategic_market.active_company_count<s.strategic_market.active_company_count:
   exits+=1;bad=deepcopy(after.to_dict());bad['companies']['company_A']['financial']['cash_balance_cents']+=1
   with pytest.raises(AssertionError,match='company cash'):audit_settlement(s.to_dict(),bad,c.to_dict())
  s=after
 assert exits>0
