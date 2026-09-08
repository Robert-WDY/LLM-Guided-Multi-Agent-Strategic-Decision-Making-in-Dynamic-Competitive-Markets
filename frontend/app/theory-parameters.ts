// Match the Python API defaults when older saved requests omit parameters.
const defaults:Record<string,Record<string,unknown>>={
 advisor:{advisor_version:'v16',goal:'company',phase:'auto',company_id:'company_A',company_count:4,market_rounds:10,horizon:3,scenarios:3,max_candidates:16,step_budget:1800,risk_aversion:.25,discount:.95,cash_floor_cents:0,seed:360101,diagnostics:true,backtest:true},
 matrix:{game:'prisoners_dilemma'},
 repeated:{row:'tit_for_tat',column:'cooperate',rounds:50,noise:0,discount:.95,seed:1},
 learning:{game:'matching_pennies',algorithm:'regret_matching',rounds:2000,seed:1},
 sequential:{intercept:30,cost:6,maximum:24},
 information:{weak_prior:.5,accuracy:.8,win:8,loss:6},
 bargaining:{surplus:100,disagreement_row:20,disagreement_column:10,weight:.5},
 public_goods:{players:4,endowment:10,contribution:4,multiplier:1.6},
 market_matrix:{seed:330101}
};
export function restoreTheoryParameters(kind:string,parameters:Record<string,unknown>):Record<string,unknown>{return {...defaults[kind],...parameters};}
