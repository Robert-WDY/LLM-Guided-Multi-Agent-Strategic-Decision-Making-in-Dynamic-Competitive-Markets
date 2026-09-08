import pytest
from game_theory_agent.game_theory.lab import GAMES,analyze,exploitability


def test_known_normal_form_solutions():
    pd=analyze(GAMES['prisoners_dilemma']['payoffs'])
    assert pd['pure_nash']==[[1,1]] and len(pd['strictly_dominated'])==2
    assert [0,0] in pd['pareto_cells'] and [1,1] not in pd['pareto_cells']
    pennies=analyze(GAMES['matching_pennies']['payoffs'])
    assert pennies['pure_nash']==[] and pennies['interior_mixed_nash'][0]['exact']==['1/2','1/2']
    assert all(x['guaranteed_payoff']==0 for x in pennies['security'])
    stag=analyze(GAMES['stag_hunt']['payoffs'])
    assert stag['pure_nash']==[[0,0],[1,1]] and stag['interior_mixed_nash'][0]['exact']==['3/4','3/4']


def test_equilibria_have_no_unilateral_gain_and_invalid_matrices_rejected():
    import random
    rng=random.Random(29)
    for _ in range(200):
        m=[[[rng.randint(-5,5) for _ in range(2)] for _ in range(2)] for _ in range(2)]
        r=analyze(m)
        for i,j in r['pure_nash']:assert max(exploitability(m,1-i,1-j))<1e-8
        for s in r['interior_mixed_nash']:assert max(s['deviation_gains'])<1e-8
    for m in ([],[[[1,2]]],[[[float('nan'),0],[0,0]],[[0,0],[0,0]]]):
        with pytest.raises(ValueError):analyze(m)
    assert analyze([[[0,0],[0,0]],[[0,0],[0,0]]])['degenerate']


def test_repeated_policies_and_noise_are_reproducible():
    from game_theory_agent.game_theory.lab import repeated
    assert repeated(row='cooperate',column='cooperate',rounds=10)['average_payoffs']==[3,3]
    r=repeated(row='tit_for_tat',column='defect',rounds=10)
    assert r['average_payoffs']==[.9,1.4] and r['history'][1]['intended']==[1,1]
    args=dict(row='generous_tft',column='grim_trigger',seed=30,noise=.1)
    assert repeated(**args)==repeated(**args)
    assert any(any(h['trembles']) for h in repeated(**args)['history'])


def test_cce_certificate_equals_external_regret():
    from game_theory_agent.game_theory.lab import learning
    for algorithm in ('regret_matching','fictitious_play'):
        r=learning(algorithm=algorithm,seed=31,rounds=3000)
        assert r['cce_deviation_gains']==pytest.approx(r['history'][-1]['external_regret_per_round'])
        assert r['cce_gap']<.15
        assert sum(map(sum,r['joint_distribution']))==pytest.approx(1)


def test_sequential_information_bargaining_and_public_goods():
    from game_theory_agent.game_theory.lab import sequential,information,bargaining,public_goods
    r=sequential();assert any(x['quantities']==[8,8] for x in r['simultaneous_nash'])
    assert max(x['payoffs'][0] for x in r['commitment_solutions'])==72
    assert any(x['leader_quantity']==12 and x['selected_follower']==6 for x in r['commitment_solutions'])
    for prior in (0,.1,.5,.9,1):
        for accuracy in (.5,.8,1):
            r=information(weak_prior=prior,accuracy=accuracy)
            assert -1e-9<=r['value_of_information']<=r['perfect_information_bound']+1e-9
            assert sum(s['probability'] for s in r['signals'])==pytest.approx(1)
    assert bargaining()['solutions'][0]['allocation']==[55,45]
    assert bargaining(surplus=20)['solutions']==[]
    r=public_goods();assert r['individual_marginal_return']<0<r['social_marginal_return']
    assert sum(o['pure_nash'] for o in r['outcomes'])==1


def test_learning_trace_includes_final_round_and_matches_certificate():
    from game_theory_agent.game_theory.lab import learning
    for algorithm in ('regret_matching','fictitious_play'):
        for rounds in (201,2999,3001):
            r=learning(algorithm=algorithm,rounds=rounds,seed=1501)
            last=r['history'][-1]
            assert last['round']==rounds
            assert last['external_regret_per_round']==pytest.approx(r['cce_deviation_gains'])
            assert last['row_first_frequency']==pytest.approx(sum(r['joint_distribution'][0]))
            assert last['column_first_frequency']==pytest.approx(sum(row[0] for row in r['joint_distribution']))


def test_numeric_parameters_reject_booleans_nonfinite_and_fractional_counts():
    from game_theory_agent.game_theory.lab import repeated,learning,sequential,information,bargaining,public_goods
    cases=[(repeated,'rounds'),(repeated,'noise'),(repeated,'seed'),(learning,'rounds'),(learning,'seed'),(sequential,'maximum'),(sequential,'cost'),(information,'win'),(information,'weak_prior'),(bargaining,'surplus'),(bargaining,'weight'),(public_goods,'players'),(public_goods,'endowment')]
    for fn,key in cases:
        for value in (True,False,float('nan'),float('inf'),'1',None):
            with pytest.raises(ValueError):fn(**{key:value})
    for fn,key,value in ((repeated,'rounds',1.5),(learning,'rounds',10.5),(public_goods,'players',3.5),(bargaining,'surplus',10.5),(sequential,'maximum',3.5)):
        with pytest.raises(ValueError):fn(**{key:value})
