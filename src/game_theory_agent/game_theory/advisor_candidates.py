"""Multidimensional legal candidates with explicit finite-grid audit option."""
import itertools,random
from .advisor_market import candidates,recipe_action,economic_key

def expanded(config,state,cid,limit,seed,method='expanded'):
 rows=[];seen=set()
 def add(recipe,label):
  action=recipe_action(config,state,cid,recipe);key=economic_key(action)
  if key in seen:return
  seen.add(key);rows.append(dict(id='baseline' if not rows else f'expanded-{len(rows)}',label=label,recipe=recipe,action=action.to_dict()))
 add({},'规则基线')
 if method=='grid':
  for price,quantity,reserve in itertools.product((900000,1000000,1100000),(750000,1000000),(False,True)):
   if len(rows)>=limit:break
   add(dict(price=price,quantity=quantity,reserve=reserve),f'网格：价格{price/10000:g}% / 采购{quantity/10000:g}% / 保留现金{reserve}')
  return rows
 # Diverse compound seeds, then every v16 family, then space-filling combinations.
 for price in (900000,1100000,950000,1050000):
  add(dict(price=price,reserve=True),f'价格{price/10000:g}%与现金保全')
 for row in candidates(config,state,cid,48):
  if len(rows)>=max(5,limit//2):break
  add(row['recipe'],row['label'])
 combos=[dict(price=p,quantity=q,supplier_share=s,duration=d,capacity_investment_cents=i) for p,q,s,d,i in itertools.product((850000,1000000,1150000),(600000,1200000),(0,500000,1000000),(1,3),(0,30000))]
 random.Random(seed).shuffle(combos)
 for partner in state.company_ids:
  if partner!=cid:combos.insert(0,dict(aid='offer',partner=partner,price=1000000))
 for recipe in combos:
  if len(rows)>=limit:break
  add(recipe,'采购/投入/伙伴组合')
 return rows[:limit]
