# v20状态×策略误差隔离

单位均为合成元；误差=预测均值−D权威均值。每例固定原计划和选中候选，8条未来随机路径。

|案例|A误差|B误差|C误差|D误差|A−D配对误差95%路径区间|
|---|---:|---:|---:|---:|---|
|normal-10-521012-welfare|7,749.18|572.05|8,075.80|0.00|[3,187.62, 12,310.75]|
|normal-5-521002-profit|7,598.86|-1,191.51|4,126.17|0.00|[1,155.97, 14,041.75]|
|normal-5-521004-welfare|17,864.46|8,017.76|23,176.21|0.00|[6,206.72, 29,522.21]|
|normal-5-521012-welfare|14,515.13|4,630.47|20,365.98|0.00|[6,255.79, 22,774.48]|
|normal-5-521017-welfare|18,078.90|-3,199.66|4,138.66|0.00|[6,342.91, 29,814.89]|
|normal-5-521018-welfare|21,982.65|1,811.90|23,748.44|0.00|[16,230.00, 27,735.29]|
|normal-5-521019-profit|9,587.12|5,486.07|-1,308.39|0.00|[3,141.83, 16,032.40]|
|price_sensitive-2-521019-welfare|198.97|-24.75|118.59|0.00|[81.46, 316.48]|
|price_sensitive-5-521017-welfare|71,381.75|43,690.31|-15,052.09|0.00|[21,911.21, 120,852.29]|
|price_sensitive-5-521018-profit|19,920.05|12,478.58|20,828.19|0.00|[12,729.64, 27,110.45]|
|price_sensitive-5-521019-profit|16,486.12|1,853.70|9,194.84|0.00|[8,079.95, 24,892.30]|
|project_feasible-5-521004-welfare|17,864.46|8,017.76|23,176.21|0.00|[6,206.72, 29,522.21]|
|project_feasible-5-521012-welfare|14,515.13|4,630.47|20,365.98|0.00|[6,255.79, 22,774.48]|
|project_feasible-5-521017-welfare|18,078.90|-3,199.66|4,138.66|0.00|[6,342.91, 29,814.89]|
|project_feasible-5-521019-profit|9,587.12|5,486.07|-1,308.39|0.00|[3,141.83, 16,032.40]|
|recession-10-521004-profit|2,172.35|1,265.48|2,058.72|0.00|[570.33, 3,774.38]|
|recession-2-521004-welfare|13,618.43|-3,249.25|26,284.09|0.00|[-11,705.67, 38,942.52]|
|scaled-10-521012-welfare|24,016.01|-1,360.53|27,191.91|0.00|[10,655.65, 37,376.36]|
|scaled-10-521016-welfare|6,344.76|10,330.96|12,167.90|0.00|[-8,664.96, 21,354.47]|
|scaled-10-521018-profit|16,243.67|9,922.21|15,661.47|0.00|[10,895.30, 21,592.05]|
|scaled-10-521018-welfare|39,805.97|2,089.98|30,113.97|0.00|[18,556.05, 61,055.89]|
|scaled-2-521001-welfare|24,901.42|18,166.80|12,420.47|0.00|[13,815.46, 35,987.38]|
|scaled-2-521004-welfare|13,555.81|-2,818.63|21,506.87|0.00|[-7,992.05, 35,103.68]|
|scaled-2-521013-profit|-1,707.45|-4,782.46|-3,679.38|0.00|[-4,444.34, 1,029.43]|
|scaled-2-521019-profit|4,828.79|233.95|5,907.19|0.00|[727.68, 8,929.90]|
|scaled-2-521019-welfare|16,850.01|-600.39|16,138.70|0.00|[10,898.90, 22,801.12]|
|scaled-5-521018-welfare|20,218.56|1,115.29|29,464.64|0.00|[1,130.13, 39,306.99]|
|tight_supply-5-521020-welfare|8,000.09|1,302.67|5,899.26|0.00|[5,421.51, 10,578.66]|
|normal-10-521017-welfare|2,704.55|1,638.39|1,690.52|0.00|[-594.21, 6,003.30]|
|normal-5-521001-profit|11,697.22|-853.86|9,787.17|0.00|[9,404.49, 13,989.95]|
|normal-5-521001-welfare|-6,226.22|-3,622.91|-6,485.05|0.00|[-19,083.03, 6,630.60]|
|normal-5-521003-profit|1,563.68|922.36|-97.68|0.00|[-278.99, 3,406.35]|
|price_sensitive-5-521001-welfare|7,497.82|355.44|4,555.05|0.00|[-18,306.96, 33,302.60]|
|price_sensitive-5-521003-profit|1,114.65|783.98|1,833.46|0.00|[-2,369.11, 4,598.42]|
|price_sensitive-5-521006-profit|775.55|-608.52|-8,656.93|0.00|[-3,311.17, 4,862.26]|
|project_feasible-5-521020-welfare|9,811.98|-3,808.43|17,428.54|0.00|[3,080.02, 16,543.93]|
|project_feasible-5-521001-welfare|-6,226.22|-3,622.91|-6,485.05|0.00|[-19,083.03, 6,630.60]|
|project_feasible-5-521001-profit|11,697.22|-853.86|9,787.17|0.00|[9,404.49, 13,989.95]|
|recession-10-521015-profit|1,051.86|99.43|1,177.94|0.00|[323.00, 1,780.73]|
|recession-2-521016-welfare|9,237.20|8,738.06|11,861.90|0.00|[-2,082.44, 20,556.84]|
|scaled-10-521004-welfare|5,529.21|-9,401.07|125.31|0.00|[-9,230.65, 20,289.08]|
|scaled-10-521017-welfare|-352.01|1,169.00|-4,161.76|0.00|[-2,351.94, 1,647.93]|
|scaled-10-521003-profit|-44.85|961.36|-345.42|0.00|[-4,277.08, 4,187.38]|
|scaled-10-521020-welfare|19,592.14|11,323.54|12,241.56|0.00|[9,306.85, 29,877.43]|
|scaled-2-521012-welfare|26,658.61|4,014.23|25,773.00|0.00|[18,296.50, 35,020.71]|
|scaled-2-521004-profit|2,543.29|-6,233.79|3,141.24|0.00|[-2,486.65, 7,573.22]|
|scaled-2-521007-profit|-352.06|-1,526.06|122.52|0.00|[-1,502.42, 798.30]|
|scaled-5-521004-welfare|14,969.69|2,053.18|18,636.28|0.00|[1,187.76, 28,751.62]|
|tight_supply-5-521003-welfare|7,692.22|-1,385.53|9,104.64|0.00|[4,254.09, 11,130.35]|

这些路径区间仅描述该固定案例的模拟期望差，不是总体因果效应或现实风险区间。

## 组件敏感性

|修正组件|平均绝对误差（合成元）|
|---|---:|
|original|14,570.77|
|financial|14,523.79|
|operations|14,618.93|
|brand|12,914.76|
|risk|14,050.48|
|history|14,268.80|
|market|13,354.38|
|supply_cash|14,570.77|

## 逐例预测/权威福利分项

正数表示候选相对原计划增加该项目。外部成本项增加应从总福利扣除；转移收入不是净福利增量。

### normal-10-521012-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|3,531.21|-864.18|4,395.39|
|round_downstream_producer_surplus_cents|745.58|3,598.28|-2,852.70|
|round_upstream_producer_surplus_cents|416.90|-3,612.38|4,029.28|
|round_government_net_budget_cents|-623.44|0.00|-623.44|
|round_stockout_externality_cents|-2,303.78|496.86|-2,800.65|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|6,374.03|-1,375.15|7,749.18|

### normal-5-521002-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-15,418.71|-15,559.43|140.73|
|round_downstream_producer_surplus_cents|8,727.93|23,140.36|-14,412.43|
|round_upstream_producer_surplus_cents|1,503.38|695.80|807.58|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|1,946.35|-2,507.04|4,453.39|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-7,133.75|10,783.76|-17,917.51|

### normal-5-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|4,972.83|-1,816.56|6,789.39|
|round_downstream_producer_surplus_cents|7,235.73|-179.54|7,415.27|
|round_upstream_producer_surplus_cents|474.28|-2,249.36|2,723.64|
|round_government_net_budget_cents|-2,138.08|0.00|-2,138.08|
|round_stockout_externality_cents|-3,202.44|-128.20|-3,074.24|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|13,747.20|-4,117.26|17,864.46|

### normal-5-521012-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|4,228.68|-3,016.95|7,245.63|
|round_downstream_producer_surplus_cents|4,304.87|367.78|3,937.09|
|round_upstream_producer_surplus_cents|463.48|592.82|-129.33|
|round_government_net_budget_cents|384.20|0.00|384.20|
|round_stockout_externality_cents|-3,087.32|-9.76|-3,077.55|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|12,468.55|-2,046.58|14,515.13|

### normal-5-521017-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|6,187.20|2,769.93|3,417.27|
|round_downstream_producer_surplus_cents|39,400.55|31,182.06|8,218.49|
|round_upstream_producer_surplus_cents|820.15|561.61|258.54|
|round_government_net_budget_cents|0.00|-1,425.39|1,425.39|
|round_stockout_externality_cents|-15,843.87|-11,084.66|-4,759.21|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|62,251.77|44,172.87|18,078.90|

### normal-5-521018-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|7,999.14|87.07|7,912.08|
|round_downstream_producer_surplus_cents|5,994.69|-3,845.99|9,840.68|
|round_upstream_producer_surplus_cents|1,199.61|802.08|397.53|
|round_government_net_budget_cents|-267.19|0.00|-267.19|
|round_stockout_externality_cents|-4,221.42|-121.88|-4,099.55|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|19,147.68|-2,834.97|21,982.65|

### normal-5-521019-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-41,997.96|-12,365.02|-29,632.94|
|round_downstream_producer_surplus_cents|21,498.41|8,840.05|12,658.36|
|round_upstream_producer_surplus_cents|-224.68|463.97|-688.65|
|round_government_net_budget_cents|-668.41|0.00|-668.41|
|round_stockout_externality_cents|4,752.48|428.57|4,323.91|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-26,145.12|-3,489.57|-22,655.54|

### price_sensitive-2-521019-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|375.06|291.02|84.05|
|round_downstream_producer_surplus_cents|192.54|105.27|87.26|
|round_upstream_producer_surplus_cents|2.55|0.90|1.65|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-155.36|-129.35|-26.01|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|725.51|526.54|198.97|

### price_sensitive-5-521017-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|53,873.82|-37,111.27|90,985.10|
|round_downstream_producer_surplus_cents|32,271.66|77,895.85|-45,624.18|
|round_upstream_producer_surplus_cents|2,656.77|-5,701.17|8,357.94|
|round_government_net_budget_cents|-1,425.39|-712.69|-712.69|
|round_stockout_externality_cents|-29,636.03|-11,260.45|-18,375.59|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|117,012.91|45,631.16|71,381.75|

### price_sensitive-5-521018-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|13,750.21|15,751.24|-2,001.03|
|round_downstream_producer_surplus_cents|9,917.63|-5,226.68|15,144.30|
|round_upstream_producer_surplus_cents|2,910.62|2,932.33|-21.71|
|round_government_net_budget_cents|-519.78|0.00|-519.78|
|round_stockout_externality_cents|-5,454.98|-5,026.92|-428.07|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|31,513.65|18,483.81|13,029.84|

### price_sensitive-5-521019-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-4,305.72|-13,241.15|8,935.43|
|round_downstream_producer_surplus_cents|2,870.03|3,794.67|-924.65|
|round_upstream_producer_surplus_cents|1,855.86|1,799.51|56.35|
|round_government_net_budget_cents|-37.51|0.00|-37.51|
|round_stockout_externality_cents|-1,052.53|1,966.65|-3,019.18|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|1,435.18|-9,613.62|11,048.80|

### project_feasible-5-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|4,972.83|-1,816.56|6,789.39|
|round_downstream_producer_surplus_cents|7,235.73|-179.54|7,415.27|
|round_upstream_producer_surplus_cents|474.28|-2,249.36|2,723.64|
|round_government_net_budget_cents|-2,138.08|0.00|-2,138.08|
|round_stockout_externality_cents|-3,202.44|-128.20|-3,074.24|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|13,747.20|-4,117.26|17,864.46|

### project_feasible-5-521012-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|4,228.68|-3,016.95|7,245.63|
|round_downstream_producer_surplus_cents|4,304.87|367.78|3,937.09|
|round_upstream_producer_surplus_cents|463.48|592.82|-129.33|
|round_government_net_budget_cents|384.20|0.00|384.20|
|round_stockout_externality_cents|-3,087.32|-9.76|-3,077.55|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|12,468.55|-2,046.58|14,515.13|

### project_feasible-5-521017-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|6,187.20|2,769.93|3,417.27|
|round_downstream_producer_surplus_cents|39,400.55|31,182.06|8,218.49|
|round_upstream_producer_surplus_cents|820.15|561.61|258.54|
|round_government_net_budget_cents|0.00|-1,425.39|1,425.39|
|round_stockout_externality_cents|-15,843.87|-11,084.66|-4,759.21|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|62,251.77|44,172.87|18,078.90|

### project_feasible-5-521019-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-41,997.96|-12,365.02|-29,632.94|
|round_downstream_producer_surplus_cents|21,498.41|8,840.05|12,658.36|
|round_upstream_producer_surplus_cents|-224.68|463.97|-688.65|
|round_government_net_budget_cents|-668.41|0.00|-668.41|
|round_stockout_externality_cents|4,752.48|428.57|4,323.91|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-26,145.12|-3,489.57|-22,655.54|

### recession-10-521004-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-4,867.86|545.93|-5,413.79|
|round_downstream_producer_surplus_cents|8,317.86|-682.31|9,000.17|
|round_upstream_producer_surplus_cents|305.99|269.83|36.16|
|round_government_net_budget_cents|-29.69|0.00|-29.69|
|round_stockout_externality_cents|-1,616.57|-231.74|-1,384.83|
|round_business_exit_externality_cents|0.00|6,108.80|-6,108.80|
|round_total_economic_welfare_cents|5,342.88|-5,743.60|11,086.48|

### recession-2-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|7,675.56|2,603.27|5,072.28|
|round_downstream_producer_surplus_cents|9,856.84|-2,180.17|12,037.01|
|round_upstream_producer_surplus_cents|-235.89|969.04|-1,204.93|
|round_government_net_budget_cents|-5,227.11|43.94|-5,271.05|
|round_stockout_externality_cents|-4,559.14|-1,574.04|-2,985.10|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|16,628.54|3,010.12|13,618.43|

### scaled-10-521012-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|8,200.03|16,181.67|-7,981.63|
|round_downstream_producer_surplus_cents|9,106.85|-17,788.48|26,895.34|
|round_upstream_producer_surplus_cents|830.49|2,100.97|-1,270.48|
|round_government_net_budget_cents|-513.30|0.00|-513.30|
|round_stockout_externality_cents|-5,664.73|1,221.35|-6,886.08|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|23,288.80|-727.20|24,016.01|

### scaled-10-521016-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|10,994.02|17,585.74|-6,591.72|
|round_downstream_producer_surplus_cents|7,199.67|-3,003.87|10,203.54|
|round_upstream_producer_surplus_cents|1,125.80|-1,902.93|3,028.73|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-6,795.34|-7,091.14|295.80|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|26,114.84|19,770.08|6,344.76|

### scaled-10-521018-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-84,390.08|-10,080.64|-74,309.44|
|round_downstream_producer_surplus_cents|86,776.03|21,786.47|64,989.56|
|round_upstream_producer_surplus_cents|-31.78|5,757.64|-5,789.42|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-2,793.04|-3,791.54|998.50|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|5,147.21|21,255.01|-16,107.80|

### scaled-10-521018-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|17,809.39|85.91|17,723.48|
|round_downstream_producer_surplus_cents|10,412.39|-4,676.21|15,088.61|
|round_upstream_producer_surplus_cents|1,125.63|1,084.43|41.20|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-7,116.84|-164.16|-6,952.68|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|36,464.25|-3,341.72|39,805.97|

### scaled-2-521001-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|22,048.62|8,614.63|13,433.99|
|round_downstream_producer_surplus_cents|-8,126.48|-6,872.08|-1,254.40|
|round_upstream_producer_surplus_cents|-260.95|-11,574.45|11,313.50|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-974.90|433.44|-1,408.33|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|14,636.09|-10,265.33|24,901.42|

### scaled-2-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|6,993.21|1,616.27|5,376.93|
|round_downstream_producer_surplus_cents|642.12|-2,790.29|3,432.41|
|round_upstream_producer_surplus_cents|7,725.87|594.83|7,131.04|
|round_government_net_budget_cents|-5,104.20|92.51|-5,196.71|
|round_stockout_externality_cents|-3,808.82|-996.69|-2,812.13|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|14,065.83|510.01|13,555.81|

### scaled-2-521013-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-14,774.29|-12,451.48|-2,322.81|
|round_downstream_producer_surplus_cents|4,704.41|4,388.09|316.32|
|round_upstream_producer_surplus_cents|366.64|1,105.68|-739.04|
|round_government_net_budget_cents|1,527.20|984.29|542.91|
|round_stockout_externality_cents|1,289.63|1,509.87|-220.24|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-9,465.67|-7,483.30|-1,982.37|

### scaled-2-521019-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-7,124.86|-4,320.01|-2,804.85|
|round_downstream_producer_surplus_cents|5,698.88|7,405.23|-1,706.35|
|round_upstream_producer_surplus_cents|500.20|761.67|-261.47|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|137.36|-558.09|695.46|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-1,063.15|4,404.98|-5,468.13|

### scaled-2-521019-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|14,665.57|6,589.79|8,075.78|
|round_downstream_producer_surplus_cents|10,080.36|-1,567.58|11,647.94|
|round_upstream_producer_surplus_cents|893.32|730.92|162.40|
|round_government_net_budget_cents|-6,923.30|0.00|-6,923.30|
|round_stockout_externality_cents|-5,371.21|-1,484.02|-3,887.19|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|24,087.16|7,237.15|16,850.01|

### scaled-5-521018-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|15,531.14|3,799.55|11,731.59|
|round_downstream_producer_surplus_cents|5,127.60|-3,033.65|8,161.26|
|round_upstream_producer_surplus_cents|-3,332.60|-2,523.21|-809.39|
|round_government_net_budget_cents|-3,095.57|0.00|-3,095.57|
|round_stockout_externality_cents|-4,261.50|-30.82|-4,230.68|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|18,492.07|-1,726.49|20,218.56|

### tight_supply-5-521020-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|3,582.12|-336.59|3,918.71|
|round_downstream_producer_surplus_cents|541.48|-455.75|997.23|
|round_upstream_producer_surplus_cents|1,219.55|44.42|1,175.13|
|round_government_net_budget_cents|0.00|-80.78|80.78|
|round_stockout_externality_cents|-1,874.11|-45.86|-1,828.24|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|7,217.26|-782.83|8,000.09|

### normal-10-521017-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|5,007.40|5,593.51|-586.10|
|round_downstream_producer_surplus_cents|2,578.25|1,386.49|1,191.76|
|round_upstream_producer_surplus_cents|1,488.30|383.06|1,105.24|
|round_government_net_budget_cents|1,050.28|0.00|1,050.28|
|round_stockout_externality_cents|-2,918.03|-2,974.67|56.64|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|13,042.28|10,337.73|2,704.55|

### normal-5-521001-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-11,933.01|1,417.49|-13,350.50|
|round_downstream_producer_surplus_cents|52,498.55|43,781.56|8,716.99|
|round_upstream_producer_surplus_cents|3,581.73|4,934.00|-1,352.26|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-17,575.48|-17,815.91|240.44|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|61,722.75|67,948.97|-6,226.22|

### normal-5-521001-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-11,933.01|1,417.49|-13,350.50|
|round_downstream_producer_surplus_cents|52,498.55|43,781.56|8,716.99|
|round_upstream_producer_surplus_cents|3,581.73|4,934.00|-1,352.26|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-17,575.48|-17,815.91|240.44|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|61,722.75|67,948.97|-6,226.22|

### normal-5-521003-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-27,143.71|-13,653.73|-13,489.97|
|round_downstream_producer_surplus_cents|18,361.30|9,086.69|9,274.61|
|round_upstream_producer_surplus_cents|-1,754.50|644.25|-2,398.75|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|1,039.91|-104.33|1,144.24|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-11,576.82|-3,818.47|-7,758.35|

### price_sensitive-5-521001-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|2,562.87|2,852.54|-289.67|
|round_downstream_producer_surplus_cents|41,081.67|35,543.41|5,538.26|
|round_upstream_producer_surplus_cents|6,675.03|7,183.39|-508.36|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-19,508.91|-16,751.32|-2,757.59|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|69,828.48|62,330.66|7,497.82|

### price_sensitive-5-521003-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-8,896.21|-2,349.70|-6,546.51|
|round_downstream_producer_surplus_cents|11,080.62|5,292.16|5,788.46|
|round_upstream_producer_surplus_cents|5,886.23|5,221.43|664.80|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-2,487.28|-1,750.34|-736.94|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|10,557.92|9,914.23|643.69|

### price_sensitive-5-521006-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-12,731.98|-8,859.39|-3,872.59|
|round_downstream_producer_surplus_cents|23,326.48|6,577.91|16,748.57|
|round_upstream_producer_surplus_cents|-900.17|2,433.60|-3,333.77|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-4,111.51|-1,753.11|-2,358.40|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|13,805.85|1,905.23|11,900.62|

### project_feasible-5-521020-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|7,463.93|758.70|6,705.23|
|round_downstream_producer_surplus_cents|3,810.07|1,258.83|2,551.24|
|round_upstream_producer_surplus_cents|1,483.37|1,670.39|-187.01|
|round_government_net_budget_cents|-5.64|1,398.59|-1,404.23|
|round_stockout_externality_cents|-3,878.71|-1,731.96|-2,146.75|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|16,630.44|6,818.46|9,811.98|

### project_feasible-5-521001-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-11,933.01|1,417.49|-13,350.50|
|round_downstream_producer_surplus_cents|52,498.55|43,781.56|8,716.99|
|round_upstream_producer_surplus_cents|3,581.73|4,934.00|-1,352.26|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-17,575.48|-17,815.91|240.44|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|61,722.75|67,948.97|-6,226.22|

### project_feasible-5-521001-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-11,933.01|1,417.49|-13,350.50|
|round_downstream_producer_surplus_cents|52,498.55|43,781.56|8,716.99|
|round_upstream_producer_surplus_cents|3,581.73|4,934.00|-1,352.26|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-17,575.48|-17,815.91|240.44|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|61,722.75|67,948.97|-6,226.22|

### recession-10-521015-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|1,323.40|1,572.24|-248.83|
|round_downstream_producer_surplus_cents|2,479.66|-36.75|2,516.41|
|round_upstream_producer_surplus_cents|144.43|196.71|-52.28|
|round_government_net_budget_cents|0.00|-59.38|59.38|
|round_stockout_externality_cents|-1,075.20|-1,185.63|110.43|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|5,022.69|2,858.45|2,164.24|

### recession-2-521016-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|5,405.83|4,877.45|528.37|
|round_downstream_producer_surplus_cents|6,400.72|-359.75|6,760.47|
|round_upstream_producer_surplus_cents|766.23|996.03|-229.81|
|round_government_net_budget_cents|0.00|-156.81|156.81|
|round_stockout_externality_cents|-3,686.64|-1,665.29|-2,021.35|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|16,259.42|7,022.22|9,237.20|

### scaled-10-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|8,505.11|8,977.50|-472.38|
|round_downstream_producer_surplus_cents|6,258.36|817.94|5,440.42|
|round_upstream_producer_surplus_cents|907.22|736.47|170.75|
|round_government_net_budget_cents|-144.19|0.00|-144.19|
|round_stockout_externality_cents|-5,134.34|-4,599.72|-534.62|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|20,660.85|15,131.63|5,529.21|

### scaled-10-521017-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|14,796.22|16,047.82|-1,251.60|
|round_downstream_producer_surplus_cents|11,142.90|10,205.80|937.10|
|round_upstream_producer_surplus_cents|1,209.79|1,111.01|98.78|
|round_government_net_budget_cents|0.00|305.44|-305.44|
|round_stockout_externality_cents|-7,447.48|-7,278.32|-169.15|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|34,596.38|34,948.39|-352.01|

### scaled-10-521003-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-71,596.42|-66,751.64|-4,844.78|
|round_downstream_producer_surplus_cents|71,003.68|50,217.44|20,786.24|
|round_upstream_producer_surplus_cents|-2,468.30|-6,009.98|3,541.68|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-6,604.86|5,708.54|-12,313.41|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|3,543.83|-28,252.72|31,796.55|

### scaled-10-521020-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|12,046.08|7,000.45|5,045.63|
|round_downstream_producer_surplus_cents|7,830.89|-3,266.22|11,097.10|
|round_upstream_producer_surplus_cents|2,032.71|2,419.92|-387.22|
|round_government_net_budget_cents|-514.42|-750.50|236.07|
|round_stockout_externality_cents|-6,154.16|-2,553.62|-3,600.55|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|27,549.41|7,957.27|19,592.14|

### scaled-2-521012-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|14,724.45|4,600.80|10,123.64|
|round_downstream_producer_surplus_cents|2,291.62|-9,230.35|11,521.97|
|round_upstream_producer_surplus_cents|1,290.39|2,244.90|-954.52|
|round_government_net_budget_cents|-172.19|0.00|-172.19|
|round_stockout_externality_cents|-5,437.58|702.11|-6,139.70|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|23,571.85|-3,086.75|26,658.61|

### scaled-2-521004-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|4,524.16|-12,578.01|17,102.17|
|round_downstream_producer_surplus_cents|-8,181.49|4,608.80|-12,790.29|
|round_upstream_producer_surplus_cents|-413.35|2,148.14|-2,561.49|
|round_government_net_budget_cents|0.00|1,346.42|-1,346.42|
|round_stockout_externality_cents|335.29|-77.86|413.15|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|-4,405.97|-4,396.80|-9.18|

### scaled-2-521007-profit

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|-22,164.58|-28,454.19|6,289.61|
|round_downstream_producer_surplus_cents|27,416.55|21,259.13|6,157.42|
|round_upstream_producer_surplus_cents|-78.14|345.07|-423.22|
|round_government_net_budget_cents|-64.30|0.00|-64.30|
|round_stockout_externality_cents|-3,012.46|1,909.16|-4,921.62|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|8,121.99|-8,759.15|16,881.14|

### scaled-5-521004-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|13,903.77|-812.57|14,716.34|
|round_downstream_producer_surplus_cents|3,370.81|2,429.57|941.24|
|round_upstream_producer_surplus_cents|942.15|1,486.25|-544.10|
|round_government_net_budget_cents|536.90|0.00|536.90|
|round_stockout_externality_cents|-1,620.58|-2,301.28|680.69|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|20,374.21|5,404.53|14,969.69|

### tight_supply-5-521003-welfare

|福利分项|A预测增量|D权威增量|误差|
|---|---:|---:|---:|
|round_consumer_surplus_cents|5,032.08|-740.60|5,772.68|
|round_downstream_producer_surplus_cents|-432.72|450.94|-883.66|
|round_upstream_producer_surplus_cents|903.61|59.81|843.81|
|round_government_net_budget_cents|0.00|0.00|0.00|
|round_stockout_externality_cents|-1,844.87|114.53|-1,959.40|
|round_business_exit_externality_cents|0.00|0.00|0.00|
|round_total_economic_welfare_cents|7,347.84|-344.38|7,692.22|
