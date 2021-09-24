import pandas as pd
import numpy as np
import pyfolio as pf

from functools import reduce

# 사용할 변수들 클래스 (얼마든지 변경 가능)
class Variables:

    # 자산별로 동일하게 적용되는 한 포지션 진입 시 홀딩 기간 (int)
    holding_period = 60

    # 전체 포트폴리오 리밸런싱 주기 (int)
    rebalancing_period = 1

    # 팩터별 횡적 리스크 계산 어떤 종류 사용할 것인지 (string)
    # -> 종류 (추가가능): 'equal_weight', 'equal_marginal_volatility', 'rank'
    cs_factor_type = 'equal_marginal_volatility'

    # cs_factor_type 'rank' 방식 사용시 롱/숏별로 리밸런싱 시 진입할 총 자산 개수 (int)
    n_selection = 30

    # cs_factor_type 'rank' 방식 사용시 선정된 자산 대한 배분 방식 (string)
    # -> 종류는 cs_factor_type에서 'rank' 제외한 모든 방식 가능
    rank_cs_next_type = 'equal_weight'

    # 종적 리스크 계산 어떤 종류 사용할 것인지 (string)
    # -> 종류 (추가가능): 'volatility_targeting', 'fixed_weighted'
    ts_weight_type = 'volatility_targeting'

    # 종적 리스크 volatility targeting 사용시 목표로 하는 변동성 (float)
    target_vol = 0.05

    # 종적 리스크 volatility targeting 사용시 적용 안 할 백테스팅 기간의 앞부분 일수 (int)
    # -> RiskEngine 클래스 내의 해당 함수 참고
    no_ts_days = 300

    # 종적 리스크 fixed weighted 사용시 설정하는 고정 포트폴리오 대 현금 비중
    # 예) 1 : 포폴에 전부 투자, 0.6 : 40% 현금보유, 2: 레버리지 2배로 투자
    fixed_portfolio_weight = 1

    # 1회 매매시 거래비용 (포지션 진입시 적용 - 홀딩 기간 안에서는 미적용)
    cost = 0.003

    # 롱만 사용할 것인지, 롱/숏 둘 다 사용할 것인지 (boolean)
    long_only = True

# 횡적, 종적 리스크 계산 클래스
class RiskEngine:

    # DataFrame 단위의 참, 거짓 값을 1, 0의 정수로 전환시켜주는 유틸리티 함수
    def bool_calc_util(self, bool_var):
        '''
        Input: boolean (True or False)
        Output: boolean (1 or 0)
        '''
        if bool_var == True:
            result = 1
        elif bool_var == False:
            result = 0
        return result

    # 홀딩 기간만큼의 수익률 계산 함수 (거래비용 포함)
    def get_holding_returns(self, prices, holding_period):
        '''
        Input: 일별 자산별 가격(DataFrame), 포지션 홀딩 일수(int)
        Output: 포지션 홀딩 일수에 해당하는 일별 자산별 수익률(DataFrame)
            -> Output의 경우 holding_period만큼의 수익률을 포지션 진입 일자로 당겨서 처리
        '''
        holding_returns = prices.pct_change(periods=holding_period).shift(-holding_period).fillna(0) - Variables.cost
        return holding_returns

    # 횡적 리스크 계산 함수 - equal weight 방식
    def cs_weight_equal_weight(self, signal):
        '''
        :param signal: equal_weight으로 횡적 리스크 계산 대상이 되는 매매시그널 (DataFrame)
        :return: equal_weight 횡적 리스크 계산된 매매시그널 (DataFrame)
        '''
        total_signal = 1 / abs(signal).sum(axis=1)
        total_signal.replace([np.inf, -np.inf], 0, inplace=True)
        weight = pd.DataFrame(index=signal.index, columns=signal.columns).fillna(value=1)
        weighted_signal = weight.mul(total_signal, axis=0)
        return weighted_signal

    # 횡적 리스크 계산 함수 - equal marginal volatility 방식
    def cs_weight_equal_marginal_volatility(self, returns, signal=None):
        '''
        :param returns: 홀딩 기간 수익률 (DataFrame)
        :param signal: equal_marginal_volatility로 횡적 리스크 계산 대상이 되는 매매시그널 (DataFrame)
        :return: equal_marginal_volatility 횡적 리스크 계산된 매매시그널 (DataFrame)
        '''
        vol = (returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        if signal is None:
            vol_signal = vol
        else:
            vol_signal = vol * abs(signal)
        inv_vol = 1 / vol_signal
        inv_vol.replace([np.inf, -np.inf], 0, inplace=True)
        weighted_signal = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
        return weighted_signal

    # 횡적 리스크 계산 함수 - rank 방식 (팩터들에 걸쳐 계산할때만 해당 됨)
    def cs_weight_rank(self, factor_ranks):
        '''
        :param factor_ranks: 팩터별 일별 자산별 rank 순위 데이터프레임들 리스트 (list of DataFrames)
        :return: 팩터 총합 rank 통한 매매 시그널 (DataFrame)
        '''
        summed_ranks = reduce(lambda x, y: x.add(y, fill_value=0), factor_ranks)
        final_ranks = summed_ranks.rank(axis=1, ascending=False)
        long_signal = (final_ranks <= Variables.n_selection).applymap(self.bool_calc_util)
        short_signal = -(final_ranks >= len(final_ranks.columns) - Variables.n_selection + 1).applymap(
            self.bool_calc_util)
        if Variables.long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
        return signal

    # 종적 리스크 계산 함수 - volatility targeting 방식
    def ts_weight_volatility_targeting(self, backtest_returns, target_vol=0.05, no_ts_days=300):
        '''
        Input: 종적리스크 도입 전 백테스트 수익률(DataFrame), 목표 연간 변동성(float),
            처음 no_ts_days (int) 일수만큼은 반영 x (1년 지나서 가중치 계산 시작시 발생하는 이상치 방지 위함, 매매빈도따라 조절)
        Output: 종적리스크관리 일별 가중치들(DataFrame)
            -> shift(1)로 하루 당겨서 look-ahead bias 방지
        '''
        weights = target_vol / (backtest_returns.rolling(252).std() * np.sqrt(252)).fillna(0)
        weights.replace([np.inf, -np.inf], 0, inplace=True)
        weights = weights.shift(1).fillna(0)
        weights[:no_ts_days] = 0
        return weights

# 백테스트 클래스
class MultiFactorBacktest(RiskEngine):

    # 초기화 함수 - 백테스트 전체 실행 트리거 함수
    def __init__(self, *factors, multi, prices):
        '''
        ---- 클래스 Input Layout ----
        :param factors: 팩터 인스턴스 튜플 (tuple of classes)
        :param multi: 멀티 팩터 여부 (boolean)
        :param prices: 매매기준 일별 자산별 가격 (DataFrame) - 보통 Close나 Adj. Close 값 사용
        '''
        # 홀딩 기간만큼의 수익률 (자산별로 포지션 진입 일에 기록)
        self.holding_returns = self.get_holding_returns(prices, Variables.holding_period)

        # 리밸런싱 가중치: 리밸런싱 주기 / 홀딩기간으로 나누어서 리밸런싱 주기별 포지션 진입
        self.rebalance_weight = Variables.rebalancing_period / Variables.holding_period

        # 팩터별 횡적 리스크 계산 (final signal)
        self.factors = factors
        if multi == True:
            self.final_signal = self.factors_cs_weights_calc()
        else:
            self.final_signal = self.factors[0].factor_weighted_signal

        # 거래비용 및 팩터별 횡적 리스크 반영 백테스트 수익률 계산
        self.port_returns_bf_ts = self.backtest(self.holding_returns, self.rebalance_weight, self.final_signal)
        
        # 종적 리스크 계산
        if Variables.ts_weight_type == 'volatility_targeting':
            self.ts_weights = self.ts_weight_volatility_targeting(self.port_returns_bf_ts, target_vol=Variables.target_vol, no_ts_days=Variables.no_ts_days)
        elif Variables.ts_weight_type == 'fixed_weighted':
            self.ts_weights = Variables.fixed_portfolio_weight

        # 종적 리스크 포함 최종 포트폴리오 수익률 계산
        self.portfolio_returns = self.port_returns_bf_ts * self.ts_weights

        # 성과 분석 결과 시각화
        self.performance_analysis(self.portfolio_returns)

    # 팩터들에 걸친 횡적 리스크 계산 (final signals)
    def factors_cs_weights_calc(self):
        '''
        :return: 팩터별 최종 횡적 리스크 분산이 끝난 일별 자산별 final signal (DataFrame)
        '''
        if Variables.cs_factor_type == 'rank':
                ranked_signal = self.cs_weight_rank([f.ranks for f in self.factors])
                if Variables.rank_cs_next_type == 'equal_weight':
                    final_signal = self.cs_weight_equal_weight(ranked_signal)
                elif Variables.rank_cs_next_type == 'equal_marginal_volatility':
                    final_signal = self.cs_weight_equal_marginal_volatility(self.holding_returns, ranked_signal)
        else:
            factors_signals = [f.factor_weighted_signal for f in self.factors]
            factors_returns = [self.backtest(self.holding_returns, self.rebalance_weight, f) for f in factors_signals]
            if Variables.cs_factor_type == 'equal_weight':
                final_signal = reduce(lambda x, y: x.add(y, fill_value=0), factors_signals) / len(factors_signals)
            elif Variables.cs_factor_type == 'equal_marginal_volatility':
                factor_weights = self.cs_weight_equal_marginal_volatility(pd.concat(factors_returns, axis=1))
                final_signal = factors_signals[0].mul(factor_weights[[0]].values, axis=0)
                for i in range(1, len(factors_signals)):
                    final_signal += factors_signals[i].mul(factor_weights[[i]].values, axis=0)
        return final_signal

    # 종적 리스크 도입 전 백테스트 수익률 계산 함수
    def backtest(self, holding_returns, rebalance_weight, final_signal):
        '''
        Input:
            (1) holding_returns(DataFrame): 홀딩 기간에 따른 일별 자산별 수익률
            (2) rebalance_weight(float): 리밸런싱 주기에 따른 일별 한 자산에의 분산매매 비중
            (3) final_signal(DataFrame): 자산별 팩터별 횡적 리스크 모두 계산된 최종 일별 자산별 매매시그널
        Output:
            종적리스크 도입 전 백테스트 결과 일별 수익률 (DataFrame)
        '''
        port_returns_bf_ts = (final_signal * holding_returns * rebalance_weight).sum(axis=1)
        return port_returns_bf_ts

    # 최종 백테스트 성과 시각화 함수
    def performance_analysis(self, final_backtest_returns):
        '''
        Input: 최종 백테스트 성과 일별 수익률(DataFrame)
        Output: 없음 - 시각화가 목적
        '''
        pf.create_returns_tear_sheet(final_backtest_returns)

# Factor별 시그널 및 횡적 리스크 계산 & 최종 수익률 도출 클래스
class Factor(RiskEngine):
    '''
    <계산 목표>
    1. self.ranks: 팩터 조건들 따라 선정된 일별 자산별 최종 순위
    2. self.factor_signal: rank 순으로 선정한 일별 자산별 매매시그널
    3. self.cs_assets_weights: 일별로 factor_signal에 의해 선정된 자산들의 각각 가중치
    4. self.factor_weighted_signal = self.factor_signal * self.cs_assets_weights
    '''

    def __init__(self, *df_list, ascending_bool_list, prices):
        '''
        :param df_list: 필요한 정보 담긴 데이터프레임들의 튜플 (tuple of DataFrames)
        :param ascending_bool_list: 각 df마다 ranking ascending 방식인지 구별 (list of booleans)
        :param prices: 매매기준 일별 자산별 가격 (DataFrame) - 보통 Close나 Adj. Close 값 사용
        함수기능: 전체실행 용도
        '''
        self.df_list = df_list
        self.ranks = self.calc_ranks(ascending_bool_list)
        if Variables.cs_factor_type == 'rank':
            return
        self.factor_signal = self.calc_factor_signal(self.ranks)
        self.cs_assets_weights = self.calc_cs_assets_weights(prices, self.factor_signal)
        self.factor_weighted_signal = self.calc_factor_weighted_signal(self.factor_signal, self.cs_assets_weights)

    # (1) 합산 rank 계산 함수
    def calc_ranks(self, ascending_bool_list):
        '''
        :param df_list: 필요한 정보 담긴 데이터프레임들의 튜플 (tuple of DataFrames)
        :param ascending_bool_list: 각 df마다 ranking ascending 방식인지 구별 (list of booleans)
            -> ascending=True: 값 높을수록 rank 숫자 값이 높음
            -> ascending=False: 값 높을수록 rank 숫자 값이 낮음
        :return: 합산 rank (DataFrame)
        '''
        rank_df_list = []
        for i in range(len(self.df_list)):
            rank_df = self.df_list[i].rank(axis=1, ascending=ascending_bool_list[i])
            rank_df_list.append(rank_df)
        ranks = reduce(lambda x, y: x.add(y, fill_value=0), rank_df_list)
        return ranks

    # (2) 팩터 시그널 계산 함수
    def calc_factor_signal(self, total_rank_df):
        '''
        :param total_rank_df: 합산 rank (DataFrame)
        :return: 팩터 시그널 (DataFrame)
        '''
        rank_on_rank = total_rank_df.rank(axis=1, ascending=False)
        long_signal = (rank_on_rank <= Variables.n_selection).applymap(self.bool_calc_util)
        short_signal = -(rank_on_rank >= len(rank_on_rank.columns) - Variables.n_selection + 1).applymap(self.bool_calc_util)
        
        if Variables.long_only == True:
            signal = long_signal
        else:
            signal = long_signal + short_signal
            
        # 리밸런싱 하는 일에만 시그널 적용
        rebalance_bool = pd.Series(index=signal.index, dtype='float64').fillna(0)
        for i in range(len(signal.index)):
            if i % Variables.rebalancing_period == 0:
                rebalance_bool[i] = 1
        
        signal = signal.mul(rebalance_bool, axis=0)
        
        return signal

    # (3) 자산별 횡적 리스크 계산 함수
    def calc_cs_assets_weights(self, prices, factor_signal):
        '''
        :param factor_signal: 팩터 시그널 (DataFrame)
        :return: 자산별 횡적 리스크 (DataFrame)
        '''
        if Variables.cs_factor_type == 'equal_weight':
            cs_assets_weights = self.cs_weight_equal_weight(factor_signal)
        elif Variables.cs_factor_type == 'equal_marginal_volatility':
            holding_returns = self.get_holding_returns(prices, Variables.holding_period)
            cs_assets_weights = self.cs_weight_equal_marginal_volatility(holding_returns, factor_signal)
        return cs_assets_weights

    # (4) 팩터 자산별 노출도 고려한 시그널 계산 함수
    def calc_factor_weighted_signal(self, factor_signal, cs_assets_weights):
        '''
        :param factor_signal: 팩터 시그널 (DataFrame)
        :param cs_assets_weights: 자산별 횡적 리스크 (DataFrame)
        :return: 팩터 자산별 노출도 고려한 시그널 (DataFrame)
        '''
        factor_weighted_signal = factor_signal * cs_assets_weights
        return factor_weighted_signal

'''
<<추가구현필요>>
-필요한 데이터 긁어오기 (github or GoogleDrive)
-적절한 데이터 전처리 과정
-팩터별 인스턴스 생성
-백테스트 함수 인스턴스 생성으로 전체 실행
'''

##### 가장 중요한 사항 #####
'''
-Factor 클래스에 넣어주는 df들의 사이즈가 모두 동일해야 함
-각 df는 해당지표에 대해 일(행) x 종목(열) 로 이루어져 있어야 
하며 행 & 일의 index들은 모든 df에 걸쳐 순서가 동일해야 함
-각 지표는 종목들에 걸쳐 오름차순 혹은 내림차순으로 점수를 부여할 수 
있게끔 단방향으로 판단가능하게끔 나열되어 있어야 함 (음수 값 처리 중요)
'''

# 사용 예시
if __name__ == "__main__":
   
    prices = pd.read_csv('https://raw.githubusercontent.com/fr32code/backtest_skeleton/main/Adj.%20Close_data.csv')
    prices.index = pd.to_datetime(prices['Date'])
    prices = prices.drop(columns=['Date'])

    df1 = pd.read_csv('https://raw.githubusercontent.com/fr32code/backtest_skeleton/main/Market-Cap_data.csv')
    df1.index = pd.to_datetime(df1['Date'])
    df1 = df1.drop(columns=['Date'])

    df2 = pd.read_csv('https://raw.githubusercontent.com/fr32code/backtest_skeleton/main/PDIVE_data.csv')
    df2.index = pd.to_datetime(df2['Date'])
    df2 = df2.drop(columns=['Date'])

    df3 = pd.read_csv('https://raw.githubusercontent.com/fr32code/backtest_skeleton/main/Price%20to%20Book%20Value_data.csv')
    df3.index = pd.to_datetime(df3['Date'])
    df3 = df3.drop(columns=['Date'])
    
    sizeFactor = Factor(df1, ascending_bool_list=[True], prices=prices)
    
    # 이 경우 데이터 특성을 고려해 df2, df3의 음수 값 점수 부여 용이하게끔 수정
    df_list = [df2, df3]
    for i in range(len(df_list)):
        B1 = df_list[i][df_list[i].columns.tolist()]
        B1[B1 >= 0] = 1
        B1[B1 < 0] = 0
        
        B2 = df_list[i][df_list[i].columns.tolist()]
        B2[B2 >= 0] = 0
        B2[B2 < 0] = 1
        
        max_df = B2.mul(df_list[i].max(axis=1).values, axis=0)
        
        df_list[i] = B1 * df_list[i] + max_df + abs(B2 * df_list[i])
            
    
    valueFactor = Factor(df_list[0], df_list[1], ascending_bool_list=[False, False], prices=prices)
    backtest = MultiFactorBacktest(sizeFactor, valueFactor, multi=True, prices=prices)
